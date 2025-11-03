#!/usr/bin/env python3
"""
Training script for UV Map Diffusion Model
Adapted from train_scalp_diffusion.py to use k_diffusion training infrastructure
Input: RGB images (FOTD) + UV maps (gsplat aces_diffuse_alb)
Output: Denoised UV maps conditioned on RGB images via DINOv2
"""

import argparse
from copy import deepcopy
from functools import partial
import math
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

import accelerate
import safetensors.torch as safetorch
import torch
import torch._dynamo
from torch import distributed as dist
from torch import multiprocessing as mp
from torch import optim
from torch.utils import data
from torchvision import utils
from tqdm.auto import tqdm
import sys
import os
import random

# Import k_diffusion
import k_diffusion as K

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import our modules
from uv_diffusion_model import create_uv_diffusion_model, initialize_dinov2, extract_dinov2_features
from dataloader_gsplat import SilentDubDataset

torch._dynamo.config.optimize_ddp = False

def ensure_distributed():
    if not dist.is_initialized():
        dist.init_process_group(world_size=1, rank=0, store=dist.HashStore())

def get_cli_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--root_dir', required=True, help='Root directory with gsplat and fotd folders')
    p.add_argument('--json_dir', required=True, help='Path to JSON directory with frame scores')
    p.add_argument('--mask_path', required=True, help='Path to mask image')
    p.add_argument('--batch-size', type=int, default=4,
                   help='the batch size')
    p.add_argument('--checkpointing', action='store_true',
                   help='enable gradient checkpointing')
    p.add_argument('--compile', action='store_true',
                   help='compile the model')
    p.add_argument('--config', type=str, required=True,
                   help='the configuration file')
    p.add_argument('--demo-every', type=int, default=500,
                   help='save a demo grid every this many steps')
    p.add_argument('--end-step', type=int, default=None,
                   help='the step to end training at')
    p.add_argument('--gns', action='store_true',
                   help='measure the gradient noise scale (DDP only, disables stratified sampling)')
    p.add_argument('--grad-accum-steps', type=int, default=1,
                   help='the number of gradient accumulation steps')
    p.add_argument('--lr', type=float,
                   help='the learning rate')
    p.add_argument('--mixed-precision', type=str,
                   help='the mixed precision type')
    p.add_argument('--name', type=str, default='uv_diffusion',
                   help='the name of the run')
    p.add_argument('--num-workers', type=int, default=4,
                   help='the number of data loader workers')
    p.add_argument('--reset-ema', action='store_true',
                   help='reset the EMA')
    p.add_argument('--resume', type=str,
                   help='the checkpoint to resume from')
    p.add_argument('--resume-inference', type=str,
                   help='the inference checkpoint to resume from')
    p.add_argument('--save-checkpoints', action='store_true',
                   help='save checkpoints every save-every steps')
    p.add_argument('--save-every', type=int, default=10000,
                   help='save every this many steps')
    p.add_argument('--seed', type=int, default=0,
                   help='the random seed')
    p.add_argument('--start-method', type=str, default='spawn',
                   choices=['fork', 'forkserver', 'spawn'],
                   help='the multiprocessing start method')
    p.add_argument('--use-tensorboard', action='store_true',
                   help='flag to use tensorboard for logging scalars and images')
    p.add_argument('--open-ratio-threshold', type=float, default=0.1,
                   help='Threshold for open/closed classification')
    args = p.parse_args()

    return args

class UVDatasetWithDINOv2(data.Dataset):
    """
    Wrapper dataset that extracts DINOv2 features from FOTD images.
    
    Training setup:
    - Target (reals): open["gsplat_params"]["aces_diffuse_alb"] - open mouth UV map
    - Conditioning: closed["fotd_params"]["ref_img"] - closed RGB image (via DINOv2)
    - Closed UV is available for initialization during inference, not used in training loss
    """
    def __init__(self, base_dataset, dinov2_model, device='cpu'):
        self.base_dataset = base_dataset
        self.dinov2_model = dinov2_model
        self.device = device
        self.dinov2_model.eval()
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        
        # UV maps: 
        # - closed: available for initialization during inference [3, 512, 512]
        # - open: target UV map for diffusion training [3, 512, 512]
        uv_closed = sample["closed"]["gsplat_params"]["aces_diffuse_alb"]  # [3, 512, 512]
        uv_open = sample["open"]["gsplat_params"]["aces_diffuse_alb"]  # [3, 512, 512]
        
        # DINOv2 conditioning: closed RGB reference image
        # Note: fotd_ref is already preprocessed (resized to 770x770 and ImageNet normalized)
        # Input to HDiT: DINOv2 features from closed["fotd_params"]["ref_img"]
        fotd_ref = sample["closed"]["fotd_params"]["ref_img"]  # [3, 770, 770]
        
        # Extract DINOv2 features directly from preprocessed image
        with torch.no_grad():
            rgb_input = fotd_ref.unsqueeze(0).to(self.device)  # [1, 3, 770, 770]
            dinov2_output = self.dinov2_model.forward_features(rgb_input)
            
            # Extract features (same as extract_dinov2_features but without re-preprocessing)
            patch_tok = dinov2_output["x_norm_patchtokens"].clone()
            cls_tok = dinov2_output["x_norm_clstoken"].clone()
            
            # Reshape patch tokens to spatial format
            batch_size, num_patches, hidden_size = patch_tok.shape
            h = w = int(num_patches ** 0.5)  # Should be 55x55 for 770x770 input
            patch_embeddings = patch_tok.reshape(batch_size, h, w, hidden_size)
            patch_embeddings = patch_embeddings.permute(0, 3, 1, 2).contiguous()
            
            dinov2_features = {
                "cls_token": cls_tok.squeeze(0),  # [1024]
                "final_latent": patch_embeddings.squeeze(0)  # [1024, 55, 55]
            }
        
        return {
            "uv_closed": uv_closed,  # Closed UV: used for initialization during inference
            "uv_open": uv_open,      # Open UV: target for diffusion training (reals)
            "latents": {
                "dinov2": dinov2_features  # DINOv2 features from closed RGB image
            }
        }

def main():
    args = get_cli_args()

    mp.set_start_method(args.start_method)
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch._dynamo.config.automatic_dynamic_shapes = False
    except AttributeError:
        pass

    config = K.config.load_config(args.config)
    model_config = config['model']
    dataset_config = config['dataset']
    opt_config = config['optimizer']
    sched_config = config['lr_sched']
    ema_sched_config = config['ema_sched']
    cross_cond = bool(model_config['cross_cond'])

    # TODO: allow non-square input sizes
    assert len(model_config['input_size']) == 2 and model_config['input_size'][0] == model_config['input_size'][1]
    size = model_config['input_size']

    accelerator = accelerate.Accelerator(gradient_accumulation_steps=args.grad_accum_steps, mixed_precision=args.mixed_precision)
    ensure_distributed()
    device = accelerator.device
    unwrap = accelerator.unwrap_model
    print(f'Process {accelerator.process_index} using device: {device}', flush=True)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(f'World size: {accelerator.num_processes}', flush=True)
        print(f'Batch size: {args.batch_size * accelerator.num_processes}', flush=True)

    if args.seed is not None:
        seeds = torch.randint(0, 2 ** 32 - 1, [accelerator.num_processes], generator=torch.Generator().manual_seed(args.seed))
        print("seeds[accelerator.process_index]", seeds[accelerator.process_index])
        torch.manual_seed(seeds[accelerator.process_index])
        np.random.seed(seeds[accelerator.process_index])
        random.seed(seeds[accelerator.process_index])
    demo_gen = torch.Generator().manual_seed(torch.randint(-2 ** 63, 2 ** 63 - 1, ()).item())

    # Initialize DINOv2 model (needed for dataset)
    print("Initializing DINOv2...")
    dinov2_model, dinov2_preprocessor = initialize_dinov2()
    dinov2_model = dinov2_model.to(device)
    dinov2_model.eval()

    # Create UV diffusion model
    inner_model = create_uv_diffusion_model()
    inner_model_ema = deepcopy(inner_model)

    print("args.compile", args.compile)
    if args.compile:
        inner_model.compile()

    if accelerator.is_main_process:
        print(f'Parameters: {K.utils.n_params(inner_model):,}')

    use_tensorboard = args.use_tensorboard
    if use_tensorboard and accelerator.is_main_process:
        from torch.utils.tensorboard import SummaryWriter
        tensorboard_writer = SummaryWriter("tensorboard_logs/"+args.name)

    lr = opt_config['lr'] if args.lr is None else args.lr
    groups = inner_model.param_groups(lr)
    opt = optim.AdamW(groups,
                      lr=lr,
                      betas=tuple(opt_config['betas']),
                      eps=opt_config['eps'],
                      weight_decay=opt_config['weight_decay'])

    if sched_config['type'] == 'inverse':
        sched = K.utils.InverseLR(opt,
                                  inv_gamma=sched_config['inv_gamma'],
                                  power=sched_config['power'],
                                  warmup=sched_config['warmup'])
    elif sched_config['type'] == 'exponential':
        sched = K.utils.ExponentialLR(opt,
                                      num_steps=sched_config['num_steps'],
                                      decay=sched_config['decay'],
                                      warmup=sched_config['warmup'])
    elif sched_config['type'] == 'constant':
        sched = K.utils.ConstantLRWithWarmup(opt, warmup=sched_config['warmup'])
    else:
        raise ValueError('Invalid schedule type')

    assert ema_sched_config['type'] == 'inverse'
    ema_sched = K.utils.EMAWarmup(power=ema_sched_config['power'],
                                  max_value=ema_sched_config['max_value'])
    ema_stats = {}

    # Create base dataset
    base_dataset = SilentDubDataset(
        root_dir=args.root_dir,
        json_dir=args.json_dir,
        mask_path=args.mask_path,
        device=device,
        open_ratio_threshold=args.open_ratio_threshold
    )
    
    # Wrap with DINOv2 feature extraction
    train_set = UVDatasetWithDINOv2(base_dataset, dinov2_model, device=device)

    if accelerator.is_main_process:
        try:
            print(f'Number of items in dataset: {len(train_set):,}')
        except TypeError:
            pass

    num_classes = dataset_config.get('num_classes', 0)
    cond_dropout_rate = dataset_config.get('cond_dropout_rate', 0.1)

    train_dl = data.DataLoader(train_set, args.batch_size, shuffle=True, drop_last=True,
                               num_workers=args.num_workers, persistent_workers=True, pin_memory=True)

    inner_model, inner_model_ema, opt, train_dl = accelerator.prepare(inner_model, inner_model_ema, opt, train_dl)

    if accelerator.num_processes == 1:
        args.gns = False
    if args.gns:
        gns_stats_hook = K.gns.DDPGradientStatsHook(inner_model)
        gns_stats = K.gns.GradientNoiseScale()
    else:
        gns_stats = None
    sigma_min = model_config['sigma_min']
    sigma_max = model_config['sigma_max']
    sample_density = K.config.make_sample_density(model_config)

    # Create denoiser wrapper
    model = K.config.make_denoiser_wrapper(config)(inner_model)
    model_ema = K.config.make_denoiser_wrapper(config)(inner_model_ema)

    state_path = Path(f'{args.name}_state.json')

    if state_path.exists() or args.resume:
        if args.resume:
            ckpt_path = args.resume
        if not args.resume:
            state = json.load(open(state_path))
            ckpt_path = state['latest_checkpoint']
        if accelerator.is_main_process:
            print(f'Resuming from {ckpt_path}...')
        ckpt = torch.load(ckpt_path, map_location='cpu')
        unwrap(model.inner_model).load_state_dict(ckpt['model'])
        unwrap(model_ema.inner_model).load_state_dict(ckpt['model_ema'])
        opt.load_state_dict(ckpt['opt'])
        sched.load_state_dict(ckpt['sched'])
        ema_sched.load_state_dict(ckpt['ema_sched'])
        ema_stats = ckpt.get('ema_stats', ema_stats)
        epoch = ckpt['epoch'] + 1
        step = ckpt['step'] + 1
        if args.gns and ckpt.get('gns_stats', None) is not None:
            gns_stats.load_state_dict(ckpt['gns_stats'])
        demo_gen.set_state(ckpt['demo_gen'])

        del ckpt
    else:
        epoch = 0
        step = 0

    if args.reset_ema:
        unwrap(model.inner_model).load_state_dict(unwrap(model_ema.inner_model).state_dict())
        ema_sched = K.utils.EMAWarmup(power=ema_sched_config['power'],
                                      max_value=ema_sched_config['max_value'])
        ema_stats = {}

    if args.resume_inference:
        if accelerator.is_main_process:
            print(f'Loading {args.resume_inference}...')
        ckpt = safetorch.load_file(args.resume_inference)
        unwrap(model.inner_model).load_state_dict(ckpt)
        unwrap(model_ema.inner_model).load_state_dict(ckpt)
        del ckpt

    cfg_scale = 1.

    def make_cfg_model_fn(model):
        def cfg_model_fn(x, sigma, class_cond):
            x_in = torch.cat([x, x])
            sigma_in = torch.cat([sigma, sigma])
            class_uncond = torch.full_like(class_cond, num_classes)
            class_cond_in = torch.cat([class_uncond, class_cond])
            out = model(x_in, sigma_in, class_cond=class_cond_in)
            out_uncond, out_cond = out.chunk(2)
            return out_uncond + (out_cond - out_uncond) * cfg_scale
        if cfg_scale != 1:
            return cfg_model_fn
        return model

    @torch.no_grad()
    @K.utils.eval_mode(model_ema)
    def sample_images(nr_images):
        if accelerator.is_main_process:
            tqdm.write('Sampling...')
        n_per_proc = math.ceil(nr_images / accelerator.num_processes)
        x = torch.randn([accelerator.num_processes, n_per_proc, model_config['input_channels'], size[0], size[1]], generator=demo_gen).to(device)
        dist.broadcast(x, 0)
        x = x[accelerator.process_index] * sigma_max
        model_fn, extra_args = model_ema, {}
        # For UV diffusion, we can sample unconditionally or conditionally
        # For now, sample unconditionally (no DINOv2 features)
        sigmas = K.sampling.get_sigmas_karras(100, sigma_min, sigma_max, rho=7., device=device)
        x_0 = K.sampling.sample_dpmpp_2m_sde(model_fn, x, sigmas, extra_args=extra_args, eta=0.0, solver_type='heun', disable=not accelerator.is_main_process)
        x_0 = accelerator.gather(x_0)[:nr_images]
        return x_0

    def save():
        accelerator.wait_for_everyone()
        filename = f'{args.name}_{step:08}.pth'
        path_checkpoints_root = os.path.join("./out_training/", args.name)
        os.makedirs(path_checkpoints_root, exist_ok=True)
        filename = os.path.join(path_checkpoints_root, filename)
        if accelerator.is_main_process:
            tqdm.write(f'Saving to {filename}...')
        inner_model = unwrap(model.inner_model)
        inner_model_ema = unwrap(model_ema.inner_model)
        obj = {
            'config': config,
            'model': inner_model.state_dict(),
            'model_ema': inner_model_ema.state_dict(),
            'opt': opt.state_dict(),
            'sched': sched.state_dict(),
            'ema_sched': ema_sched.state_dict(),
            'epoch': epoch,
            'step': step,
            'gns_stats': gns_stats.state_dict() if gns_stats is not None else None,
            'ema_stats': ema_stats,
            'demo_gen': demo_gen.get_state(),
        }
        accelerator.save(obj, filename)
        if accelerator.is_main_process:
            state_obj = {'latest_checkpoint': filename}
            json.dump(state_obj, open(state_path, 'w'))
        config_path = os.path.join(path_checkpoints_root, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=4)

    losses_since_last_print = []

    do_overfit = False
    if do_overfit:
        overfitted_batch = next(iter(train_dl))
    first_batch = None

    try:
        while True:
            for batch in tqdm(train_dl, smoothing=0.1, disable=not accelerator.is_main_process):
                with accelerator.accumulate(model):
                    if do_overfit:
                        batch = overfitted_batch
                    if first_batch is None:
                        first_batch = batch

                    with torch.no_grad():
                        # Training setup:
                        # - reals: open mouth UV map (target) [B, 3, 512, 512]
                        # - Input to HDiT during training: noise + open_UV (via diffusion)
                        # - Conditioning: DINOv2 features from closed RGB image
                        # - Closed UV (batch["uv_closed"]) available but not used in training loss
                        #   (can be used for initialization during inference)
                        reals = batch["uv_open"]  # Target: open mouth UV map [B, 3, 512, 512]

                    class_cond, extra_args = None, {}
                    cross_cond = bool(model_config['cross_cond'])
                    if num_classes:
                        # Not used for UV diffusion currently
                        pass
                    if cross_cond:
                        extra_args["latents_dict"] = batch["latents"]
                    
                    noise = torch.randn_like(reals)
                    
                    # Immiscible diffusion (optional, disabled for now)
                    do_immiscible_diffusion = False
                    if do_immiscible_diffusion:
                        with torch.no_grad():
                            gathered_noise = [torch.zeros_like(noise) for _ in range(accelerator.num_processes)]
                            dist.all_gather(gathered_noise, noise)
                            gathered_noise = torch.cat(gathered_noise, dim=0)
                            distance = torch.linalg.vector_norm(0.10 * reals.to(torch.float16).flatten(start_dim=1).unsqueeze(1) - 0.10 * gathered_noise.to(torch.float16).flatten(start_dim=1).unsqueeze(0), dim=2)
                            gathered_distance = [torch.zeros_like(torch.tensor(distance)) for _ in range(accelerator.num_processes)]
                            dist.all_gather(gathered_distance, torch.tensor(distance))
                            
                            if accelerator.is_main_process:
                                gathered_distance = torch.cat(gathered_distance, dim=0).cpu().numpy()
                                _, col_ind = linear_sum_assignment(gathered_distance)
                                gathered_noise = gathered_noise[col_ind]
                                
                                for process in range(accelerator.num_processes):
                                    start_idx = args.batch_size * process
                                    end_idx = start_idx + args.batch_size
                                    if process == accelerator.process_index:
                                        noise = gathered_noise[start_idx:end_idx].to(accelerator.device)
                                    else:
                                        dist.send(tensor=gathered_noise[start_idx:end_idx].to(accelerator.device), dst=process)
                            else:
                                dist.recv(tensor=noise, src=0)
                            accelerator.wait_for_everyone()

                    with K.utils.enable_stratified_accelerate(accelerator, disable=args.gns):
                        sigma = sample_density([reals.shape[0]], device=device)
                    with K.models.checkpointing(args.checkpointing):
                        losses, singleres_losses, multires_losses, mse_losses = model.loss(reals, noise, sigma, **extra_args)
                    loss = accelerator.gather(losses).mean().item()
                    singleres_loss = accelerator.gather(singleres_losses).mean().item()
                    multires_loss = accelerator.gather(multires_losses).mean().item()
                    mse_loss = accelerator.gather(mse_losses).mean().item()
                    losses_since_last_print.append(loss)
                    accelerator.backward(losses.mean())
                    if args.gns:
                        sq_norm_small_batch, sq_norm_large_batch = gns_stats_hook.get_stats()
                        gns_stats.update(sq_norm_small_batch, sq_norm_large_batch, reals.shape[0], reals.shape[0] * accelerator.num_processes)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), 1.)
                    opt.step()
                    sched.step()
                    opt.zero_grad()

                    ema_decay = ema_sched.get_value()
                    K.utils.ema_update_dict(ema_stats, {'loss': loss}, ema_decay ** (1 / args.grad_accum_steps))
                    K.utils.ema_update_dict(ema_stats, {'singleres_loss': singleres_loss}, ema_decay ** (1 / args.grad_accum_steps))
                    K.utils.ema_update_dict(ema_stats, {'multires_loss': multires_loss}, ema_decay ** (1 / args.grad_accum_steps))
                    K.utils.ema_update_dict(ema_stats, {'mse_loss': mse_loss}, ema_decay ** (1 / args.grad_accum_steps))
                    if accelerator.sync_gradients:
                        K.utils.ema_update(model, model_ema, ema_decay)
                        ema_sched.step()

                if step % 25 == 0:
                    loss_disp = sum(losses_since_last_print) / len(losses_since_last_print)
                    losses_since_last_print.clear()
                    avg_loss = ema_stats['loss']
                    if accelerator.is_main_process:
                        if args.gns:
                            tqdm.write(f'Epoch: {epoch}, step: {step}, loss: {loss_disp:g}, avg loss: {avg_loss:g}, gns: {gns_stats.get_gns():g}')
                        else:
                            tqdm.write(f'Epoch: {epoch}, step: {step}, loss: {loss_disp:g}, avg loss: {avg_loss:g}')

                with torch.no_grad():
                    if use_tensorboard and step % 50 == 0 and accelerator.is_main_process:
                        tensorboard_writer.add_scalar('uv_diffuse/avg_loss', ema_stats['loss'], step)
                        tensorboard_writer.add_scalar('uv_diffuse/avg_mse_loss', ema_stats['mse_loss'], step)
                        tensorboard_writer.add_scalar('uv_diffuse/loss', loss, step)
                        tensorboard_writer.add_scalar('uv_diffuse/lr', opt.param_groups[0]['lr'], step)
                    if use_tensorboard and step % 500 == 0:
                        # Sample images from the model
                        nr_imgs_sample = 4
                        sampled_imgs = sample_images(nr_imgs_sample)
                        if accelerator.is_main_process:
                            # UV maps are 3-channel, visualize directly
                            grid = utils.make_grid(sampled_imgs, nrow=math.ceil(nr_imgs_sample ** 0.5), padding=0)
                            grid = (grid.clamp(-1, 1) + 1) / 2
                            tensorboard_writer.add_image('sampled_uv', grid, step)

                step += 1

                if step == args.end_step or (step > 0 and step % args.save_every == 0) and args.save_checkpoints:
                    save()

                if step == args.end_step:
                    if accelerator.is_main_process:
                        tqdm.write('Done!')
                    return

            epoch += 1
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
