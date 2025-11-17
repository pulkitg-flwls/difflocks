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

# Add parent directory to path (before importing k_diffusion)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import k_diffusion (local module in difflocks/)
import k_diffusion as K

# Import our modules
from dataloader_gsplat import SilentDubDataset
from uv_diffusion_model import create_uv_diffusion_model_from_config

torch._dynamo.config.optimize_ddp = False

class MaskedUVDenoiser(K.layers.Denoiser):
    """
    Custom Denoiser for masked UV diffusion.
    Only computes loss on the first 3 channels (uv_comp prediction).
    The remaining channels (uv_og, mask) are context, not ground truth.
    """
    def loss(self, input, noise, sigma, **kwargs):
        """
        Compute loss only on first 3 channels (uv_comp).
        Input shape: [B, 7, H, W] = [uv_comp, uv_og, mask_1ch]
        Ground truth: only uv_comp (first 3 channels)
        """
        from k_diffusion import utils
        
        c_skip, c_out, c_in = [utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        c_weight = self.weighting(sigma)
        weight = c_weight
        
        noised_input = input + noise * utils.append_dims(sigma, input.ndim)
        
        step = None
        if 'step' in kwargs:
            step = kwargs['step']
            del kwargs['step']
        result = self.inner_model(noised_input * c_in, sigma, **kwargs)
        clip_feature_embedding = None
        if len(result) == 2:
            model_output, logvar = result
        elif len(result) == 4:
            model_output, multires_output, logvar, clip_feature_embedding = result
        else:
            raise ValueError(f"Unexpected model output length: {len(result)}")
        
        # Extract components from input
        # Input: [B, 7, H, W] = [uv_comp, uv_og, mask_1ch]
        input_gt = input[:, :3, :, :]  # [B, 3, H, W] - uv_comp
        noised_input_gt = noised_input[:, :3, :, :]  # [B, 3, H, W]
        model_output_gt = model_output[:, :3, :, :]  # [B, 3, H, W]
        noise_gt = noise[:, :3, :, :]  # [B, 3, H, W] - actual noise that was added
        uv_og = input[:, 3:6, :, :]  # [B, 3, H, W]
        mask_1ch = input[:, 6:7, :, :]  # [B, 1, H, W]
        
        # Get denoised prediction (uv_pred) - first 3 channels
        if self.parametrization == "v":
            uv_pred = model_output_gt.to(torch.float32) * c_out + noised_input_gt * c_skip
        else:  # x0 parametrization
            uv_pred = model_output_gt.to(torch.float32)
        
        # Compute predicted noise from model output
        # noised_input = input + noise * sigma
        # So: predicted_noise = (noised_input - predicted_clean) / sigma
        sigma_expanded = utils.append_dims(sigma, input_gt.ndim)  # [B, 1, 1, 1]
        predicted_noise = (noised_input_gt - uv_pred) / sigma_expanded  # [B, 3, H, W]
        
        # Losses:
        # 1. Diffusion loss: compare predicted noise with actual noise
        #    ||predicted_noise - actual_noise||²
        diffusion_loss = ((predicted_noise - noise_gt) ** 2).flatten(1).mean(1)  # [B]
        
        # 2. Noise loss: ensure unmasked region matches uv_og
        #    ||uv_og*(1-mask) - uv_pred*(1-mask)||²
        unmasked_uv_og = uv_og * (1 - mask_1ch)  # [B, 3, H, W]
        unmasked_uv_pred = uv_pred * (1 - mask_1ch)  # [B, 3, H, W]
        noise_loss = ((unmasked_uv_og - unmasked_uv_pred) ** 2).flatten(1).mean(1)  # [B]
        
        # 3. TV loss on masked region: uv_pred*mask
        masked_uv_pred = uv_pred * mask_1ch  # [B, 3, H, W]
        # Total Variation: sum of absolute differences between adjacent pixels
        # TV = |x[i+1,j] - x[i,j]| + |x[i,j+1] - x[i,j]|
        tv_h = torch.abs(masked_uv_pred[:, :, 1:, :] - masked_uv_pred[:, :, :-1, :])  # [B, 3, H-1, W]
        tv_w = torch.abs(masked_uv_pred[:, :, :, 1:] - masked_uv_pred[:, :, :, :-1])  # [B, 3, H, W-1]
        tv_loss = (tv_h.sum(dim=(1, 2, 3)) + tv_w.sum(dim=(1, 2, 3))) / (masked_uv_pred.shape[2] * masked_uv_pred.shape[3])  # [B]
        
        # Weight the losses (can be made configurable)
        diffusion_loss_weight = 1.0
        noise_loss_weight = 1.0
        tv_loss_weight = 0.1
        total_loss = diffusion_loss_weight * diffusion_loss + noise_loss_weight * noise_loss + tv_loss_weight * tv_loss
        
        # Return losses: (total_loss, diffusion_loss, noise_loss, tv_loss)
        return total_loss, diffusion_loss, noise_loss, tv_loss

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
    p.add_argument('--template-dir', type=str, default=None,
                   help='Template directory with gsplat.npy and fotd.png (optional)')
    p.add_argument('--out-channels', type=int, default=None,
                   help='Number of output channels (overrides config.json, defaults to input_channels if not specified)')
    args = p.parse_args()

    return args

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
    # Override out_channels from CLI argument if provided
    if args.out_channels is not None:
        model_config['out_channels'] = args.out_channels
        print(f'Overriding out_channels from CLI: {args.out_channels}')
    # dataset_config is required by k_diffusion config loader but not used for SilentDubDataset
    # dataset_config is not used for UV diffusion and is for classifier-free guidance (CFG)
    # SilentDubDataset is created from CLI arguments, not from config
    dataset_config = config.get('dataset', {})  # Get with default to avoid KeyError
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

    # Create UV diffusion model from config using uv_diffusion_model utility
    # This ensures consistent model creation and handles config validation
    # Pass the modified config dict so CLI overrides are respected
    inner_model = create_uv_diffusion_model_from_config(config_path=args.config, config_dict=config, device=device)
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

    # Optimizer for the UV diffusion model
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

    # Create dataset (DINOv2 extraction is now done inside SilentDubDataset)
    train_set = SilentDubDataset(
        root_dir=args.root_dir,
        json_dir=args.json_dir,
        mask_path=args.mask_path,
        template_dir=args.template_dir,
        device=device,
        open_ratio_threshold=args.open_ratio_threshold,
        load_dinov2=True  # Enable DINOv2 feature extraction in dataset
    )

    if accelerator.is_main_process:
        try:
            print(f'Number of items in dataset: {len(train_set):,}')
        except TypeError:
            pass

    # Note: num_classes and cond_dropout_rate from dataset_config are not used
    # SilentDubDataset handles its own configuration via CLI arguments
    # These are kept for compatibility with k_diffusion framework expectations
    num_classes = dataset_config.get('num_classes', 0)  # Not used for UV diffusion
    # cond_dropout_rate is not used - model uses model_config['condition_dropout_rate'] instead

    # DINOv2 extraction is now handled inside SilentDubDataset.__getitem__
    # If using num_workers > 0, each worker will load its own DINOv2 model
    # This may use more memory but should work correctly
    train_dl = data.DataLoader(train_set, args.batch_size, shuffle=True, drop_last=True,
                               num_workers=args.num_workers, 
                               persistent_workers=args.num_workers > 0, 
                               pin_memory=True)

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

    # Create custom denoiser wrapper for masked UV diffusion
    # Only computes loss on first 3 channels (uv_comp), not on uv_og and mask
    sigma_data = model_config.get('sigma_data', 1.)
    weighting = model_config.get('loss_weighting', 'karras')
    scales = model_config.get('loss_scales', 1)
    parametrization = model_config.get('parametrization', 'v')
    loss_weight_per_channel = model_config.get('loss_weight_per_channel', None)
    
    model = MaskedUVDenoiser(
        inner_model,
        sigma_data=sigma_data,
        weighting=weighting,
        scales=scales,
        parametrization=parametrization,
        loss_weight_per_channel=loss_weight_per_channel
    )
    model_ema = MaskedUVDenoiser(
        inner_model_ema,
        sigma_data=sigma_data,
        weighting=weighting,
        scales=scales,
        parametrization=parametrization,
        loss_weight_per_channel=loss_weight_per_channel
    )

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
        # CFG (Classifier-Free Guidance) function - not used for UV diffusion (num_classes=0)
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
    def sample_images(nr_images, uv_og_batch, latents_dict_batch, mask_1ch_batch):
        """
        Sample images conditionally on DINOv2 latents and uv_og.
        
        Args:
            nr_images: Number of images to sample
            uv_og_batch: Closed mouth UV maps [B, 3, H, W]
            latents_dict_batch: DINOv2 latents dict from batch
            mask_1ch_batch: Mask tensor [B, 1, H, W]
        """
        if accelerator.is_main_process:
            tqdm.write('Sampling conditionally with DINOv2 and uv_og...')
        
        # Limit to available batch size
        batch_size = uv_og_batch.shape[0]
        nr_images = min(nr_images, batch_size)
        
        # Use first nr_images samples from batch
        uv_og = uv_og_batch[:nr_images]  # [nr_images, 3, H, W]
        mask_1ch = mask_1ch_batch[:nr_images]  # [nr_images, 1, H, W]
        
        # Prepare latents for conditioning
        latents_dict = {}
        if latents_dict_batch is not None and "dinov2" in latents_dict_batch:
            dinov2_latents = latents_dict_batch["dinov2"]
            # Take first nr_images samples
            latents_dict["dinov2"] = {
                k: v[:nr_images] if isinstance(v, torch.Tensor) else v
                for k, v in dinov2_latents.items()
            }
        
        # Start with noisy_uv_comp = uv_og * (1-mask) + noise * mask (matching training input structure)
        # Input shape: [nr_images, 7, H, W] = [noisy_uv_comp, uv_og, mask_1ch]
        n_per_proc = math.ceil(nr_images / accelerator.num_processes)
        noise_uv = torch.randn([accelerator.num_processes, n_per_proc, 3, size[0], size[1]], generator=demo_gen).to(device)
        dist.broadcast(noise_uv, 0)
        noise_uv = noise_uv[accelerator.process_index]  # [n_per_proc, 3, H, W]
        
        # Distribute uv_og and mask_1ch across processes
        # Pad to ensure we have enough for all processes
        total_needed = accelerator.num_processes * n_per_proc
        if uv_og.shape[0] < total_needed:
            pad_size = total_needed - uv_og.shape[0]
            uv_og = torch.cat([uv_og, uv_og[-1:].repeat(pad_size, 1, 1, 1)], dim=0)
            mask_1ch = torch.cat([mask_1ch, mask_1ch[-1:].repeat(pad_size, 1, 1, 1)], dim=0)
        
        # Split across processes
        start_idx = accelerator.process_index * n_per_proc
        end_idx = start_idx + n_per_proc
        uv_og_proc = uv_og[start_idx:end_idx].to(device)  # [n_per_proc, 3, H, W]
        mask_1ch_proc = mask_1ch[start_idx:end_idx].to(device)  # [n_per_proc, 1, H, W]
        
        # Construct noisy_uv_comp: uv_og in unmasked region, noise in masked region
        # This matches training: uv_comp = uv_og * (1-mask) + template * mask
        # At test time: noisy_uv_comp = uv_og * (1-mask) + noise * mask (at sigma_max)
        noise_uv_masked = noise_uv * mask_1ch_proc  # [n_per_proc, 3, H, W] - noise only in masked region
        noisy_uv_comp = uv_og_proc * (1 - mask_1ch_proc) + noise_uv_masked * sigma_max  # [n_per_proc, 3, H, W]
        
        # Create initial x: [noisy_uv_comp, uv_og, mask_1ch] at sigma_max
        # This matches the training input structure: [uv_comp + noise*sigma, uv_og, mask_1ch]
        x = torch.cat([noisy_uv_comp, uv_og_proc, mask_1ch_proc], dim=1)  # [n_per_proc, 7, H, W]
        
        # Prepare extra_args with DINOv2 conditioning
        extra_args = {}
        if cross_cond and latents_dict:
            # Prepare latents for this process
            proc_latents_dict = {}
            if "dinov2" in latents_dict:
                proc_dinov2 = latents_dict["dinov2"]
                start_idx = accelerator.process_index * n_per_proc
                end_idx = start_idx + n_per_proc
                # Pad latents if needed
                proc_latents_dict["dinov2"] = {}
                for k, v in proc_dinov2.items():
                    if isinstance(v, torch.Tensor):
                        if v.shape[0] >= end_idx:
                            proc_v = v[start_idx:end_idx].to(device)
                        else:
                            # Pad with last element
                            proc_v = v[-1:].repeat(n_per_proc, *([1] * (v.ndim - 1))).to(device)
                        proc_latents_dict["dinov2"][k] = proc_v
                    else:
                        proc_latents_dict["dinov2"][k] = v
            extra_args["latents_dict"] = proc_latents_dict
        
        model_fn = model_ema
        sigmas = K.sampling.get_sigmas_karras(100, sigma_min, sigma_max, rho=7., device=device)
        x_0 = K.sampling.sample_dpmpp_2m_sde(model_fn, x, sigmas, extra_args=extra_args, eta=0.0, solver_type='heun', disable=not accelerator.is_main_process)
        x_0 = accelerator.gather(x_0)[:nr_images]
        
        # Extract only first 3 channels (uv_comp prediction)
        x_0 = x_0[:, :3, :, :]  # [nr_images, 3, H, W]
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
                        # Training setup with masked diffusion (similar to scalp_diffusion):
                        # - uv_og: closed mouth UV map [B, 3, 512, 512]
                        # - template: template teeth UV map [B, 3, 512, 512]
                        # - mask: mask tensor [3, 512, 512] or [B, 3, 512, 512] -> converted to 1 channel
                        # - uv_comp: composite target = uv_og * (1-mask) + template * mask
                        # - reals: input to model = concat([uv_comp, uv_og, mask]) [B, 7, 512, 512]
                        #   Similar to scalp_diffusion where reals = concat([scalp_texture, density_img])
                        uv_og = batch["closed"]["gsplat_params"]["aces_diffuse_alb"]  # [B, 3, 512, 512]
                        template = batch["template_teeth"]["gsplat_params"]["aces_diffuse_alb"]  # [B, 3, 512, 512]
                        mask = batch["mask"]  # [3, 512, 512] or [B, 3, 512, 512]
                        
                        # Ensure mask is batched and on correct device
                        if mask.dim() == 3:
                            mask = mask.unsqueeze(0)  # [1, 3, 512, 512]
                        if mask.shape[0] != uv_og.shape[0]:
                            mask = mask.expand(uv_og.shape[0], -1, -1, -1)  # [B, 3, 512, 512]
                        mask = mask.to(uv_og.device)
                        
                        # Convert mask from 3 channels to 1 channel (take first channel)
                        mask_1ch = mask[:, 0:1, :, :]  # [B, 1, 512, 512]
                        
                        # Create composite target: closed UV with template teeth in masked region
                        # mask_1ch broadcasts with [B, 3, 512, 512] UV maps
                        uv_comp = uv_og * (1 - mask_1ch) + template * mask_1ch  # [B, 3, 512, 512]
                        
                        # Input to model: concatenate [uv_comp, uv_og, mask_1ch] along channel dimension
                        # This is the "clean" version - noise will be added by the denoiser
                        reals = torch.cat([uv_comp, uv_og, mask_1ch], dim=1)  # [B, 7, 512, 512]

                    class_cond, extra_args = None, {}
                    cross_cond = bool(model_config['cross_cond'])
                    if num_classes:
                        # Not used for UV diffusion currently
                        pass
                    if cross_cond:
                        extra_args["latents_dict"] = batch["latents"]
                    
                    # Generate noise matching reals shape [B, 7, 512, 512]
                    # Noise only in first 3 channels (uv_comp), zeros for uv_og and mask channels
                    noise_uv = torch.randn_like(uv_comp)  # [B, 3, 512, 512]
                    noise_uv = noise_uv * mask_1ch  # Only noise in masked region (broadcasts)
                    noise_zeros = torch.zeros_like(uv_og)  # [B, 3, 512, 512] - no noise for uv_og
                    noise_mask = torch.zeros_like(mask_1ch)  # [B, 1, 512, 512] - no noise for mask
                    noise = torch.cat([noise_uv, noise_zeros, noise_mask], dim=1)  # [B, 7, 512, 512]
                    
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
                        # Model receives reals [B, 7, 512, 512] = [uv_comp, uv_og, mask_1ch]
                        # Noise [B, 7, 512, 512] with noise only in first 3 channels (masked region)
                        # Denoiser will compute: noised_input = reals + noise * sigma
                        # Model will receive: noised_input * c_in = [noisy_uv_comp, uv_og, mask_1ch] * c_in
                        # Custom loss function returns: (total_loss, diffusion_loss, noise_loss, tv_loss)
                        losses, diffusion_losses, noise_losses, tv_losses = model.loss(reals, noise, sigma, **extra_args)
                    
                    loss = accelerator.gather(losses).mean().item()
                    diffusion_loss = accelerator.gather(diffusion_losses).mean().item()
                    noise_loss = accelerator.gather(noise_losses).mean().item()
                    tv_loss = accelerator.gather(tv_losses).mean().item()
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
                    K.utils.ema_update_dict(ema_stats, {'diffusion_loss': diffusion_loss}, ema_decay ** (1 / args.grad_accum_steps))
                    K.utils.ema_update_dict(ema_stats, {'noise_loss': noise_loss}, ema_decay ** (1 / args.grad_accum_steps))
                    K.utils.ema_update_dict(ema_stats, {'tv_loss': tv_loss}, ema_decay ** (1 / args.grad_accum_steps))
                    if accelerator.sync_gradients:
                        K.utils.ema_update(model, model_ema, ema_decay)
                        ema_sched.step()

                if step % 25 == 0:
                    loss_disp = sum(losses_since_last_print) / len(losses_since_last_print)
                    losses_since_last_print.clear()
                    avg_loss = ema_stats['loss']
                    avg_diffusion_loss = ema_stats.get('diffusion_loss', 0)
                    avg_noise_loss = ema_stats.get('noise_loss', 0)
                    avg_tv_loss = ema_stats.get('tv_loss', 0)
                    if accelerator.is_main_process:
                        if args.gns:
                            tqdm.write(f'Epoch: {epoch}, step: {step}, loss: {loss_disp:g}, avg_loss: {avg_loss:g}, '
                                     f'diff: {avg_diffusion_loss:g}, noise: {avg_noise_loss:g}, tv: {avg_tv_loss:g}, '
                                     f'gns: {gns_stats.get_gns():g}')
                        else:
                            tqdm.write(f'Epoch: {epoch}, step: {step}, loss: {loss_disp:g}, avg_loss: {avg_loss:g}, '
                                     f'diff: {avg_diffusion_loss:g}, noise: {avg_noise_loss:g}, tv: {avg_tv_loss:g}')

                with torch.no_grad():
                    if use_tensorboard and step % 50 == 0 and accelerator.is_main_process:
                        # Log all losses
                        tensorboard_writer.add_scalar('uv_diffuse/avg_loss', ema_stats['loss'], step)
                        tensorboard_writer.add_scalar('uv_diffuse/avg_diffusion_loss', ema_stats.get('diffusion_loss', 0), step)
                        tensorboard_writer.add_scalar('uv_diffuse/avg_noise_loss', ema_stats.get('noise_loss', 0), step)
                        tensorboard_writer.add_scalar('uv_diffuse/avg_tv_loss', ema_stats.get('tv_loss', 0), step)
                        tensorboard_writer.add_scalar('uv_diffuse/loss', loss, step)
                        tensorboard_writer.add_scalar('uv_diffuse/diffusion_loss', diffusion_loss, step)
                        tensorboard_writer.add_scalar('uv_diffuse/noise_loss', noise_loss, step)
                        tensorboard_writer.add_scalar('uv_diffuse/tv_loss', tv_loss, step)
                        tensorboard_writer.add_scalar('uv_diffuse/lr', opt.param_groups[0]['lr'], step)
                    
                    if use_tensorboard and step % 500 == 0:
                        # Get model prediction for visualization
                        # Use first batch sample for visualization
                        if first_batch is not None:
                            vis_batch = first_batch
                            vis_uv_og = vis_batch["closed"]["gsplat_params"]["aces_diffuse_alb"][:1]  # [1, 3, H, W]
                            vis_template = vis_batch["template_teeth"]["gsplat_params"]["aces_diffuse_alb"][:1]
                            vis_mask = vis_batch["mask"]
                            if vis_mask.dim() == 3:
                                vis_mask = vis_mask.unsqueeze(0)
                            vis_mask_1ch = vis_mask[:1, 0:1, :, :].to(vis_uv_og.device)
                            vis_uv_comp = vis_uv_og * (1 - vis_mask_1ch) + vis_template * vis_mask_1ch
                            
                            # Create input for model
                            vis_reals = torch.cat([vis_uv_comp, vis_uv_og, vis_mask_1ch], dim=1)
                            
                            # Generate noise and get prediction
                            vis_noise_uv = torch.randn_like(vis_uv_comp) * vis_mask_1ch
                            vis_noise_zeros = torch.zeros_like(vis_uv_og)
                            vis_noise_mask = torch.zeros_like(vis_mask_1ch)
                            vis_noise = torch.cat([vis_noise_uv, vis_noise_zeros, vis_noise_mask], dim=1)
                            vis_sigma = sample_density([1], device=device)
                            
                            # Prepare extra_args for visualization (use latents from first batch if available)
                            vis_extra_args = {}
                            if cross_cond and "latents" in vis_batch:
                                vis_extra_args["latents_dict"] = {
                                    k: v[:1] if isinstance(v, torch.Tensor) else v 
                                    for k, v in vis_batch["latents"].items()
                                }
                            
                            # Forward pass to get prediction
                            vis_noised_input = vis_reals + vis_noise * K.utils.append_dims(vis_sigma, vis_reals.ndim)
                            vis_c_skip, vis_c_out, vis_c_in = [K.utils.append_dims(x, vis_reals.ndim) for x in model.get_scalings(vis_sigma)]
                            vis_result = model.inner_model(vis_noised_input * vis_c_in, vis_sigma, **vis_extra_args)
                            if isinstance(vis_result, tuple):
                                vis_model_output = vis_result[0]
                            else:
                                vis_model_output = vis_result
                            # Model outputs 7 channels (same as input), but we only need first 3 channels (uv_comp prediction)
                            vis_model_output_gt = vis_model_output[:, :3, :, :]  # Extract 3 channels from 7-channel output
                            vis_noised_input_gt = vis_noised_input[:, :3, :, :]
                            
                            if model.parametrization == "v":
                                vis_uv_pred = vis_model_output_gt.to(torch.float32) * vis_c_out + vis_noised_input_gt * vis_c_skip
                            else:
                                vis_uv_pred = vis_model_output_gt.to(torch.float32)
                            
                            # Sample images conditionally using sample_images function
                            sampled_imgs_norm = None
                            nr_imgs_sample = 4
                            if first_batch is not None:
                                sample_batch = first_batch
                                sample_uv_og = sample_batch["closed"]["gsplat_params"]["aces_diffuse_alb"]  # [B, 3, H, W]
                                sample_mask = sample_batch["mask"]
                                if sample_mask.dim() == 3:
                                    sample_mask = sample_mask.unsqueeze(0)
                                sample_mask_1ch = sample_mask[:, 0:1, :, :].to(sample_uv_og.device)  # [B, 1, H, W]
                                sample_latents = sample_batch.get("latents", None)
                                
                                sampled_imgs = sample_images(
                                    nr_imgs_sample,
                                    sample_uv_og,
                                    sample_latents,
                                    sample_mask_1ch
                                )  # [nr_imgs_sample, 3, H, W] - output is already 3 channels
                                
                                # Normalize sampled images to [0, 1]
                                sampled_imgs_norm = (sampled_imgs.clamp(-1, 1) + 1) / 2
                            
                            # Normalize images to [0, 1] for visualization (assuming they're in [-1, 1] range)
                            vis_uv_comp_norm = (vis_uv_comp.clamp(-1, 1) + 1) / 2
                            vis_uv_og_norm = (vis_uv_og.clamp(-1, 1) + 1) / 2
                            vis_uv_pred_norm = (vis_uv_pred.clamp(-1, 1) + 1) / 2
                            
                            if accelerator.is_main_process:
                                # Create grid for each image type
                                grid_comp = utils.make_grid(vis_uv_comp_norm, nrow=1, padding=2)
                                grid_og = utils.make_grid(vis_uv_og_norm, nrow=1, padding=2)
                                grid_pred = utils.make_grid(vis_uv_pred_norm, nrow=1, padding=2)
                                
                                tensorboard_writer.add_image('images/uv_comp', grid_comp, step)
                                tensorboard_writer.add_image('images/uv_og', grid_og, step)
                                tensorboard_writer.add_image('images/uv_pred', grid_pred, step)
                                
                                # Create comparison with sampled images
                                if sampled_imgs_norm is not None:
                                    # Use first sampled image for comparison
                                    sampled_first = sampled_imgs_norm[:1]  # [1, 3, H, W]
                                    # Side-by-side: uv_comp, uv_og, uv_pred, sampled
                                    comparison = torch.cat([vis_uv_comp_norm, vis_uv_og_norm, vis_uv_pred_norm, sampled_first], dim=0)
                                    grid_comparison = utils.make_grid(comparison, nrow=4, padding=2)
                                    tensorboard_writer.add_image('images/comparison', grid_comparison, step)
                                    
                                    # Also log all sampled images separately
                                    grid_sampled = utils.make_grid(sampled_imgs_norm, nrow=math.ceil(nr_imgs_sample ** 0.5), padding=2)
                                    tensorboard_writer.add_image('images/sampled_uv', grid_sampled, step)
                                else:
                                    # Fallback if sampling failed
                                    comparison = torch.cat([vis_uv_comp_norm, vis_uv_og_norm, vis_uv_pred_norm], dim=0)
                                    grid_comparison = utils.make_grid(comparison, nrow=3, padding=2)
                                    tensorboard_writer.add_image('images/comparison', grid_comparison, step)

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
