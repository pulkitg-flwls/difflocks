"""
UV Map Diffusion Model - Adapted from DiffLocks DiT
Input: 3-channel UV maps (512x512) conditioned on RGB images via DINOv2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from functools import lru_cache, reduce
import math
from typing import Callable, Union

from einops import rearrange
import sys
import os

# Add parent directory to path to import k_diffusion modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from k_diffusion.models.modules import (
    GlobalTransformerLayer, Level, Linear, LocalCondProj, 
    NeighborhoodTransformerLayer, NoAttentionTransformerLayer, 
    RMSNorm, ShiftedWindowTransformerLayer, TokenMerge, 
    MappingNetwork, TokenSplit, TokenSplitWithoutSkip, 
    downscale_pos, filter_params, tag_module
)
from k_diffusion.models.attention import SpatialTransformerSimpleV2
from k_diffusion import layers
from k_diffusion.models.axial_rope import make_axial_pos
from modules.edm2_modules import MPFourier

# Configuration
@dataclass
class GlobalAttentionSpec:
    d_head: int

@dataclass
class NeighborhoodAttentionSpec:
    d_head: int
    kernel_size: int

@dataclass
class ShiftedWindowAttentionSpec:
    d_head: int
    window_size: int

@dataclass
class NoAttentionSpec:
    pass

@dataclass
class LevelSpec:
    depth: int
    width: int
    d_ff: int
    self_attn: Union[GlobalAttentionSpec, NeighborhoodAttentionSpec, ShiftedWindowAttentionSpec, NoAttentionSpec]
    dropout: float

@dataclass
class MappingSpec:
    depth: int
    width: int
    d_ff: int
    dropout: float

class UVDiffusionModel(nn.Module):
    """
    UV Map Diffusion Model - Adapted from ImageTransformerDenoiserModelV2Conditional
    Input: 3-channel UV maps (512x512) conditioned on RGB images via DINOv2
    """
    
    def __init__(self, levels, mapping, in_channels=3, out_channels=3, patch_size=[4, 4], 
                 input_size=[512, 512], condition_dropout_rate=0.1, rgb_condition_config=None, 
                 num_classes=0, mapping_cond_dim=0, do_multires=False):
        super().__init__()
        self.num_classes = num_classes
        self.do_multires = do_multires
        self.condition_dropout_rate = condition_dropout_rate
        self.rgb_condition_config = rgb_condition_config

        # Patch input - 3 channels + 16 bias channels
        self.patch_in = TokenMerge(in_channels + 16, levels[0].width, patch_size)

        # Time embedding
        self.time_emb = layers.FourierFeatures(1, mapping.width)
        self.time_in_proj = Linear(mapping.width, mapping.width, bias=False)
        self.time_in_proj_only_t = Linear(mapping.width, mapping.width, bias=False)
        self.mapping = tag_module(MappingNetwork(mapping.depth, mapping.width, mapping.d_ff, dropout=mapping.dropout), "mapping")
        self.mapping_only_t = tag_module(MappingNetwork(mapping.depth, mapping.width, mapping.d_ff, dropout=mapping.dropout), "mapping")

        # DINOv2 conditioning
        self.cross_down_layer = nn.ModuleList()
        self.cross_mid_layer = nn.ModuleList()
        self.cross_up_layer = nn.ModuleList()

        # DINOv2 global conditioning
        print("global_condition_shape dim1", rgb_condition_config['global_condition_shape'][1] * 2)
        self.global_latent_encoder = nn.Sequential(
            nn.Linear(rgb_condition_config['global_condition_shape'][1] * 2, mapping.width, bias=True),
        )

        # Local conditioning projection
        self.local_proj = LocalCondProj(
            rgb_condition_config["local_condition_shapes"][0]["shape"][1], 
            rgb_condition_config["cross_condition_dim"][0], 
            mapping.width
        )

        # Transformer levels
        self.down_levels, self.up_levels = nn.ModuleList(), nn.ModuleList()

        for i, spec in enumerate(levels):
            print("initializing LVL ", i, " with width", spec.width)
            
            # Create layer factory based on attention type
            if isinstance(spec.self_attn, GlobalAttentionSpec):
                layer_factory = lambda _: GlobalTransformerLayer(spec.width, spec.d_ff, spec.self_attn.d_head, mapping.width, dropout=spec.dropout)
            elif isinstance(spec.self_attn, NeighborhoodAttentionSpec):
                layer_factory = lambda _: NeighborhoodTransformerLayer(spec.width, spec.d_ff, spec.self_attn.d_head, mapping.width, spec.self_attn.kernel_size, dropout=spec.dropout)
            elif isinstance(spec.self_attn, ShiftedWindowAttentionSpec):
                layer_factory = lambda i: ShiftedWindowTransformerLayer(spec.width, spec.d_ff, spec.self_attn.d_head, mapping.width, spec.self_attn.window_size, i, dropout=spec.dropout)
            elif isinstance(spec.self_attn, NoAttentionSpec):
                layer_factory = lambda _: NoAttentionTransformerLayer(spec.width, spec.d_ff, mapping.width, dropout=spec.dropout)
            else:
                raise ValueError(f'unsupported self attention spec {spec.self_attn}')

            if i < len(levels) - 1:
                # Down and up levels
                self.down_levels.append(Level([layer_factory(i) for i in range(spec.depth)]))
                self.up_levels.append(Level([layer_factory(i + spec.depth) for i in range(spec.depth)]))

                # Cross-attention layers
                d_head = 64
                n_heads = spec.width // d_head
                
                up_condition_dim = rgb_condition_config["cross_condition_dim"][i]
                up_do_self_attn = rgb_condition_config["self_attn"][i]
                down_condition_dim = rgb_condition_config["cross_condition_dim"][i]
                down_do_self_attn = rgb_condition_config["self_attn"][i]

                self.cross_up_layer.append(SpatialTransformerSimpleV2(
                    spec.width, n_heads, d_head, 
                    global_cond_dim=mapping.width,
                    context_dim=up_condition_dim,
                    do_self_attention=up_do_self_attn,
                    dropout=0.0
                ))
                
                self.cross_down_layer.append(SpatialTransformerSimpleV2(
                    spec.width, n_heads, d_head, 
                    global_cond_dim=mapping.width,
                    context_dim=down_condition_dim,
                    do_self_attention=down_do_self_attn,
                    dropout=0.0
                ))

            else:
                # Middle level
                self.mid_level = Level([layer_factory(i) for i in range(spec.depth)])

                # Cross-attention for middle level
                d_head = 64
                n_heads = spec.width // d_head
                mid_condition_dim = rgb_condition_config["cross_condition_dim"][-1]
                mid_do_self_attn = rgb_condition_config["self_attn"][-1]

                self.cross_mid_layer.append(SpatialTransformerSimpleV2(
                    spec.width, n_heads, d_head, 
                    global_cond_dim=mapping.width,
                    context_dim=mid_condition_dim,
                    do_self_attention=mid_do_self_attn,
                    dropout=0.0
                ))

        # Token merge/split operations
        self.merges = nn.ModuleList([TokenMerge(spec_1.width, spec_2.width) for spec_1, spec_2 in zip(levels[:-1], levels[1:])])
        self.splits = nn.ModuleList([TokenSplit(spec_2.width, spec_1.width) for spec_1, spec_2 in zip(levels[:-1], levels[1:])])

        # Output layers
        self.out_norm = RMSNorm(levels[0].width)
        self.patch_out = TokenSplitWithoutSkip(levels[0].width, out_channels, patch_size)

        # Variance prediction
        logvar_channels = 128
        self.logvar_fourier = MPFourier(logvar_channels)
        self.logvar_linear = Linear(logvar_channels, 1, bias=False)
        nn.init.zeros_(self.logvar_linear.weight)

        # Untied bias
        self.untied_bias = nn.Parameter(torch.zeros((1, 16, input_size[0], input_size[1])))

    def param_groups(self, base_lr=5e-4, mapping_lr_scale=1 / 3):
        """Parameter groups for different learning rates"""
        wd = filter_params(lambda tags: "wd" in tags and "mapping" not in tags, self)
        no_wd = filter_params(lambda tags: "wd" not in tags and "mapping" not in tags, self)
        mapping_wd = filter_params(lambda tags: "wd" in tags and "mapping" in tags, self)
        mapping_no_wd = filter_params(lambda tags: "wd" not in tags and "mapping" in tags, self)
        groups = [
            {"params": list(wd), "lr": base_lr},
            {"params": list(no_wd), "lr": base_lr, "weight_decay": 0.0},
            {"params": list(mapping_wd), "lr": base_lr * mapping_lr_scale},
            {"params": list(mapping_no_wd), "lr": base_lr * mapping_lr_scale, "weight_decay": 0.0}
        ]
        return groups

    def forward(self, x, sigma, latents_dict=None, aug_cond=None, class_cond=None, mapping_cond=None, cross_cond=None, cam=None):
        """
        Forward pass
        Args:
            x: [B, 3, 512, 512] - noisy UV map
            sigma: noise level
            latents_dict: DINOv2 features from RGB image
        """
        # Time embedding
        c_noise = torch.log(sigma) / 4
        time_emb = self.time_in_proj_only_t(self.time_emb(c_noise[..., None]))
        noise_cond = self.mapping_only_t(time_emb)

        # DINOv2 conditioning
        with torch.no_grad():
            nr_batches = x.shape[0]
            rand = torch.rand((nr_batches), device=x.device)
            cond_batches_to_drop = (rand < self.condition_dropout_rate) * 1.0
            cond_batches_to_keep = 1.0 - cond_batches_to_drop

            # Prepare conditioning
            global_cond = None
            locals_cond_list = []
            
            if latents_dict is None:
                # Unconditional
                global_cond = torch.zeros((1, self.rgb_condition_config["global_condition_shape"][1] * 2), device=x.device)
                for local_shape in self.rgb_condition_config["local_condition_shapes"]:
                    locals_cond_list.append(torch.zeros(local_shape["shape"], device=x.device))
            else:
                # Use DINOv2 features
                dino_latent_mean = latents_dict["dinov2"]["final_latent"].mean(dim=[2, 3])
                dino_latent_cls = latents_dict["dinov2"]["cls_token"]
                global_cond = torch.cat([dino_latent_mean, dino_latent_cls], 1)
                
                local = latents_dict["dinov2"]["final_latent"].contiguous()
                for local_shape in self.rgb_condition_config["local_condition_shapes"]:
                    locals_cond_list.append(local)

            # Apply condition dropout
            global_cond = global_cond * cond_batches_to_keep.view(nr_batches, 1)
            for l_idx in range(len(locals_cond_list)):
                locals_cond_list[l_idx] = locals_cond_list[l_idx] * cond_batches_to_keep.view(nr_batches, 1, 1, 1)

            # Position embeddings for local conditioning
            locals_pos_list = []
            for local_cond in locals_cond_list:
                pos_img = make_axial_pos(local_cond.shape[-1], local_cond.shape[-2], device=x.device).view(local_cond.shape[-1], local_cond.shape[-2], 2)
                locals_pos_list.append(pos_img)

        # Add bias channels
        x = torch.cat([x, self.untied_bias.repeat(x.shape[0], 1, 1, 1)], dim=1)

        # Patching
        x = x.movedim(-3, -1)
        x = self.patch_in(x)
        pos = make_axial_pos(x.shape[-3], x.shape[-2], device=x.device).view(x.shape[-3], x.shape[-2], 2)

        # Mapping network
        c_noise = torch.log(sigma) / 4
        time_emb = self.time_in_proj(self.time_emb(c_noise[..., None]))
        aug_cond = x.new_zeros([x.shape[0], 9]) if aug_cond is None else aug_cond
        global_cond_embedded = self.global_latent_encoder(global_cond)

        embedding_summed = time_emb + global_cond_embedded
        cond = self.mapping(embedding_summed)

        # Local conditioning projection
        local_cond = locals_cond_list[0]
        local_cond = self.local_proj(local_cond, noise_cond)

        # Hourglass transformer
        skips, poses = [], []
        for i, (down_level, merge) in enumerate(zip(self.down_levels, self.merges)):
            x = down_level(x, pos, cond)

            # Cross-attention with DINOv2 features
            x = x.movedim(-1, -3)
            local_pos = locals_pos_list[i]
            x = self.cross_down_layer[i](x, pos, cond, local_cond, local_pos)
            x = x.movedim(-3, -1)

            skips.append(x)
            poses.append(pos)
            x = merge(x)
            pos = downscale_pos(pos)

        # Middle level
        x = self.mid_level(x, pos, cond)

        # Cross-attention at middle level
        x = x.movedim(-1, -3)
        local_pos = locals_pos_list[-1]
        x = self.cross_mid_layer[0](x, pos, cond, local_cond, local_pos)
        x = x.movedim(-3, -1)

        # Up levels
        for i, (up_level, split, skip, pos) in enumerate(reversed(list(zip(self.up_levels, self.splits, skips, poses)))):
            x = split(x, skip)
            x = up_level(x, pos, cond)

            # Cross-attention
            x = x.movedim(-1, -3)
            local_pos = locals_pos_list[-i-1]
            x = self.cross_up_layer[-i-1](x, pos, cond, local_cond, local_pos)
            x = x.movedim(-3, -1)

        # Unpatching
        x = self.out_norm(x)
        x = self.patch_out(x)
        x = x.movedim(-1, -3)

        # Variance prediction
        logvar = 1.0 + self.logvar_linear(self.logvar_fourier(c_noise)).reshape(-1, 1, 1, 1)

        return x, logvar

def create_uv_diffusion_model():
    """Create UV diffusion model with default configuration"""
    
    # Model configuration adapted for UV maps
    levels = [
        LevelSpec(depth=2, width=1024, d_ff=1024, self_attn=NeighborhoodAttentionSpec(d_head=64, kernel_size=7), dropout=0.0),
        LevelSpec(depth=2, width=2048, d_ff=2048, self_attn=NeighborhoodAttentionSpec(d_head=64, kernel_size=7), dropout=0.0),
        LevelSpec(depth=2, width=3072, d_ff=3072, self_attn=GlobalAttentionSpec(d_head=64), dropout=0.1),
    ]
    
    mapping = MappingSpec(depth=2, width=768, d_ff=1024, dropout=0.0)
    
    # DINOv2 conditioning configuration
    rgb_condition_config = {
        "global_condition_shape": [1, 1024],
        "local_condition_shapes": [
            {"shape": [1, 1024, 55, 55]},
            {"shape": [1, 1024, 55, 55]},
            {"shape": [1, 1024, 55, 55]}
        ],
        "cross_condition_dim": [512, 512, 512],
        "self_attn": [False, True, True]
    }
    
    model = UVDiffusionModel(
        levels=levels,
        mapping=mapping,
        in_channels=3,
        out_channels=3,
        patch_size=[4, 4],  # Larger patches for 512x512
        input_size=[512, 512],
        condition_dropout_rate=0.1,
        rgb_condition_config=rgb_condition_config,
        num_classes=0,
        mapping_cond_dim=0,
        do_multires=False
    )
    
    return model

def initialize_dinov2():
    """Initialize DINOv2 model for feature extraction"""
    import torchvision.transforms as T
    
    # DINOv2 preprocessing (same as original)
    image_size = 770  # Multiple of 14 for patch size
    preprocessor = T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    
    # Load DINOv2 model
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
    model.eval()
    
    return model, preprocessor

def extract_dinov2_features(dinov2_model, preprocessor, rgb_image, device='cuda'):
    """Extract DINOv2 features from RGB image"""
    with torch.no_grad():
        # Preprocess image
        rgb_input = preprocessor(rgb_image).to(device)
        
        # Get DINOv2 features
        dinov2_output = dinov2_model.forward_features(rgb_input)
        
        # Extract features
        patch_tok = dinov2_output["x_norm_patchtokens"].clone()
        cls_tok = dinov2_output["x_norm_clstoken"].clone()
        
        # Reshape patch tokens to spatial format
        batch_size, num_patches, hidden_size = patch_tok.shape
        h = w = int(num_patches ** 0.5)  # Should be 55x55 for 770x770 input
        patch_embeddings = patch_tok.reshape(batch_size, h, w, hidden_size)
        patch_embeddings = patch_embeddings.permute(0, 3, 1, 2).contiguous()
        
        return {
            "cls_token": cls_tok,
            "final_latent": patch_embeddings
        }

def test_model():
    """Test the UV diffusion model with actual DINOv2 features"""
    import k_diffusion as K
    from torchvision.utils import save_image
    
    print("=== Testing UV Diffusion Model ===")
    
    # Set precision to avoid FlashAttention warnings
    torch.set_float32_matmul_precision('high')
    
    # Create model
    model = create_uv_diffusion_model()
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Initialize DINOv2
    print("\nInitializing DINOv2...")
    dinov2_model, dinov2_preprocessor = initialize_dinov2()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dinov2_model = dinov2_model.to(device)
    model = model.to(device)
    model.eval()
    
    # Create dummy data
    batch_size = 2
    
    # Dummy RGB image (for DINOv2)
    rgb_image = torch.randn(batch_size, 3, 770, 770).to(device)  # DINOv2 input size
    print(f"RGB image shape: {rgb_image.shape}")
    
    # Dummy UV map (noisy input)
    uv_map = torch.randn(batch_size, 3, 512, 512).to(device)
    print(f"UV map shape: {uv_map.shape}")
    
    # Dummy noise level
    sigma = torch.rand(batch_size).to(device) * 10 + 0.1
    print(f"Sigma shape: {sigma.shape}")
    
    # Extract actual DINOv2 features
    print("\nExtracting DINOv2 features...")
    dinov2_features = extract_dinov2_features(dinov2_model, dinov2_preprocessor, rgb_image, device)
    latents_dict = {"dinov2": dinov2_features}
    
    print(f"DINOv2 cls_token shape: {dinov2_features['cls_token'].shape}")
    print(f"DINOv2 final_latent shape: {dinov2_features['final_latent'].shape}")
    
    # Forward pass with actual DINOv2 features (using bfloat16 for FlashAttention2)
    print("\nRunning forward pass with DINOv2 features...")
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output, logvar = model(uv_map, sigma, latents_dict=latents_dict)
    
    print(f"Output shape: {output.shape}")
    print(f"Logvar shape: {logvar.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Test unconditional generation
    print("\nTesting unconditional generation...")
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output_uncond, logvar_uncond = model(uv_map, sigma, latents_dict=None)
    
    print(f"Unconditional output shape: {output_uncond.shape}")
    print(f"Unconditional logvar shape: {logvar_uncond.shape}")
    
    # Conditional generation with DINOv2 features
    print("\n=== Testing Conditional Generation ===")
    
    # Diffusion parameters (matching UV model config)
    sigma_min = 1e-2
    sigma_max = 160.0
    sigma_data = 0.3
    num_steps = 50
    
    # Create denoiser wrapper for Karras preconditioning (v-parametrization)
    # Same as train_scalp_diffusion.py uses K.config.make_denoiser_wrapper
    from k_diffusion import layers
    denoiser = layers.Denoiser(
        model, 
        sigma_data=sigma_data,
        weighting='snr',
        parametrization='v'
    )
    
    # Generate conditionally (pass latents_dict via extra_args, like train_scalp_diffusion.py)
    print(f"Generating {batch_size} UV maps conditioned on RGB images...")
    x_start = torch.randn(batch_size, 3, 512, 512, device=device) * sigma_max
    sigmas = K.sampling.get_sigmas_karras(num_steps, sigma_min, sigma_max, rho=7.0, device=device)
    
    # Pass latents_dict via extra_args - Denoiser forwards it to inner model
    extra_args_cond = {"latents_dict": latents_dict}
    
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            x_0_cond = K.sampling.sample_dpmpp_2m_sde(
                denoiser, 
                x_start, 
                sigmas, 
                extra_args=extra_args_cond, 
                eta=0.0, 
                solver_type='heun', 
                disable=False
            )
    
    print(f"Generated conditional UV maps shape: {x_0_cond.shape}")
    print(f"Generated range: [{x_0_cond.min():.3f}, {x_0_cond.max():.3f}]")
    
    # Unconditional generation (latents_dict=None via extra_args)
    print("\nGenerating unconditionally...")
    extra_args_uncond = {"latents_dict": None}
    x_start_uncond = torch.randn(batch_size, 3, 512, 512, device=device) * sigma_max
    
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            x_0_uncond = K.sampling.sample_dpmpp_2m_sde(
                denoiser,
                x_start_uncond,
                sigmas,
                extra_args=extra_args_uncond,
                eta=0.0,
                solver_type='heun',
                disable=False
            )
    
    print(f"Generated unconditional UV maps shape: {x_0_uncond.shape}")
    print(f"Generated range: [{x_0_uncond.min():.3f}, {x_0_uncond.max():.3f}]")
    
    # Save generated samples for visualization
    try:
        os.makedirs("test_samples", exist_ok=True)
        
        # Normalize to [0, 1] for visualization
        def normalize_for_viz(x):
            x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)
            return x_norm
        
        # Save conditional samples
        cond_viz = normalize_for_viz(x_0_cond)
        save_image(cond_viz, "test_samples/conditional_generation.png", nrow=2, normalize=False)
        print("\nSaved conditional samples to test_samples/conditional_generation.png")
        
        # Save unconditional samples
        uncond_viz = normalize_for_viz(x_0_uncond)
        save_image(uncond_viz, "test_samples/unconditional_generation.png", nrow=2, normalize=False)
        print("Saved unconditional samples to test_samples/unconditional_generation.png")
        
    except Exception as e:
        print(f"Could not save images: {e}")
    
    print("\n=== Test completed successfully! ===")
    return model, dinov2_model, dinov2_preprocessor

if __name__ == "__main__":
    test_model()
