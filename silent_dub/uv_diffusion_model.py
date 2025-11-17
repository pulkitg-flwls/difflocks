"""
UV Map Diffusion Model Utilities
Creates k_diffusion models for 512x512x3 UV maps conditioned on RGB images via DINOv2
"""

import torch
import os
import sys

# Add parent directory to path to import k_diffusion modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import k_diffusion as K


def load_config_from_silent_dub(config_path=None):
    """
    Load config from silent_dub directory.
    
    Args:
        config_path: Optional path to config file. If None, uses config.json in silent_dub directory.
    
    Returns:
        Loaded config dictionary
    """
    if config_path is None:
        # Use config.json from silent_dub directory
        silent_dub_dir = os.path.dirname(__file__)
        config_path = os.path.join(silent_dub_dir, 'config.json')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    # Use k_diffusion's config loader which handles defaults and merging
    config = K.config.load_config(config_path)
    return config


def create_uv_diffusion_model_from_config(config_path=None, config_dict=None, device='cpu'):
    """
    Create UV diffusion model from config.json.
    
    The model expects:
    - Input: UV maps with shape [B, input_channels, H, W] where input_channels is specified in config
    - Conditioning: DINOv2 features via latents_dict parameter
    - Output: Denoised UV maps with shape [B, out_channels, H, W] where out_channels is specified in config
    
    Note: The model internally adds 16 bias channels, so patch_in expects (input_channels + 16) channels.
    
    Args:
        config_path: Optional path to config file. If None, uses config.json in silent_dub directory.
        config_dict: Optional config dict to use instead of loading from file. If provided, config_path is ignored.
        device: Device to create the model on
    
    Returns:
        Model instance (ImageTransformerDenoiserModelV2Conditional)
    """
    # Load config (use provided dict if available, otherwise load from file)
    if config_dict is not None:
        config = config_dict
    else:
        config = load_config_from_silent_dub(config_path)
    
    # Create model using k_diffusion
    model = K.config.make_model(config)
    model = model.to(device)
    
    # Print model configuration for verification
    model_config = config['model']
    out_channels = model_config.get('out_channels', model_config['input_channels'])
    print(f"Created model with:")
    print(f"  - input_channels: {model_config['input_channels']} (model will add 16 bias channels internally)")
    print(f"  - out_channels: {out_channels}")
    print(f"  - input_size: {model_config['input_size']}")
    print(f"  - patch_size: {model_config['patch_size']}")
    
    return model


def initialize_dinov2(device='cpu'):
    """
    Initialize DINOv2 model for feature extraction.
    
    Args:
        device: Device to load DINOv2 model on
    
    Returns:
        (dinov2_model, preprocessor) tuple
    """
    import torchvision.transforms as T
    
    # DINOv2 preprocessing
    image_size = 770  # Multiple of 14 for patch size
    preprocessor = T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    
    # Load DINOv2 model
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
    model.eval()
    model = model.to(device)
    
    return model, preprocessor


def extract_dinov2_features(dinov2_model, preprocessor, rgb_image, device='cpu'):
    """
    Extract DINOv2 features from RGB image.
    
    Args:
        dinov2_model: DINOv2 model instance
        preprocessor: DINOv2 preprocessor (from initialize_dinov2)
        rgb_image: RGB image tensor [B, 3, H, W] (will be resized to 770x770)
        device: Device to run on
    
    Returns:
        dict with 'cls_token' [B, 1024] and 'final_latent' [B, 1024, 55, 55]
    """
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
    from torchvision.utils import save_image
    
    print("=== Testing UV Diffusion Model ===")
    
    # Set precision to avoid FlashAttention warnings
    torch.set_float32_matmul_precision('high')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create model from config
    print("\n--- Creating model from config ---")
    try:
        model = create_uv_diffusion_model_from_config(device=device)
        print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        print(f"Model config: input_channels={model.rgb_condition_config is not None}")
    except Exception as e:
        print(f"Failed to create model: {e}")
        return
    
    # Initialize DINOv2
    print("\n--- Initializing DINOv2 ---")
    dinov2_model, dinov2_preprocessor = initialize_dinov2(device=device)
    
    model.eval()
    
    # Get model config to determine input shape
    config = load_config_from_silent_dub()
    model_config = config['model']
    input_channels = model_config['input_channels']
    input_size = model_config['input_size']
    
    # Create test data
    batch_size = 2
    print(f"\n--- Testing with batch_size={batch_size} ---")
    print(f"Model expects input: [B, {input_channels}, {input_size[0]}, {input_size[1]}]")
    
    # Dummy RGB image (for DINOv2) - will be resized to 770x770
    rgb_image = torch.randn(batch_size, 3, 256, 256).to(device)
    print(f"RGB image shape: {rgb_image.shape}")
    
    # Dummy UV map (noisy input) - match config dimensions
    uv_map = torch.randn(batch_size, input_channels, input_size[0], input_size[1]).to(device)
    print(f"UV map shape: {uv_map.shape}")
    
    # Dummy noise level
    sigma = torch.rand(batch_size).to(device) * 10 + 0.1
    print(f"Sigma shape: {sigma.shape}")
    
    # Extract DINOv2 features
    print("\n--- Extracting DINOv2 features ---")
    dinov2_features = extract_dinov2_features(dinov2_model, dinov2_preprocessor, rgb_image, device)
    latents_dict = {"dinov2": dinov2_features}
    
    print(f"DINOv2 cls_token shape: {dinov2_features['cls_token'].shape}")
    print(f"DINOv2 final_latent shape: {dinov2_features['final_latent'].shape}")
    
    # Forward pass with DINOv2 conditioning
    print("\n--- Testing forward pass with DINOv2 conditioning ---")
    with torch.no_grad():
        with torch.autocast(device_type="cuda" if device == 'cuda' else "cpu", dtype=torch.bfloat16 if device == 'cuda' else torch.float32):
            output, logvar = model(uv_map, sigma, latents_dict=latents_dict)
    
    print(f"Output shape: {output.shape}")
    print(f"Logvar shape: {logvar.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Test unconditional generation
    print("\n--- Testing unconditional generation ---")
    with torch.no_grad():
        with torch.autocast(device_type="cuda" if device == 'cuda' else "cpu", dtype=torch.bfloat16 if device == 'cuda' else torch.float32):
            output_uncond, logvar_uncond = model(uv_map, sigma, latents_dict=None)
    
    print(f"Unconditional output shape: {output_uncond.shape}")
    print(f"Unconditional logvar shape: {logvar_uncond.shape}")
    
    # Test sampling with denoiser wrapper
    print("\n--- Testing sampling with denoiser wrapper ---")
    config = load_config_from_silent_dub()
    model_config = config['model']
    
    sigma_min = model_config['sigma_min']
    sigma_max = model_config['sigma_max']
    sigma_data = model_config['sigma_data']
    num_steps = 50
    
    # Create denoiser wrapper
    denoiser = K.config.make_denoiser_wrapper(config)(model)
    
    # Generate conditionally
    print(f"Generating {batch_size} UV maps conditioned on RGB images...")
    x_start = torch.randn(batch_size, input_channels, input_size[0], input_size[1], device=device) * sigma_max
    sigmas = K.sampling.get_sigmas_karras(num_steps, sigma_min, sigma_max, rho=7.0, device=device)
    
    extra_args_cond = {"latents_dict": latents_dict}
    
    with torch.no_grad():
        with torch.autocast(device_type="cuda" if device == 'cuda' else "cpu", dtype=torch.bfloat16 if device == 'cuda' else torch.float32):
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
        
    except Exception as e:
        print(f"Could not save images: {e}")
    
    print("\n=== Test completed successfully! ===")
    return model, dinov2_model, dinov2_preprocessor


if __name__ == "__main__":
    test_model()
