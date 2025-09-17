#!/usr/bin/env python3
"""
Training script for UV Map Diffusion Model
Input: RGB images + UV maps with edited teeth regions
Output: Denoised UV maps conditioned on RGB images
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from PIL import Image
import os
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import deepcopy

# Import our modules
from uv_diffusion_model import UVDiffusionModel, create_uv_diffusion_model, initialize_dinov2, extract_dinov2_features
from dataloader import OpenClosedDataset

# Import k_diffusion utilities
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import k_diffusion as K

def tensor_to_pil(tensor):
    """Convert tensor to PIL Image for visualization."""
    # Convert from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to PIL
    if tensor.dim() == 4:
        tensor = tensor[0]
    return T.ToPILImage()(tensor)

def save_comparison(model, dinov2_model, dinov2_preprocessor, dataloader, save_dir, device='cuda', num_samples=3):
    """Save comparison images during training."""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break
                
            # Get batch data
            uv_map = batch["uv"].to(device)  # Edited UV map (input)
            target_uv = batch["combined_uv"].to(device)  # Target UV map
            rgb_image = batch["img"].to(device)  # RGB image for conditioning
            
            # Resize RGB image for DINOv2 (770x770)
            rgb_resized = torch.nn.functional.interpolate(rgb_image, size=(770, 770), mode='bilinear', align_corners=False)
            
            # Extract DINOv2 features
            dinov2_features = extract_dinov2_features(dinov2_model, dinov2_preprocessor, rgb_resized, device)
            latents_dict = {"dinov2": dinov2_features}
            
            # Add some noise to UV map
            noise = torch.randn_like(uv_map) * 0.1
            noisy_uv = uv_map + noise
            
            # Get prediction
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output, logvar = model(noisy_uv, torch.tensor([0.1] * uv_map.shape[0]).to(device), latents_dict=latents_dict)
            
            # Save images
            uv_pil = tensor_to_pil(uv_map)
            target_pil = tensor_to_pil(target_uv)
            output_pil = tensor_to_pil(output)
            rgb_pil = tensor_to_pil(rgb_image)
            
            # Create comparison
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            
            axes[0, 0].imshow(rgb_pil)
            axes[0, 0].set_title('RGB Image (Conditioning)')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(uv_pil)
            axes[0, 1].set_title('Input UV Map (Edited)')
            axes[0, 1].axis('off')
            
            axes[1, 0].imshow(target_pil)
            axes[1, 0].set_title('Target UV Map')
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(output_pil)
            axes[1, 1].set_title('Predicted UV Map')
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'{save_dir}/comparison_{i}.png', dpi=150, bbox_inches='tight')
            plt.close()

def train_model(model, dinov2_model, dinov2_preprocessor, train_loader, val_loader, 
                num_epochs=100, lr=1e-4, device='cuda', save_dir='checkpoints'):
    """Train the UV diffusion model with validation."""
    
    # Set precision for FlashAttention2
    torch.set_float32_matmul_precision('high')
    
    model = model.to(device)
    dinov2_model = dinov2_model.to(device)
    
    # Use parameter groups from the model
    optimizer = optim.AdamW(model.param_groups(lr=lr), weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    
    # Loss function
    criterion = nn.MSELoss()
    
    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for batch_idx, batch in enumerate(train_pbar):
            # Get batch data
            uv_map = batch["uv"].to(device)  # Edited UV map (input)
            target_uv = batch["combined_uv"].to(device)  # Target UV map
            rgb_image = batch["img"].to(device)  # RGB image for conditioning
            
            # Resize RGB image for DINOv2 (770x770)
            rgb_resized = torch.nn.functional.interpolate(rgb_image, size=(770, 770), mode='bilinear', align_corners=False)
            
            # Extract DINOv2 features
            dinov2_features = extract_dinov2_features(dinov2_model, dinov2_preprocessor, rgb_resized, device)
            latents_dict = {"dinov2": dinov2_features}
            
            # Add noise to UV map (this is the noisy input)
            noise = torch.randn_like(uv_map)
            sigma = torch.rand(uv_map.shape[0]).to(device) * 10 + 0.1  # Random noise levels
            noisy_uv = uv_map + noise * sigma.view(-1, 1, 1, 1)
            
            # Forward pass
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output, logvar = model(noisy_uv, sigma, latents_dict=latents_dict)
            
            # Loss (MSE between predicted and target)
            loss = criterion(output, target_uv)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for batch in val_pbar:
                # Get batch data
                uv_map = batch["uv"].to(device)
                target_uv = batch["combined_uv"].to(device)
                rgb_image = batch["img"].to(device)
                
                # Resize RGB image for DINOv2
                rgb_resized = torch.nn.functional.interpolate(rgb_image, size=(770, 770), mode='bilinear', align_corners=False)
                
                # Extract DINOv2 features
                dinov2_features = extract_dinov2_features(dinov2_model, dinov2_preprocessor, rgb_resized, device)
                latents_dict = {"dinov2": dinov2_features}
                
                # Add noise
                noise = torch.randn_like(uv_map)
                sigma = torch.rand(uv_map.shape[0]).to(device) * 10 + 0.1
                noisy_uv = uv_map + noise * sigma.view(-1, 1, 1, 1)
                
                # Forward pass
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output, logvar = model(noisy_uv, sigma, latents_dict=latents_dict)
                
                # Loss
                loss = criterion(output, target_uv)
                val_loss += loss.item()
                val_pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f'{save_dir}/best_model.pth')
            print(f'New best model saved with val loss: {best_val_loss:.6f}')
        
        # Save comparison images every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_comparison(model, dinov2_model, dinov2_preprocessor, val_loader, 
                          f'{save_dir}/comparisons_epoch_{epoch+1}', device)
        
        # Save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, f'{save_dir}/checkpoint_epoch_{epoch+1}.pth')

def main():
    parser = argparse.ArgumentParser(description='Train UV diffusion model')
    parser.add_argument('--data_dir', required=True, help='Directory containing frame images')
    parser.add_argument('--json_path', required=True, help='JSON file with frame scores')
    parser.add_argument('--mask_path', required=True, help='Path to mask image')
    parser.add_argument('--template_path', required=True, help='Path to template image')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', default='cuda', help='Device to use')
    parser.add_argument('--save_dir', default='checkpoints', help='Directory to save models')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--thresh', type=float, default=0.5, help='Open/closed threshold')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    
    args = parser.parse_args()
    
    # Create dataset
    print("Loading dataset...")
    full_dataset = OpenClosedDataset(
        data_dir=args.data_dir,
        json_path=args.json_path,
        mask_path=args.mask_path,
        template_path=args.template_path,
        thresh=args.thresh
    )
    
    # Split into train/val
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Dataset loaded: {len(train_dataset)} train, {len(val_dataset)} val samples")
    
    # Create models
    print("Creating models...")
    model = create_uv_diffusion_model()
    dinov2_model, dinov2_preprocessor = initialize_dinov2()
    
    print(f"UV Diffusion Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"DINOv2 Model: {sum(p.numel() for p in dinov2_model.parameters()):,} parameters")
    
    # Train model
    train_model(
        model=model,
        dinov2_model=dinov2_model,
        dinov2_preprocessor=dinov2_preprocessor,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        lr=args.lr,
        device=args.device,
        save_dir=args.save_dir
    )
    
    print("Training completed!")

if __name__ == "__main__":
    main()
