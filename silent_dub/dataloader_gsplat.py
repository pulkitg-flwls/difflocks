import os, json, argparse, random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch
import numpy as np
import cv2
from pathlib import Path
import matplotlib.cm as cm

def tensor_to_cv(img_tensor, minmax_normalize=False):
    """
    Convert a PyTorch image tensor (C,H,W) in [-1,1] → uint8 BGR numpy array.

    Args
    ----
    img_tensor : torch.Tensor
        Shape [3, H, W]  or [B, 3, H, W] (only first item used).
        Values are assumed to be in [-1, 1] after T.Normalize(0.5, 0.5).

    Returns
    -------
    np.ndarray
        OpenCV-friendly array of shape (H, W, 3) in BGR, dtype=uint8.
    """
    if img_tensor.dim() == 4:          # batch – use first element
        img_tensor = img_tensor[0]

    # detach → CPU → numpy
    img = img_tensor.detach().cpu().float().numpy()  # Still in CHW format
    
    # [-1,1] → [0,255] or unnormalize ImageNet normalization
    if minmax_normalize:
        # Simple [-1,1] → [0,1] → [0,255]
        img = ((img * 0.5) + 0.5) * 255.0
    else:
        # Unnormalize from T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        # img is [C,H,W]; unnormalize: img = img * std + mean
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        img = img * std + mean
        img = img * 255.0
    
    # Convert CHW to HWC (channel-last) for OpenCV
    img = img.transpose(2, 1, 0)  # (C, H, W) → (H, W, C)
    img = np.clip(img, 0, 255).astype(np.uint8)


    return img


def replace_masked(template_uv, target_uv, mask):
    """
    Replace masked region in target_uv with template_uv.
    All inputs: (3, 512, 512) in [-1, 1]
    mask: (3, 512, 512) in [-1, 1], where white (1) means "keep template".
    """
    # Create binary mask: 1 where mask > 0 (threshold at 0)
    binary_mask = (mask > 0).float()

    # Blend: template * mask + target * (1 - mask)
    return template_uv * binary_mask + target_uv * (1 - binary_mask)

def load_gsplat(gsplat_path, frame_idx, device):
    """
    Load gsplat parameters for a given frame.
    Args:
        gsplat_path: path to gsplat parameters
        frame_idx: frame index
        device: device to load the gsplat parameters to
    Returns:
        aces_diffuse_alb: aces diffuse albedo
        delta_xyz: delta xyz
        delta_rot: delta rotation
        delta_scale: delta scale
        opacity: opacity
        verts: vertices
    """
    gsplat_params_path = gsplat_path / f"{frame_idx}.npy"
    gsplat_params_arr = np.load(gsplat_params_path)
    gsplat_params_tensor = torch.from_numpy(gsplat_params_arr).to(torch.float32).contiguous().permute(2,1,0).to(device)
    
    aces_diffuse_alb = gsplat_params_tensor[0:3,:,:]    #[3,512,512]    # aces diffuse albedo
    # Normalize from [0,1] to [-1,1]
    aces_diffuse_alb = aces_diffuse_alb * 2.0 - 1.0
    delta_xyz = gsplat_params_tensor[3:6,:,:]    #[3,512,512]    # delta xyz
    delta_rot = gsplat_params_tensor[6:10,:,:]    #[4,512,512]    # delta rotation
    delta_scale = gsplat_params_tensor[10:13,:,:]    #[3,512,512]    # delta scale
    opacity = gsplat_params_tensor[13:14,:,:]    #[1,512,512]    # opacity
    verts = gsplat_params_tensor[14:,:,:]    #[3,512,512]    # vertices
    return {
        "aces_diffuse_alb": aces_diffuse_alb,
        "delta_xyz": delta_xyz,
        "delta_rot": delta_rot,
        "delta_scale": delta_scale,
        "opacity": opacity,
        "verts": verts
    }

def load_fotd(fotd_path, frame_idx, device):
    """
    Load fotd parameters for a given frame.
    Args:
        fotd_path: path to fotd parameters
        frame_idx: frame index
        device: device to load the fotd parameters to
    Returns:
        fotd_params: list of 4 tiles (1024x1024x3 each)
    """
    fotd_params_path = fotd_path / f"{frame_idx}.png"
    fotd_params_arr = np.array(Image.open(fotd_params_path))
    fotd_params_arr = (fotd_params_arr / 255.0).astype(np.float32) # Normalize from [0,255] to [0,1]
    # Dinov2 preprocessing requires image to be in [0,1] and normalized with ImageNet normalization
    preprocessor = T.Compose([
        T.Resize((770, 770), interpolation=T.InterpolationMode.BICUBIC),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    ref_img = torch.from_numpy(fotd_params_arr[:, :1024, :]).to(torch.float32).contiguous().permute(2,1,0)
    ref_img = preprocessor(ref_img)
    fotd_img = torch.from_numpy(fotd_params_arr[:, 1024:2048, :]).to(torch.float32).contiguous().permute(2,1,0)
    st_img = torch.from_numpy(fotd_params_arr[:, 2048:3072, :]).to(torch.float32).contiguous().permute(2,1,0)
    mask_img = torch.from_numpy(fotd_params_arr[:, 3072:4096, :]).to(torch.float32).contiguous().permute(2,1,0)
    return {
        "ref_img": ref_img,
        "fotd_img": fotd_img,
        "st_img": st_img,
        "mask_img": mask_img
    }


class SilentDubDataset(Dataset):
    """
    Dataset for SilentDub with gsplat and fotd data.
    
    Structure:
    - gsplat_dir: contains trackname folders, each with frame files (.npy)
    - fotd_dir: contains corresponding trackname folders with frame files (.png)
    - json_dir: contains metadata JSON files with frame scores (open-ratio)
    
    __len__ : returns the number of tracks
    __getitem__ : returns a random pair of open/closed frames from the same track.
    During training, the input is reference image and gsplat parameters of closed frame and output is gsplat parameters of open frame.
    Each __getitem__ returns a pair of open/closed frames from the same track.
    
    Also extracts DINOv2 features from closed RGB image for conditioning.
    """

    def __init__(self, root_dir, json_dir, mask_path, device='cpu', open_ratio_threshold=0.1, load_dinov2=True, template_dir=None):
        self.root_dir = Path(root_dir)
        self.json_dir = Path(json_dir)
        self.device = device
        self.open_ratio_threshold = open_ratio_threshold
        self.load_dinov2 = load_dinov2
        
        self.mask = (np.array(Image.open(mask_path).convert("RGB"))/255.0).astype(np.float32)    #[512,512,3]    # mask
        self.mask = torch.from_numpy(self.mask).to(torch.float32).contiguous().permute(2,1,0).to(device)
        print(f"Mask shapes: {self.mask.shape}")
        
        # Set up directory paths
        self.gsplat_dir = self.root_dir / "train_gsplatParams_BOHR_trackformer-5.2.gteeth-dev-1"
        self.fotd_dir = self.root_dir / "train_plate_fotd_BOHR_trackformer-5.2.gteeth-dev-1_sym"
        
        # Load template teeth if provided
        self.template_teeth = None
        if template_dir is not None:
            template_dir = Path(template_dir)
            template_gsplat_path = template_dir / "gsplat.npy"
            template_fotd_path = template_dir / "fotd.png"
            if template_gsplat_path.exists() and template_fotd_path.exists():
                # Load template gsplat - expects directory and frame_idx (stem)
                template_gsplat_params = load_gsplat(template_dir, "gsplat", self.device)
                # Load template fotd - expects directory and frame_idx (stem)
                template_fotd_params = load_fotd(template_dir, "fotd", self.device)
                self.template_teeth = {
                    "gsplat_params": template_gsplat_params,
                    "fotd_params": template_fotd_params
                }
                print(f"Loaded template teeth from {template_dir}")
            else:
                print(f"Warning: Template files not found in {template_dir}: expected gsplat.npy and fotd.png")
        
        # Build track data structure: trackname -> {open_frames, closed_frames}
        self.track_data = {}
        self.tracks = []
        self._scan_directories()
        
        # Initialize DINOv2 model for feature extraction
        self.dinov2_model = None
        if self.load_dinov2:
            print("Initializing DINOv2 model...")
            self.dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
            self.dinov2_model.eval()
            self.dinov2_model = self.dinov2_model.to(device)
            print("DINOv2 model loaded and moved to device")
        
        print(f"Loaded {len(self.tracks)} tracks with open/closed frame pairs")
    
    def _scan_directories(self):
        """Scan JSON directory first, then process only tracks with corresponding JSON files."""
        if not self.gsplat_dir.exists() or not self.fotd_dir.exists():
            raise FileNotFoundError(f"Required directories not found: {self.gsplat_dir} or {self.fotd_dir}")
        
        if not self.json_dir.exists():
            raise FileNotFoundError(f"JSON directory not found: {self.json_dir}")
        
        # Scan JSON directory for track subdirectories
        json_tracks = [d.name for d in self.json_dir.iterdir() if d.is_file()]
        for trackname in json_tracks:
            # Check if JSON file exists in subdirectory
            json_path = self.json_dir / trackname
            
            # Check if corresponding track folders exist in gsplat and fotd directories
            trackdir = trackname.replace(".json", ".dir")
            gsplat_track_dir = self.gsplat_dir / trackdir
            fotd_track_dir = self.fotd_dir / trackdir
            
            if not gsplat_track_dir.exists() or not fotd_track_dir.exists():
                continue
            
            # Load JSON file
            try:
                with open(json_path, "r") as f:
                    json_data = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load {json_path}: {e}")
                continue
            
            # Extract frame data from JSON (key is "{trackname}.dir")
            json_key = f"{trackdir}"
            
            
            if json_key not in json_data:
                continue
            
            frame_scores = json_data[json_key]
            
            # Build open and closed frame lists
            open_frames = []
            closed_frames = []
            
            for frame_filename, open_ratio in frame_scores.items():
                # Strip .png extension to get frame index
                frame_idx = frame_filename.replace(".png", "")
                
                # Verify frame files exist
                
                gsplat_file = gsplat_track_dir / f"{frame_idx}.npy"
                fotd_file = fotd_track_dir / f"{frame_idx}.png"
                
                if not gsplat_file.exists() or not fotd_file.exists():
                    continue
                
                # Classify based on open_ratio threshold
                if open_ratio > self.open_ratio_threshold:
                    open_frames.append(frame_idx)
                elif open_ratio < self.open_ratio_threshold:
                    closed_frames.append(frame_idx)
                # If exactly equal, skip (edge case)
            
            # Only add track if it has both open and closed frames
            if len(open_frames) > 0 and len(closed_frames) > 0:
                self.track_data[trackname] = {
                    "open_frames": open_frames,
                    "closed_frames": closed_frames
                }
                self.tracks.append(trackname)
        
        if not self.tracks:
            raise ValueError("No tracks with valid open/closed frame pairs found")

    def __len__(self):
        """Return total number of tracks (each call returns one open/closed pair)."""
        return len(self.tracks)

    def _extract_dinov2_features(self, rgb_image):
        """
        Extract DINOv2 features from RGB image.
        
        Args:
            rgb_image: Preprocessed RGB image tensor [3, 770, 770] (already normalized)
        
        Returns:
            dict with 'cls_token' [1024] and 'final_latent' [1024, 55, 55]
        """
        if self.dinov2_model is None:
            return None
        
        with torch.no_grad():
            rgb_input = rgb_image.unsqueeze(0).to(torch.float32).to(self.device)  # [1, 3, 770, 770]
            dinov2_output = self.dinov2_model.forward_features(rgb_input)
            
            # Extract features
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
        
        return dinov2_features
    
    def __getitem__(self, idx):
        """Return one open/closed frame pair from a randomly sampled track."""
        # Randomly sample a track
        trackname = random.choice(self.tracks)
        trackdir = trackname.replace(".json", ".dir")
        # Get open and closed frame lists for this track
        open_frames = self.track_data[trackname]["open_frames"]
        closed_frames = self.track_data[trackname]["closed_frames"]
        
        # Randomly sample one open frame and one closed frame
        open_frame_idx = random.choice(open_frames)
        closed_frame_idx = random.choice(closed_frames)
        
        # Construct paths for this track
        gsplat_track_dir = self.gsplat_dir / trackdir
        fotd_track_dir = self.fotd_dir / trackdir
        
        # Load gsplat parameters for both frames
        open_gsplat_params = load_gsplat(gsplat_track_dir, open_frame_idx, self.device)
        closed_gsplat_params = load_gsplat(gsplat_track_dir, closed_frame_idx, self.device)
        
        # Load fotd parameters for both frames
        open_fotd_params = load_fotd(fotd_track_dir, open_frame_idx, self.device)
        closed_fotd_params = load_fotd(fotd_track_dir, closed_frame_idx, self.device)
        
        # Extract DINOv2 features from closed RGB image for conditioning
        latents = None
        if self.load_dinov2:
            # closed_fotd_params["ref_img"] is already preprocessed [3, 770, 770]
            dinov2_features = self._extract_dinov2_features(closed_fotd_params["ref_img"])
            if dinov2_features is not None:
                latents = {
                    "dinov2": dinov2_features
                }
        
        result = {
            "open": {
                "gsplat_params": open_gsplat_params,
                "fotd_params": open_fotd_params
            },
            "closed": {
                "gsplat_params": closed_gsplat_params,
                "fotd_params": closed_fotd_params
            },
            "mask": self.mask
        }
        
        if self.template_teeth is not None:
            result["template_teeth"] = self.template_teeth
        
        if latents is not None:
            result["latents"] = latents
        
        return result

# ------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="Folder with %05d.png images")
    parser.add_argument("--json_dir", help="Original per-frame scores JSON")
    parser.add_argument("--mask_path", help="UV Mask")
    parser.add_argument("--template_dir", help="Template directory")
    parser.add_argument("--thresh",    type=float, default=0.5, help="Open/closed cut-off")
    parser.add_argument("--batch_size", type=int,  default=4)
    parser.add_argument("--fp16",     action="store_true", help="Half-precision tensors")
    parser.add_argument("--save_json", help="Write split JSON here (optional)")
    parser.add_argument("--root_dir", required=True, help="Root directory")
    parser.add_argument("--track_clip", help="Track clip name")
    parser.add_argument("--load_dinov2", action="store_true", help="Load DINOv2 features")
    args = parser.parse_args()

    ROOT = Path(args.root_dir)
    TRACK_CLIP = args.track_clip
    STORE_ROOT = ROOT / "ingest/"
    REPO_ROOT = ROOT / "vfhq/"
    # INGEST_CLIP = TRACK_CLIP.replace("_track_000", "")

    # ingest_dir = STORE_ROOT / f"{INGEST_CLIP}"
    gsplats_dir = REPO_ROOT / "train_gsplatParams_BOHR_trackformer-5.2.gteeth-dev-1" / f"{TRACK_CLIP}.dir"
    fotd_dir = REPO_ROOT / "train_plate_fotd_BOHR_trackformer-5.2.gteeth-dev-1_sym" / f"{TRACK_CLIP}.dir"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Simple usage - requires json_dir argument now
    dataset = SilentDubDataset(ROOT, json_dir=args.json_dir, mask_path=args.mask_path, \
                                template_dir=args.template_dir,load_dinov2=args.load_dinov2, \
                                open_ratio_threshold=args.thresh, device=device)
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading a sample
    if len(dataset) > 0:
        idx = np.random.randint(0, len(dataset))
        sample = dataset[idx]
        open_data = sample["open"]
        closed_data = sample["closed"]
        
        
        # Compose visualization: top row open [fotd_open  uvmap_open], bottom row closed [fotd_closed  uvmap_closed]

        # Convert fotd (770x770) and gsplat_uv (512x512) to np.uint8 BGR
        fotd_open = tensor_to_cv(open_data['fotd_params']['ref_img'], minmax_normalize=False)
        fotd_close = tensor_to_cv(closed_data['fotd_params']['ref_img'], minmax_normalize=False)

        gsplat_open = tensor_to_cv(open_data['gsplat_params']['aces_diffuse_alb'], minmax_normalize=True)
        gsplat_close = tensor_to_cv(closed_data['gsplat_params']['aces_diffuse_alb'], minmax_normalize=True)

        # Pad gsplat images from 512x512 to 770x770 (BGR)
        def pad_to_770(img):
            h, w, c = img.shape
            pad_h = 770 - h
            pad_w = 770 - w
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            return np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=0)

        gsplat_open_pad = pad_to_770(gsplat_open)
        gsplat_close_pad = pad_to_770(gsplat_close)

        # Add labels to images
        def add_label(img, text, position='top-left'):
            """Add text label to image. img is BGR uint8."""
            img_labeled = img.copy()
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5
            thickness = 3
            color = (255, 255, 255)  # White text
            bg_color = (0, 0, 0)  # Black background
            
            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            if position == 'top-left':
                x, y = 10, 40
            elif position == 'top-right':
                x, y = img.shape[1] - text_width - 10, 40
            else:
                x, y = 10, 40
            
            # Draw background rectangle
            cv2.rectangle(img_labeled, (x - 5, y - text_height - 5), 
                         (x + text_width + 5, y + baseline + 5), bg_color, -1)
            # Draw text
            cv2.putText(img_labeled, text, (x, y), font, font_scale, color, thickness)
            return img_labeled
        
        # Add labels to individual images before concatenation
        fotd_open_labeled = add_label(fotd_open, "Open Mouth")
        gsplat_open_labeled = add_label(gsplat_open_pad, "Open Mouth", position='top-right')
        fotd_close_labeled = add_label(fotd_close, "Closed Mouth")
        gsplat_close_labeled = add_label(gsplat_close_pad, "Closed Mouth", position='top-right')
        
        # Concatenate top row: [fotd_open, gsplat_open_pad]
        row1 = np.concatenate([fotd_open_labeled, gsplat_open_labeled], axis=1)
        # Concatenate bottom row: [fotd_close, gsplat_close_pad]
        row2 = np.concatenate([fotd_close_labeled, gsplat_close_labeled], axis=1)
        # Stack rows vertically
        vis_img = np.concatenate([row1, row2], axis=0)

        
        
        if sample.get("template_teeth") is not None:
            template_gsplat = tensor_to_cv(sample["template_teeth"]["gsplat_params"]["aces_diffuse_alb"], minmax_normalize=True)
            template_fotd = tensor_to_cv(sample["template_teeth"]["fotd_params"]["ref_img"], minmax_normalize=False)
            template_gsplat_pad = pad_to_770(template_gsplat)
            template_fotd_pad = pad_to_770(template_fotd)
            
            # Add labels to template images
            template_fotd_labeled = add_label(template_fotd_pad, "Template Teeth")
            template_gsplat_labeled = add_label(template_gsplat_pad, "Template Teeth", position='top-right')
            
            row3 = np.concatenate([template_fotd_labeled, template_gsplat_labeled], axis=1)
            vis_img = np.concatenate([vis_img, row3], axis=0)

        print(f"viz img shape (HxW): {vis_img.shape}")
        out_path = f"test_images/composite_fotd_gsplat_grid.png"
        Image.fromarray(vis_img).save(out_path)
        
        # Visualize DINOv2 global latent (cls_token) if available
        # if sample.get("latents") is not None and "dinov2" in sample["latents"]:
        #     print("Visualizing DINOv2 global latent...")
        #     cls_token = sample["latents"]["dinov2"]["cls_token"]  # [1024]
            
        #     # Convert to numpy and reshape to 2D grid (32x32 = 1024)
        #     cls_token_np = cls_token.detach().cpu().numpy()
        #     cls_token_2d = cls_token_np.reshape(32, 32)
            
        #     # Normalize to [0, 1] for visualization
        #     cls_token_min = cls_token_2d.min()
        #     cls_token_max = cls_token_2d.max()
        #     if cls_token_max > cls_token_min:
        #         cls_token_norm = (cls_token_2d - cls_token_min) / (cls_token_max - cls_token_min)
        #     else:
        #         cls_token_norm = cls_token_2d
            
        #     # Apply colormap (viridis) and convert to uint8
        #     colormap = cm.get_cmap('viridis')
        #     cls_token_colored = colormap(cls_token_norm)[:, :, :3]  # Remove alpha channel
        #     cls_token_uint8 = (cls_token_colored * 255).astype(np.uint8)
            
        #     # Resize to a larger size for better visibility (e.g., 512x512)
        #     cls_token_resized = cv2.resize(cls_token_uint8, (512, 512), interpolation=cv2.INTER_NEAREST)
            
        #     # Convert RGB to BGR for consistency with other images
        #     cls_token_bgr = cv2.cvtColor(cls_token_resized, cv2.COLOR_RGB2BGR)
            
        #     # Add label
        #     cls_token_labeled = add_label(cls_token_bgr, "DINOv2 Global Latent")
            
        #     # Save visualization
        #     dinov2_out_path = f"test_images/dinov2_global_latent.png"
        #     Image.fromarray(cls_token_labeled).save(dinov2_out_path)
        #     print(f"Saved DINOv2 global latent visualization to {dinov2_out_path}")

        # img_path = f"test_images/gsplat_open_uvmap.png"
        # img = open_data['gsplat_params']['aces_diffuse_alb']
        # np_img = tensor_to_cv(img,minmax_normalize=True)
        # print(f"Open frame gsplat shapes: {np_img.shape}")
        # Image.fromarray(np_img).save(img_path)

        # img_path = f"test_images/gsplat_closed_uvmap.png"
        # img = closed_data['gsplat_params']['aces_diffuse_alb']
        # np_img = tensor_to_cv(img,minmax_normalize=True)
        # print(f"Closed frame gsplat shapes: {np_img.shape}")
        # Image.fromarray(np_img).save(img_path)

        # img_path = f"test_images/mask_sample.png"
        # img = sample['mask']
        # np_img = tensor_to_cv(img,minmax_normalize=False)
        # print(f"Mask shapes: {np_img.shape}")
        # Image.fromarray(np_img).save(img_path)
        