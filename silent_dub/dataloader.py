import os, json, argparse, random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch
import numpy as np
import cv2

def tensor_to_cv(img_tensor):
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

    # detach → CPU → numpy, channel-last
    img = img_tensor.detach().cpu().permute(1, 2, 0).float().numpy()

    # [-1,1] → [0,255]
    img = ((img * 0.5) + 0.5) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)

    # RGB → BGR for OpenCV
    img = img[..., ::-1]

    return img

def center_crop_512(img):
    """
    Center-crop a 512x512 region from an OpenCV image (H, W, 3).
    Works for uint8 or float images.

    Args
    ----
    img : np.ndarray
        Image of shape (H, W, 3), typically from OpenCV (BGR).

    Returns
    -------
    np.ndarray
        Cropped image of shape (512, 512, 3).
    """
    h, w = img.shape[:2]
    ch, cw = 512, 512

    if h < ch or w < cw:
        raise ValueError(f"Image too small to center crop: got ({h},{w})")

    start_y = (h - ch) // 2
    start_x = (w - cw) // 2

    return img[start_y:start_y+ch, start_x:start_x+cw]

def center_crop_tensor(tensor):
    """Center-crop a (C,H,W) tensor to 512×512."""
    _, h, w = tensor.shape
    top  = (h - 512) // 2
    left = (w - 512) // 2
    return tensor[:, top:top+512, left:left+512]

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

class OpenClosedDataset(Dataset):
    """
    Given:
      • data_dir containing images named '%05d.png'
      • json_path with structure  { clip_name : { '000000': score, ... } }
    Produces pairs (input_from_closed, uv_from_open).
    """

    def __init__(self, data_dir, json_path,mask_path,template_path, thresh=0.5, fp16=False):
        self.data_dir = data_dir
        self.fp16 = fp16
        self.thresh = thresh
        self.mask_path = mask_path

        # self.open_frames, self.closed_frames = [], []
        # self.open_dict,  self.closed_dict  = {}, {}
        with open(json_path, "r") as f:
            meta = json.load(f)

        # --- read & split ---------------------------------------------------
        self.frames = []
        for _, frames in meta.items():  # clip name doesn't matter
            for f_str, val in frames.items():
                label = 1 if val > thresh else 0
                self.frames.append({
                    "frame": f_str,
                    "label": label,
                    "score": val  # optional, keep if needed
                })

        assert any(f["label"] == 1 for f in self.frames), "No open frames found."
        assert any(f["label"] == 0 for f in self.frames), "No closed frames found."

        # --- basic transform ------------------------------------------------
        self.tf = T.Compose([
            T.PILToTensor(),               # uint8 → [C,H,W]
            T.ConvertImageDtype(torch.float32),
            T.Normalize(0.5, 0.5)          # [-1,1]
        ])
        self.mask = self.tf(Image.open(mask_path).convert("RGB"))
        if fp16: self.mask = self.mask.half()

        template_img = Image.open(template_path).convert("RGB")
        if template_img.size != (5120, 1024):
            template_img = template_img.resize((5120, 1024), Image.BICUBIC)
        
        template_tensor = self.tf(template_img)
        if fp16: template_tensor = template_tensor.half()

        self.template_slices = [template_tensor[:, :, i*1024:(i+1)*1024] for i in range(5)]


        
    # --------------------------------------------------------------------- #
    def __len__(self):
        # we draw one open + one closed each time, so the pair count is the min
        return len(self.frames)

    # --------------------------------------------------------------------- #
    def _load_and_slice(self, frame_key):
        """
        frame_key: '000137' (6-digit) → maps to image '00137.png' (5-digit)
        Splits into five 1024-pixel slices horizontally and returns the list.
        """
        frame_id = int(frame_key)               # 137
        img_path = os.path.join(self.data_dir, f"{frame_id:05d}.png")
        img = Image.open(img_path).convert("RGB")

        if img.size != (5120, 1024):            # (W, H)
            img = img.resize((5120, 1024), Image.BICUBIC)

        tensor = self.tf(img)                   # [3,1024,5120]
        if self.fp16:
            tensor = tensor.half()

        # five horizontal tiles
        return [tensor[:, :, i*1024:(i+1)*1024] for i in range(5)]

    # --------------------------------------------------------------------- #
    def __getitem__(self, idx):
        # pick random open / closed frames each call
        # open_key   = random.choice(self.open_frames)
        # closed_key = random.choice(self.closed_frames)

        # closed_slices = self._load_and_slice(closed_key)
        # open_slices   = self._load_and_slice(open_key)
        slices = self._load_and_slice(self.frames[idx]['frame'])
        template_uv = center_crop_tensor(self.template_slices[3])  # template UV
        uv_slice = center_crop_tensor(slices[3])
        final_uv = replace_masked(template_uv, uv_slice, self.mask)
        # return {
        #     "closed_img" : closed_slices[0],   # first tile from closed frame
        #     "open_img" : open_slices[0],
        #     "closed_uv": center_crop_tensor(closed_slices[3]),
        #     "open_uv": center_crop_tensor(open_slices[3]),      # fourth tile from open frame
        #     "closed_st": closed_slices[2],
        #     "open_st": open_slices[2]
        # }
        return {
            "img": slices[0],
            "uv": uv_slice,
            "combined_uv": final_uv,
            "st": slices[2]
        }

# ------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",  required=True, help="Folder with %05d.png images")
    parser.add_argument("--json_path", required=True, help="Original per-frame scores JSON")
    parser.add_argument("--mask_path", required=True, help="UV Mask")
    parser.add_argument("--template_path", required=True, help="UV Mask")
    parser.add_argument("--thresh",    type=float, default=0.5, help="Open/closed cut-off")
    parser.add_argument("--batch_size", type=int,  default=4)
    parser.add_argument("--fp16",     action="store_true", help="Half-precision tensors")
    parser.add_argument("--save_json", help="Write split JSON here (optional)")
    args = parser.parse_args()

    ds = OpenClosedDataset(args.data_dir, args.json_path,args.mask_path,args.template_path ,args.thresh, fp16=args.fp16)

    # optional: dump the split for later use
    # if args.save_json:
    #     with open(args.save_json, "w") as f:
    #         json.dump({"open": ds.open_dict, "closed": ds.closed_dict}, f, indent=2)
    #         print(f"[✓] Split JSON saved to {args.save_json}")

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4)

    for batch in dl:
        print("input :", batch["img"].shape, "uv_map :", batch["uv"].shape)
        cv2.imwrite('uv.png',tensor_to_cv(batch["uv"][0]))
        cv2.imwrite('combined_uv.png',tensor_to_cv(batch["combined_uv"][0]))
        # cv2.imwrite('open_uv.png',center_crop_512(tensor_to_cv(batch["open_uv"][0])))
        break