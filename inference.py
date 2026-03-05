"""
inference.py

Run a trained MGRRN checkpoint on single images or an entire directory.

Usage:
    # Single image
    python inference.py --input path/to/snowy.jpg --checkpoint checkpoints/best_model.pth

    # Directory of images
    python inference.py --input path/to/snowy_dir/ --checkpoint checkpoints/best_model.pth --output outputs/predictions/
"""

import argparse
import os
from pathlib import Path

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

from models.model import ResidualSnowRemoval
from utils.visualize import save_mask_overlay


# ──────────────────────────────────────────────────────────────────────────────
_SUPPORTED = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def collect_images(input_path: str):
    p = Path(input_path)
    if p.is_file():
        return [p] if p.suffix.lower() in _SUPPORTED else []
    return sorted(f for f in p.iterdir() if f.suffix.lower() in _SUPPORTED)


def load_model(checkpoint_path: str, device: torch.device) -> ResidualSnowRemoval:
    model = ResidualSnowRemoval().to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")
    return model


@torch.no_grad()
def run_inference(
    model:    ResidualSnowRemoval,
    img_path: Path,
    out_dir:  Path,
    img_size: int,
    device:   torch.device,
    save_overlay: bool,
):
    # Load & preprocess
    img_pil = Image.open(img_path).convert("RGB")
    orig_size = img_pil.size          # (W, H)

    resized = img_pil.resize((img_size, img_size), Image.BICUBIC)
    tensor  = TF.to_tensor(resized).unsqueeze(0).to(device)  # [1,3,H,W]

    # Forward pass
    pred_clean, pred_mask = model(tensor)

    # Resize back to original resolution
    pred_clean_pil = TF.to_pil_image(pred_clean.squeeze(0).clamp(0, 1).cpu())
    pred_clean_pil = pred_clean_pil.resize(orig_size, Image.BICUBIC)

    # Save clean image
    stem     = img_path.stem
    out_path = out_dir / f"{stem}_clean.png"
    pred_clean_pil.save(out_path)
    print(f"  ✓  Saved clean image → {out_path}")

    # Save mask overlay
    if save_overlay:
        overlay_path = str(out_dir / f"{stem}_mask_overlay.png")
        save_mask_overlay(
            tensor.squeeze(0).cpu(),
            pred_mask.squeeze(0).cpu(),
            save_path=overlay_path,
        )
        print(f"  ✓  Saved mask overlay → {overlay_path}")


# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="MGRRN Inference")
    parser.add_argument("--input",      required=True,                         help="Input image or directory")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth",  help="Checkpoint path")
    parser.add_argument("--output",     default="outputs/predictions",          help="Output directory")
    parser.add_argument("--img_size",   type=int, default=256,                  help="Resize input to this size before inference")
    parser.add_argument("--save_overlay", action="store_true",                  help="Also save a red-tinted snow mask overlay")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    model   = load_model(args.checkpoint, device)
    images  = collect_images(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not images:
        print(f"No supported images found at: {args.input}")
        return

    print(f"Processing {len(images)} image(s)…\n")
    for img_path in images:
        print(f"→ {img_path.name}")
        run_inference(
            model, img_path, out_dir,
            img_size     = args.img_size,
            device       = device,
            save_overlay = args.save_overlay,
        )

    print(f"\nDone. Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
