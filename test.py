"""
test.py

Test a trained MGRRN model in THREE ways:

  1. Single image  →  python test.py --input photo.jpg  --checkpoint checkpoints/mgrrn.pth
  2. Folder        →  python test.py --input snowy_dir/ --checkpoint checkpoints/mgrrn.pth
  3. Full dataset benchmark (with PSNR/SSIM, needs ground-truth)
                   →  python test.py --checkpoint checkpoints/mgrrn.pth

Mode 1 & 2  do NOT need masks or clean images – just give the snowy image(s).
Mode 3  reads from data/ and computes PSNR & SSIM against ground-truth.
"""

import argparse
import csv
import os
from pathlib import Path
from typing import List

import torch
import torchvision.transforms.functional as TF
import yaml
from PIL import Image
from tqdm import tqdm

from datasets.dataset_loader import SnowRemovalDataset
from models.model import ResidualSnowRemoval
from torch.utils.data import DataLoader
from utils.metrics import MetricTracker, compute_psnr, compute_ssim
from utils.visualize import save_comparison, save_mask_overlay


# ──────────────────────────────────────────────────────────────────────────────
_IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def get_device(preferred: str = "auto") -> torch.device:
    """
    Resolve the compute device.

    preferred:
        'auto'  – CUDA → MPS (Apple Silicon) → CPU  (default)
        'cuda'  – force NVIDIA GPU
        'mps'   – force Apple M-chip GPU
        'cpu'   – force CPU
    """
    if preferred == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(preferred)
    print(f"Device : {device}")
    return device


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_model(checkpoint: str, device: torch.device) -> ResidualSnowRemoval:
    model = ResidualSnowRemoval().to(device)
    ckpt  = torch.load(checkpoint, map_location=device)

    # Handle both formats:
    #   A) wrapped dict  → {"model_state_dict": ..., "epoch": ...}  (saved by train.py)
    #   B) raw state dict → {"layer.weight": tensor, ...}           (saved externally)
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt   # assume it IS the state dict

    model.load_state_dict(state_dict)
    model.eval()
    print(f"✓ Loaded checkpoint : {checkpoint}")
    return model


def collect_images(input_path: str) -> List[Path]:
    """Return list of image Paths from a file or directory."""
    p = Path(input_path)
    if p.is_file():
        return [p] if p.suffix.lower() in _IMG_EXT else []
    return sorted(f for f in p.iterdir() if f.suffix.lower() in _IMG_EXT)


# ──────────────────────────────────────────────────────────────────────────────
# MODE 1 & 2 – single image or folder  (no ground-truth needed)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def test_images(args):
    """Run the model on one image or a whole folder – no GT required."""
    device = get_device(args.device)

    model   = load_model(args.checkpoint, device)
    images  = collect_images(args.input)
    out_dir = Path(args.output_dir or "outputs/test_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not images:
        print(f"No supported images found at: {args.input}")
        return

    img_size = args.img_size  # resize before inference; 0 = keep original

    print(f"\nProcessing {len(images)} image(s)…\n")
    for img_path in tqdm(images, desc="Testing"):
        img_pil   = Image.open(img_path).convert("RGB")
        orig_size = img_pil.size   # (W, H) – restored after inference

        # Resize to square for the model
        if img_size > 0:
            resized = img_pil.resize((img_size, img_size), Image.BICUBIC)
        else:
            resized = img_pil

        tensor = TF.to_tensor(resized).unsqueeze(0).to(device)   # [1,3,H,W]

        pred_clean, pred_mask = model(tensor)

        # ── Save clean output ─────────────────────────────────────────────────
        clean_pil = TF.to_pil_image(pred_clean.squeeze(0).clamp(0, 1).cpu())
        clean_pil = clean_pil.resize(orig_size, Image.BICUBIC)
        clean_path = out_dir / f"{img_path.stem}_clean.png"
        clean_pil.save(clean_path)

        # ── Save mask overlay (optional) ──────────────────────────────────────
        if args.save_mask:
            overlay_path = str(out_dir / f"{img_path.stem}_mask_overlay.png")
            save_mask_overlay(
                tensor.squeeze(0).cpu(),
                pred_mask.squeeze(0).cpu(),
                save_path=overlay_path,
            )

    print(f"\n✓ Results saved to: {out_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# MODE 3 – full dataset benchmark with PSNR & SSIM
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def test_dataset(args):
    """Evaluate on the structured data/ folder and compute PSNR/SSIM."""
    cfg    = load_config(args.config)
    device = get_device(args.device)

    data_cfg = cfg["data"]
    dataset  = SnowRemovalDataset(
        root      = data_cfg["root"],
        snow_dir  = data_cfg["snow_images"],
        mask_dir  = data_cfg["snow_masks"],
        clean_dir = data_cfg["clean_images"],
        img_size  = data_cfg["img_size"],
        augment   = False,
    )
    loader  = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    model   = load_model(args.checkpoint, device)

    out_dir = Path(args.output_dir or "outputs/test_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    tracker  = MetricTracker()
    csv_rows = [["filename", "psnr", "ssim"]]

    for batch in tqdm(loader, desc="Evaluating dataset"):
        snow     = batch["snow"].to(device)
        mask_gt  = batch["mask"].to(device)
        clean_gt = batch["clean"].to(device)
        fname    = batch["filename"][0]

        pred_clean, pred_mask = model(snow)

        psnr_val = compute_psnr(pred_clean, clean_gt)
        ssim_val = compute_ssim(pred_clean, clean_gt)
        tracker.update(pred_clean, clean_gt)
        csv_rows.append([fname, f"{psnr_val:.4f}", f"{ssim_val:.4f}"])

        if args.save_images:
            save_comparison(
                snow[0], pred_clean[0], clean_gt[0],
                save_path = str(out_dir / f"{fname}.png"),
                pred_mask = pred_mask[0],
                gt_mask   = mask_gt[0],
                title     = fname,
            )

    # ── Print summary ─────────────────────────────────────────────────────────
    mean_psnr, mean_ssim = tracker.result()
    print(f"\n{'─'*40}")
    print(f"  Samples   : {len(dataset)}")
    print(f"  Mean PSNR : {mean_psnr:.2f} dB")
    print(f"  Mean SSIM : {mean_ssim:.4f}")
    print(f"{'─'*40}")

    csv_path = out_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
    print(f"Per-image metrics → {csv_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MGRRN Test Script",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Test a single image (no ground-truth needed):\n"
            "  python test.py --input photo.jpg --checkpoint checkpoints/mgrrn.pth\n\n"
            "  # Test a whole folder of images:\n"
            "  python test.py --input snowy_folder/ --checkpoint checkpoints/mgrrn.pth\n\n"
            "  # Full benchmark on data/ folder (computes PSNR & SSIM):\n"
            "  python test.py --checkpoint checkpoints/mgrrn.pth --save_images\n"
        ),
    )
    parser.add_argument(
        "--input",
        default=None,
        metavar="PATH",
        help="Path to a single snowy image OR a folder of images.\n"
             "If omitted, runs benchmark mode on the data/ folder.",
    )
    parser.add_argument("--checkpoint",  default="checkpoints/mgrrn.pth",  help="Model checkpoint (.pth)")
    parser.add_argument("--config",      default="configs/config.yaml",     help="Config YAML (used in benchmark mode)")
    parser.add_argument("--output_dir",  default=None,                      help="Where to save outputs (default: outputs/test_results/)")
    parser.add_argument("--img_size",    type=int, default=256,             help="Resize input image before inference (0 = keep original size)")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Compute device:\n"
             "  auto – CUDA → MPS (Apple M-chip) → CPU  [default]\n"
             "  cuda – NVIDIA GPU\n"
             "  mps  – Apple Silicon GPU\n"
             "  cpu  – CPU only",
    )
    parser.add_argument("--save_images", action="store_true",               help="[Benchmark mode] Save side-by-side comparison panels")
    parser.add_argument("--save_mask",   action="store_true",               help="[Image mode] Also save red-tinted mask overlay")
    args = parser.parse_args()

    if args.input:
        # ── Mode 1 / 2: direct image / folder input ───────────────────────────
        test_images(args)
    else:
        # ── Mode 3: full dataset benchmark ────────────────────────────────────
        test_dataset(args)


if __name__ == "__main__":
    main()
