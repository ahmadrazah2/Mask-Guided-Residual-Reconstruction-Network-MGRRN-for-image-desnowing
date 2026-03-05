"""
scripts/convert_dataset.py

Utility script to pre-process a raw snow image dataset into the directory layout
expected by SnowRemovalDataset.

Input layout (any of several common formats):
    raw/
    ├── snowy/    (or any name – specify via --snow_src)
    ├── masks/
    └── clean/

Output layout:
    data/
    ├── snow_images/
    ├── snow_masks/
    └── clean_images/

Usage:
    python scripts/convert_dataset.py \
        --snow_src  raw/snowy \
        --mask_src  raw/masks \
        --clean_src raw/clean \
        --out_dir   data \
        --img_size  256
"""

import argparse
import os
import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm

_IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def convert(src_dir: str, dst_dir: str, img_size: int, suffix: str = ".png"):
    """Resize and copy all images from *src_dir* to *dst_dir*."""
    Path(dst_dir).mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in Path(src_dir).iterdir() if p.suffix.lower() in _IMG_EXT)

    for p in tqdm(files, desc=f"  {Path(src_dir).name} → {Path(dst_dir).name}"):
        img = Image.open(p).convert("RGB")
        if img_size > 0:
            img = img.resize((img_size, img_size), Image.BICUBIC)
        dst_path = os.path.join(dst_dir, p.stem + suffix)
        img.save(dst_path)


def main():
    parser = argparse.ArgumentParser(description="Dataset conversion helper")
    parser.add_argument("--snow_src",  required=True, help="Source directory of snowy images")
    parser.add_argument("--mask_src",  required=True, help="Source directory of snow masks")
    parser.add_argument("--clean_src", required=True, help="Source directory of clean images")
    parser.add_argument("--out_dir",   default="data", help="Root output directory (default: data/)")
    parser.add_argument("--img_size",  type=int, default=0, help="Resize images (0 = no resize)")
    args = parser.parse_args()

    print(f"\nConverting dataset → '{args.out_dir}/' (img_size={args.img_size or 'original'})\n")
    convert(args.snow_src,  os.path.join(args.out_dir, "snow_images"),  args.img_size)
    convert(args.mask_src,  os.path.join(args.out_dir, "snow_masks"),   args.img_size)
    convert(args.clean_src, os.path.join(args.out_dir, "clean_images"), args.img_size)
    print(f"\nDone. Data written to '{args.out_dir}/'")


if __name__ == "__main__":
    main()
