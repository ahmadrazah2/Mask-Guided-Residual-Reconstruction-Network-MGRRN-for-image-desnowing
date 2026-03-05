"""
datasets/dataset_loader.py

PyTorch Dataset for the Snow Removal task.

Expected data layout
--------------------
data/
├── snow_images/   ← input (snowy)
├── snow_masks/    ← ground-truth snow mask
└── clean_images/  ← ground-truth clean image

Image files must share the same filename across the three folders.
Supported extensions: .png  .jpg  .jpeg  .bmp  .tif  .tiff
"""

import os
import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms


# ──────────────────────────────────────────────────────────────────────────────
_IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _collect_images(directory: str) -> List[str]:
    """Return sorted list of image filenames (stem only) in *directory*."""
    stems = sorted(
        p.stem
        for p in Path(directory).iterdir()
        if p.suffix.lower() in _IMG_EXTENSIONS
    )
    return stems


def _find_file(directory: str, stem: str) -> str:
    """Find the full path of a file given the directory and filename stem."""
    for ext in _IMG_EXTENSIONS:
        candidate = os.path.join(directory, stem + ext)
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f"No image with stem '{stem}' found in '{directory}'")


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class SnowRemovalDataset(Dataset):
    """
    Paired dataset for snow removal.

    Args:
        root         : Root data directory containing the three sub-folders.
        snow_dir     : Sub-folder name for snowy images  (default: "snow_images").
        mask_dir     : Sub-folder name for snow masks    (default: "snow_masks").
        clean_dir    : Sub-folder name for clean images  (default: "clean_images").
        img_size     : Spatial resolution to resize images to (H = W).
        augment      : Whether to apply random augmentations (flip / rotation).
        transform    : Optional custom torchvision transform applied to all three
                       images after the standard resize + to-tensor pipeline.
    """

    def __init__(
        self,
        root: str,
        snow_dir:  str = "snow_images",
        mask_dir:  str = "snow_masks",
        clean_dir: str = "clean_images",
        img_size:  int = 256,
        augment:   bool = False,
        transform: Optional[Callable] = None,
    ):
        self.root      = root
        self.snow_dir  = os.path.join(root, snow_dir)
        self.mask_dir  = os.path.join(root, mask_dir)
        self.clean_dir = os.path.join(root, clean_dir)
        self.img_size  = img_size
        self.augment   = augment
        self.transform = transform

        # Collect filenames based on the snowy-image directory
        self.stems = _collect_images(self.snow_dir)
        if len(self.stems) == 0:
            raise RuntimeError(f"No images found in '{self.snow_dir}'")

        # Standard transforms (applied before optional augmentations)
        self.resize    = transforms.Resize((img_size, img_size), antialias=True)
        self.to_tensor = transforms.ToTensor()

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.stems)

    # ------------------------------------------------------------------
    def _load(self, path: str) -> Image.Image:
        return Image.open(path).convert("RGB")

    # ------------------------------------------------------------------
    def _augment(
        self,
        snow: torch.Tensor,
        mask: torch.Tensor,
        clean: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply identical random flips/rotation to all three tensors."""
        # Horizontal flip
        if random.random() > 0.5:
            snow  = transforms.functional.hflip(snow)
            mask  = transforms.functional.hflip(mask)
            clean = transforms.functional.hflip(clean)
        # Vertical flip
        if random.random() > 0.5:
            snow  = transforms.functional.vflip(snow)
            mask  = transforms.functional.vflip(mask)
            clean = transforms.functional.vflip(clean)
        # Random 90-degree rotation
        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            snow  = transforms.functional.rotate(snow,  angle)
            mask  = transforms.functional.rotate(mask,  angle)
            clean = transforms.functional.rotate(clean, angle)
        return snow, mask, clean

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict:
        stem = self.stems[idx]

        snow_img  = self._load(_find_file(self.snow_dir,  stem))
        mask_img  = self._load(_find_file(self.mask_dir,  stem))
        clean_img = self._load(_find_file(self.clean_dir, stem))

        # Resize
        snow_img  = self.resize(snow_img)
        mask_img  = self.resize(mask_img)
        clean_img = self.resize(clean_img)

        # To tensor  ([0, 1] float)
        snow_t  = self.to_tensor(snow_img)
        mask_t  = self.to_tensor(mask_img)
        clean_t = self.to_tensor(clean_img)

        # Data augmentation
        if self.augment:
            snow_t, mask_t, clean_t = self._augment(snow_t, mask_t, clean_t)

        # Optional extra transform
        if self.transform is not None:
            snow_t  = self.transform(snow_t)
            mask_t  = self.transform(mask_t)
            clean_t = self.transform(clean_t)

        return {
            "snow":     snow_t,
            "mask":     mask_t,
            "clean":    clean_t,
            "filename": stem,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Helper – build train / val dataloaders from config
# ──────────────────────────────────────────────────────────────────────────────

def build_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders from a parsed config dict.

    Args:
        cfg: The *data* sub-dict from ``configs/config.yaml``.

    Returns:
        train_loader, val_loader
    """
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]

    full_dataset = SnowRemovalDataset(
        root      = data_cfg["root"],
        snow_dir  = data_cfg["snow_images"],
        mask_dir  = data_cfg["snow_masks"],
        clean_dir = data_cfg["clean_images"],
        img_size  = data_cfg["img_size"],
        augment   = True,
    )

    n_train = int(len(full_dataset) * data_cfg["train_split"])
    n_val   = len(full_dataset) - n_train
    train_set, val_set = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg["train"]["seed"]),
    )
    # Disable augmentation for val split
    val_set.dataset.augment = False

    train_loader = DataLoader(
        train_set,
        batch_size  = train_cfg["batch_size"],
        shuffle     = True,
        num_workers = train_cfg["num_workers"],
        pin_memory  = True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size  = train_cfg["batch_size"],
        shuffle     = False,
        num_workers = train_cfg["num_workers"],
        pin_memory  = True,
    )
    return train_loader, val_loader


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ds = SnowRemovalDataset(root="data", img_size=256, augment=True)
    print(f"Dataset size : {len(ds)}")
    if len(ds):
        sample = ds[0]
        print(f"  snow  : {sample['snow'].shape}")
        print(f"  mask  : {sample['mask'].shape}")
        print(f"  clean : {sample['clean'].shape}")
