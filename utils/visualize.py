"""
utils/visualize.py

Visualisation helpers for Snow Removal results.

Functions
---------
save_comparison   – side-by-side grid: snowy | predicted | ground-truth
save_mask_overlay – snowy image with red-tinted mask overlay
log_tensorboard   – write images to a TensorBoard SummaryWriter
"""

import os
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils as vutils
from PIL import Image


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _to_numpy(t: torch.Tensor) -> np.ndarray:
    """Convert a [C, H, W] tensor in [0,1] to a uint8 HWC numpy array."""
    return (t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def _ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def save_comparison(
    snowy:      torch.Tensor,
    pred_clean: torch.Tensor,
    gt_clean:   torch.Tensor,
    save_path:  str,
    pred_mask:  Optional[torch.Tensor] = None,
    gt_mask:    Optional[torch.Tensor] = None,
    title:      str = "",
):
    """
    Save a side-by-side comparison figure.

    If masks are provided, a 5-panel figure is saved:
      Snowy | Pred Clean | GT Clean | Pred Mask | GT Mask

    Otherwise a 3-panel figure:
      Snowy | Pred Clean | GT Clean

    Args:
        snowy       : Input snowy image         [3, H, W].
        pred_clean  : Model output clean image  [3, H, W].
        gt_clean    : Ground-truth clean image  [3, H, W].
        save_path   : Absolute file path to save the figure (.png).
        pred_mask   : (optional) Predicted mask [3, H, W].
        gt_mask     : (optional) GT mask        [3, H, W].
        title       : Figure title string.
    """
    _ensure_dir(os.path.dirname(save_path))

    panels = [snowy, pred_clean, gt_clean]
    labels = ["Snowy (Input)", "Predicted Clean", "Ground-Truth Clean"]

    if pred_mask is not None and gt_mask is not None:
        panels += [pred_mask, gt_mask]
        labels += ["Predicted Mask", "GT Mask"]

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, img_t, label in zip(axes, panels, labels):
        ax.imshow(_to_numpy(img_t))
        ax.set_title(label, fontsize=9)
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────

def save_mask_overlay(
    snowy:     torch.Tensor,
    mask:      torch.Tensor,
    save_path: str,
    alpha:     float = 0.4,
):
    """
    Overlay the predicted mask on the snowy image with a red tint.

    Args:
        snowy     : Input image  [3, H, W] in [0,1].
        mask      : Predicted snow mask [3 or 1, H, W] in [0,1].
        save_path : Path to save the overlay image.
        alpha     : Mask opacity  (0=invisible, 1=opaque).
    """
    _ensure_dir(os.path.dirname(save_path))

    img_np  = _to_numpy(snowy).astype(np.float32)
    if mask.shape[0] == 3:
        # Use mean across channels as single-channel mask
        mask_np = mask.detach().cpu().mean(dim=0).numpy()   # [H, W]
    else:
        mask_np = mask.detach().cpu().squeeze(0).numpy()

    # Red-channel overlay
    overlay = img_np.copy()
    overlay[:, :, 0] = np.clip(img_np[:, :, 0] + mask_np * 255 * alpha, 0, 255)
    overlay[:, :, 1] = np.clip(img_np[:, :, 1] * (1 - mask_np * alpha), 0, 255)
    overlay[:, :, 2] = np.clip(img_np[:, :, 2] * (1 - mask_np * alpha), 0, 255)

    Image.fromarray(overlay.astype(np.uint8)).save(save_path)


# ──────────────────────────────────────────────────────────────────────────────

def log_tensorboard(
    writer,
    tag:       str,
    snowy:     torch.Tensor,
    pred:      torch.Tensor,
    gt:        torch.Tensor,
    step:      int,
    n_images:  int = 4,
):
    """
    Write a grid of (snowy | pred | gt) images to TensorBoard.

    Args:
        writer   : ``torch.utils.tensorboard.SummaryWriter`` instance.
        tag      : Tag name in TensorBoard.
        snowy    : Batch of snowy images   [B, 3, H, W].
        pred     : Predicted clean images  [B, 3, H, W].
        gt       : Ground-truth clean      [B, 3, H, W].
        step     : Global training step / epoch.
        n_images : Maximum number of images from the batch to display.
    """
    n = min(n_images, snowy.size(0))
    grid = vutils.make_grid(
        torch.cat([snowy[:n], pred[:n], gt[:n]], dim=0).clamp(0, 1),
        nrow=n,
        normalize=False,
    )
    writer.add_image(tag, grid, global_step=step)
