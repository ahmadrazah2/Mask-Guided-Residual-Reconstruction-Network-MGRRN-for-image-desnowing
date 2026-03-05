"""
utils/metrics.py

Evaluation metrics for snow removal:
  • PSNR  – Peak Signal-to-Noise Ratio  (higher is better)
  • SSIM  – Structural Similarity Index (higher is better)
"""

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# PSNR
# ──────────────────────────────────────────────────────────────────────────────

def compute_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_val: float = 1.0,
) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR).

    Args:
        pred   : Predicted image tensor [B, C, H, W] or [C, H, W], in [0, max_val].
        target : Ground-truth tensor of the same shape.
        max_val: Maximum possible pixel value (1.0 for normalised images).

    Returns:
        Average PSNR (dB) over the batch.
    """
    with torch.no_grad():
        mse = F.mse_loss(pred, target, reduction="none")          # [B, C, H, W]
        mse = mse.mean(dim=[1, 2, 3])                             # [B]
        # Guard against log(0) for perfect predictions
        mse = torch.clamp(mse, min=1e-10)
        psnr_per_sample = 10.0 * torch.log10(max_val ** 2 / mse) # [B]
    return psnr_per_sample.mean().item()


# ──────────────────────────────────────────────────────────────────────────────
# SSIM
# ──────────────────────────────────────────────────────────────────────────────

def _gaussian_kernel(kernel_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Return a 2-D Gaussian kernel as a 4-D tensor for conv2d."""
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = torch.outer(g, g)
    kernel /= kernel.sum()
    return kernel.view(1, 1, kernel_size, kernel_size)


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
    max_val: float = 1.0,
) -> float:
    """
    Compute the mean Structural Similarity Index (SSIM).

    Args:
        pred        : Predicted image  [B, C, H, W] in [0, max_val].
        target      : Ground-truth     [B, C, H, W] in [0, max_val].
        window_size : Size of Gaussian window.
        sigma       : Standard deviation of Gaussian.
        C1, C2      : Stability constants.
        max_val     : Dynamic range of the pixel values.

    Returns:
        Mean SSIM value (scalar float) in [−1, 1].
    """
    with torch.no_grad():
        kernel = _gaussian_kernel(window_size, sigma).to(pred.device)

        # Process each channel independently, then average
        B, C, H, W = pred.shape
        ssim_vals = []

        for c in range(C):
            x = pred[:, c:c+1, :, :]
            y = target[:, c:c+1, :, :]

            pad = window_size // 2
            mu_x = F.conv2d(x, kernel, padding=pad)
            mu_y = F.conv2d(y, kernel, padding=pad)

            mu_x_sq = mu_x ** 2
            mu_y_sq = mu_y ** 2
            mu_xy   = mu_x * mu_y

            sigma_x  = F.conv2d(x * x, kernel, padding=pad) - mu_x_sq
            sigma_y  = F.conv2d(y * y, kernel, padding=pad) - mu_y_sq
            sigma_xy = F.conv2d(x * y, kernel, padding=pad) - mu_xy

            numerator   = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
            denominator = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)

            ssim_map = numerator / (denominator + 1e-8)
            ssim_vals.append(ssim_map.mean())

        return torch.stack(ssim_vals).mean().item()


# ──────────────────────────────────────────────────────────────────────────────
# Aggregator for epoch-level metrics
# ──────────────────────────────────────────────────────────────────────────────

class MetricTracker:
    """Running mean tracker for PSNR and SSIM across an epoch."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._psnr_sum = 0.0
        self._ssim_sum = 0.0
        self._count    = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        self._psnr_sum += compute_psnr(pred, target)
        self._ssim_sum += compute_ssim(pred, target)
        self._count    += 1

    def result(self) -> Tuple[float, float]:
        """Returns (mean_psnr, mean_ssim)."""
        if self._count == 0:
            return 0.0, 0.0
        return self._psnr_sum / self._count, self._ssim_sum / self._count
