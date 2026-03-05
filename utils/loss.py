"""
utils/loss.py

Combined loss function for snow removal:
  L_total = w1 * L_L1  +  w2 * L_SSIM  +  w3 * L_Perceptual  +  w4 * L_Mask

References:
  • L1 loss         – pixel-level fidelity
  • SSIM loss       – structural similarity (via pytorch-msssim)
  • Perceptual loss – VGG-16 feature matching (via lpips)
  • Mask L1         – supervision on predicted snow mask
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from pytorch_msssim import ssim as compute_ssim
    _MSSSIM_AVAILABLE = True
except ImportError:
    _MSSSIM_AVAILABLE = False
    print("[utils/loss] pytorch-msssim not found – SSIM loss will be disabled.")

try:
    import lpips
    _LPIPS_AVAILABLE = True
except ImportError:
    _LPIPS_AVAILABLE = False
    print("[utils/loss] lpips not found – Perceptual loss will be disabled.")


# ──────────────────────────────────────────────────────────────────────────────

class SnowRemovalLoss(nn.Module):
    """
    Composite loss for joint snow removal and mask prediction.

    Args:
        l1_weight          : Weight for per-pixel L1 loss on clean image.
        ssim_weight        : Weight for SSIM-based loss on clean image.
        perceptual_weight  : Weight for VGG perceptual loss on clean image.
        mask_weight        : Weight for L1 loss on the predicted snow mask.
        device             : Device string used by LPIPS network.
    """

    def __init__(
        self,
        l1_weight:         float = 1.0,
        ssim_weight:       float = 0.5,
        perceptual_weight: float = 0.1,
        mask_weight:       float = 0.5,
        device:            str   = "cpu",
    ):
        super().__init__()
        self.l1_w   = l1_weight
        self.ssim_w = ssim_weight
        self.perc_w = perceptual_weight
        self.mask_w = mask_weight

        self.l1 = nn.L1Loss()

        # SSIM (from pytorch-msssim)
        self.use_ssim = _MSSSIM_AVAILABLE and ssim_weight > 0

        # Perceptual (LPIPS – AlexNet backbone, lightweight)
        self.use_perc = _LPIPS_AVAILABLE and perceptual_weight > 0
        if self.use_perc:
            self.lpips_fn = lpips.LPIPS(net="alex").to(device)
            for p in self.lpips_fn.parameters():
                p.requires_grad = False

    # ------------------------------------------------------------------
    def forward(
        self,
        pred_clean: torch.Tensor,
        gt_clean:   torch.Tensor,
        pred_mask:  torch.Tensor,
        gt_mask:    torch.Tensor,
    ) -> dict:
        """
        Compute the composite loss.

        Args:
            pred_clean : Model-predicted clean image  [B, 3, H, W]  in [0,1].
            gt_clean   : Ground-truth clean image     [B, 3, H, W]  in [0,1].
            pred_mask  : Model-predicted snow mask    [B, 3, H, W]  in [0,1].
            gt_mask    : Ground-truth snow mask       [B, 3, H, W]  in [0,1].

        Returns:
            dict with keys "total", "l1", "ssim", "perceptual", "mask".
        """
        losses = {}

        # L1 on clean image
        l1_loss = self.l1(pred_clean, gt_clean)
        losses["l1"] = l1_loss

        # SSIM
        if self.use_ssim:
            ssim_val = compute_ssim(
                pred_clean, gt_clean,
                data_range=1.0,
                size_average=True,
            )
            losses["ssim"] = 1.0 - ssim_val
        else:
            losses["ssim"] = torch.tensor(0.0, device=pred_clean.device)

        # Perceptual
        if self.use_perc:
            # LPIPS expects [-1, 1] inputs
            p = self.lpips_fn(
                pred_clean * 2 - 1,
                gt_clean   * 2 - 1,
            )
            losses["perceptual"] = p.mean()
        else:
            losses["perceptual"] = torch.tensor(0.0, device=pred_clean.device)

        # Mask L1
        mask_loss = self.l1(pred_mask, gt_mask)
        losses["mask"] = mask_loss

        # Weighted total
        losses["total"] = (
            self.l1_w   * losses["l1"]
            + self.ssim_w * losses["ssim"]
            + self.perc_w * losses["perceptual"]
            + self.mask_w * losses["mask"]
        )

        return losses
