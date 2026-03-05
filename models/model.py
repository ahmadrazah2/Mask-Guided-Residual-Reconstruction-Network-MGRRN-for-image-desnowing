"""
models/model.py

Mask-Guided Residual Reconstruction Network (MGRRN) for Image Desnowing.

Architecture:
  - SimpleFusionNet   : predicts a 3-channel snow mask
  - ResidualReconstructNet : U-Net-style decoder → residual image
  - ResidualSnowRemoval    : end-to-end wrapper (mask + reconstruction)

Reference:
  Ahmad Raza Hussain, H.S. Lee, and H.S. Lee (2026).
  Snow Removal in Images Using a Deep Learning-Based Residual Restoration Neural Network.
  Journal of the Korea Institute of Information and Communication Engineering, 30(1), 92-102.
  https://doi.org/10.6109/jkiice.2026.30.1.92
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Building Block
# ──────────────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Double convolution block: Conv-BN-ReLU × 2."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ──────────────────────────────────────────────────────────────────────────────
# Branch 1 – Snow Mask Predictor
# ──────────────────────────────────────────────────────────────────────────────

class SimpleFusionNet(nn.Module):
    """
    Multi-scale fusion network that predicts a 3-channel soft snow mask.

    Feature maps from three resolution levels are concatenated and fused
    before producing the final output.

    Args:
        in_channels: Number of input image channels (default: 3 for RGB).
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()

        self.block1 = ConvBlock(in_channels, 32)
        self.block2 = ConvBlock(32, 64)
        self.block3 = ConvBlock(64, 128)

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(32 + 64 + 128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # 3-channel mask (matches RGB snow intensity per channel)
        self.out_conv = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.block1(x)         # [B, 32, H, W]
        x2 = self.block2(x1)        # [B, 64, H, W]
        x3 = self.block3(x2)        # [B, 128, H, W]

        fused = torch.cat([x1, x2, x3], dim=1)   # [B, 224, H, W]
        fusion = self.fusion_conv(fused)           # [B, 64, H, W]
        out = self.out_conv(fusion)                # [B, 3, H, W]

        return torch.relu(out)


# ──────────────────────────────────────────────────────────────────────────────
# Branch 2 – Residual Reconstruction (U-Net style)
# ──────────────────────────────────────────────────────────────────────────────

class ResidualReconstructNet(nn.Module):
    """
    U-Net-style encoder-decoder that predicts the snow residual to subtract
    from the snowy image in order to recover the clean image.

    Args:
        in_channels: Concatenated input channels (snowy image + mask = 6).
    """

    def __init__(self, in_channels: int = 6):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)

        # Bottleneck
        self.middle = ConvBlock(256, 256)

        # Decoder with skip connections
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2    = ConvBlock(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1    = ConvBlock(128, 64)

        self.out_conv = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode
        e1  = self.enc1(x)                         # [B, 64,  H,   W  ]
        e2  = self.enc2(F.max_pool2d(e1, 2))        # [B, 128, H/2, W/2]
        e3  = self.enc3(F.max_pool2d(e2, 2))        # [B, 256, H/4, W/4]
        mid = self.middle(e3)                        # [B, 256, H/4, W/4]

        # Decode
        d2 = self.upconv2(mid)                       # [B, 128, H/2, W/2]
        d2 = self.dec2(torch.cat([d2, e2], dim=1))  # skip from enc2

        d1 = self.upconv1(d2)                        # [B, 64,  H,   W  ]
        d1 = self.dec1(torch.cat([d1, e1], dim=1))  # skip from enc1

        residual = self.out_conv(d1)                 # [B, 3,   H,   W  ]
        return torch.relu(residual)


# ──────────────────────────────────────────────────────────────────────────────
# Combined Model – MGRRN
# ──────────────────────────────────────────────────────────────────────────────

class ResidualSnowRemoval(nn.Module):
    """
    Mask-Guided Residual Reconstruction Network (MGRRN).

    Pipeline:
        1. SimpleFusionNet predicts a soft snow mask M from the snowy image.
        2. The snowy image and mask are concatenated → fed to ResidualReconstructNet.
        3. The reconstruction net predicts a residual R.
        4. Clean image = clamp(snowy – R, 0, 1).

    Returns:
        clean_img  (torch.Tensor): Restored clean image [B, 3, H, W].
        mask_pred  (torch.Tensor): Predicted snow mask  [B, 3, H, W].
    """

    def __init__(self):
        super().__init__()
        self.mask_net        = SimpleFusionNet(in_channels=3)
        self.reconstruct_net = ResidualReconstructNet(in_channels=6)

    def forward(self, x_snowy: torch.Tensor):
        mask_pred  = self.mask_net(x_snowy)                      # [B,3,H,W]
        x_combined = torch.cat([x_snowy, mask_pred], dim=1)     # [B,6,H,W]
        residual   = self.reconstruct_net(x_combined)            # [B,3,H,W]
        clean_img  = torch.clamp(x_snowy - residual, 0.0, 1.0)  # [B,3,H,W]
        return clean_img, mask_pred


# ──────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = ResidualSnowRemoval()
    dummy = torch.randn(2, 3, 256, 256)
    clean, mask = model(dummy)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Clean  shape : {clean.shape}")
    print(f"Mask   shape : {mask.shape}")
    print(f"Trainable parameters: {total_params:,}")
