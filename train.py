"""
train.py

Entry point for training the Mask-Guided Residual Reconstruction Network (MGRRN).

Usage:
    python train.py --config configs/config.yaml
    python train.py --config configs/config.yaml --resume checkpoints/last.pth
"""

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets.dataset_loader import build_dataloaders
from models.model import ResidualSnowRemoval
from utils.loss import SnowRemovalLoss
from utils.metrics import MetricTracker
from utils.visualize import log_tensorboard, save_comparison


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_scheduler(optimizer, cfg: dict):
    t = cfg["train"]["lr_scheduler"]
    if t == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["train"]["epochs"]
        )
    elif t == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg["train"]["lr_step_size"],
            gamma=cfg["train"]["lr_gamma"],
        )
    return None


def save_checkpoint(state: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, model, optimizer=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    start_epoch = ckpt.get("epoch", 0) + 1
    best_psnr   = ckpt.get("best_psnr", 0.0)
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"[Resume] Loaded checkpoint from '{path}' (epoch {start_epoch - 1})")
    return start_epoch, best_psnr


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip, writer, epoch):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"[Train] Epoch {epoch}", leave=False)

    for batch in pbar:
        snow  = batch["snow"].to(device)
        mask  = batch["mask"].to(device)
        clean = batch["clean"].to(device)

        optimizer.zero_grad()
        pred_clean, pred_mask = model(snow)
        losses = criterion(pred_clean, clean, pred_mask, mask)
        loss   = losses["total"]
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    writer.add_scalar("Loss/train", avg_loss, epoch)
    return avg_loss


@torch.no_grad()
def validate(model, loader, criterion, device, writer, epoch, vis_dir):
    model.eval()
    total_loss = 0.0
    tracker    = MetricTracker()

    for i, batch in enumerate(tqdm(loader, desc=f"[Val]   Epoch {epoch}", leave=False)):
        snow  = batch["snow"].to(device)
        mask  = batch["mask"].to(device)
        clean = batch["clean"].to(device)

        pred_clean, pred_mask = model(snow)
        losses    = criterion(pred_clean, clean, pred_mask, mask)
        total_loss += losses["total"].item()
        tracker.update(pred_clean, clean)

        # Save first batch comparison
        if i == 0:
            log_tensorboard(writer, "Val/Predictions", snow, pred_clean, clean, epoch)
            save_comparison(
                snow[0], pred_clean[0], clean[0],
                save_path=os.path.join(vis_dir, f"epoch_{epoch:03d}.png"),
                pred_mask=pred_mask[0], gt_mask=mask[0],
                title=f"Epoch {epoch}",
            )

    avg_loss        = total_loss / len(loader)
    mean_psnr, mean_ssim = tracker.result()

    writer.add_scalar("Loss/val",  avg_loss,  epoch)
    writer.add_scalar("PSNR/val",  mean_psnr, epoch)
    writer.add_scalar("SSIM/val",  mean_ssim, epoch)

    return avg_loss, mean_psnr, mean_ssim


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MGRRN Training Script")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config")
    parser.add_argument("--resume", default=None,                   help="Path to checkpoint to resume from")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["train"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(cfg)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ResidualSnowRemoval().to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    # ── Loss ──────────────────────────────────────────────────────────────────
    loss_cfg  = cfg["loss"]
    criterion = SnowRemovalLoss(
        l1_weight         = loss_cfg["l1_weight"],
        ssim_weight       = loss_cfg["ssim_weight"],
        perceptual_weight = loss_cfg["perceptual_weight"],
        mask_weight       = loss_cfg["mask_weight"],
        device            = str(device),
    )

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimizer = optim.Adam(
        model.parameters(),
        lr           = cfg["train"]["learning_rate"],
        weight_decay = cfg["train"]["weight_decay"],
    )
    scheduler = build_scheduler(optimizer, cfg)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 1
    best_psnr   = 0.0
    if args.resume:
        start_epoch, best_psnr = load_checkpoint(args.resume, model, optimizer)

    # ── Logging ───────────────────────────────────────────────────────────────
    log_cfg = cfg["logging"]
    writer  = SummaryWriter(log_dir=log_cfg["log_dir"]) if log_cfg["use_tensorboard"] else None
    vis_dir = os.path.join("outputs", "val_viz")

    # ── Training loop ─────────────────────────────────────────────────────────
    epochs      = cfg["train"]["epochs"]
    val_every   = log_cfg["val_every"]
    save_every  = log_cfg["save_every"]
    ckpt_dir    = log_cfg["checkpoint_dir"]
    grad_clip   = cfg["train"]["grad_clip"]

    print(f"\n{'='*60}")
    print(f" MGRRN Training  |  epochs: {epochs}  |  device: {device}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, epochs + 1):
        t0       = time.time()
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, grad_clip, writer, epoch
        )

        val_loss = psnr = ssim = None
        if epoch % val_every == 0:
            val_loss, psnr, ssim = validate(
                model, val_loader, criterion, device, writer, epoch, vis_dir
            )

        if scheduler:
            scheduler.step()

        # ── Checkpointing ─────────────────────────────────────────────────
        state = {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_psnr":            best_psnr,
        }

        if epoch % save_every == 0:
            save_checkpoint(state, os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pth"))

        if psnr is not None and psnr > best_psnr:
            best_psnr = psnr
            save_checkpoint(state, os.path.join(ckpt_dir, "best_model.pth"))
            print(f"  ✓ New best PSNR: {best_psnr:.2f} dB – checkpoint saved.")

        # Always save the latest
        save_checkpoint(state, os.path.join(ckpt_dir, "last.pth"))

        # ── Console log ───────────────────────────────────────────────────
        elapsed = time.time() - t0
        log_str = (
            f"Epoch [{epoch:03d}/{epochs}] | "
            f"train_loss={train_loss:.4f} | "
            f"time={elapsed:.1f}s"
        )
        if val_loss is not None:
            log_str += f" | val_loss={val_loss:.4f} | PSNR={psnr:.2f} | SSIM={ssim:.4f}"
        print(log_str)

    if writer:
        writer.close()
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
