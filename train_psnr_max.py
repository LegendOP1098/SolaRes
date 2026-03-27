"""
Maximum PSNR Training Script
============================
End-to-end training optimized for achieving PSNR 23+ on solar magnetogram SR.

Key optimizations:
1. No early stopping - trains until completion
2. No batch limits - full dataset every epoch
3. Long training (300+ epochs)
4. Pure L1 pixel loss (best for PSNR)
5. Large model capacity
6. Slow learning rate decay
7. Large patch sizes for better context
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from solarres_sr.data import SolarSRDataset, infer_input_channels, resolve_project_root, resolve_split_dirs, file_md5
from solarres_sr.metrics import calc_psnr, calc_ssim, calc_rmse, calc_correlation
from solarres_sr.registry import MODEL_SPECS, CAPACITY_PRESETS, build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Maximum PSNR training for SR models")
    parser.add_argument("--model", default="swinir", choices=list(MODEL_SPECS.keys()))
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--save-root", default=None)
    parser.add_argument("--epochs", type=int, default=300, help="Total training epochs (default: 300)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-7, help="Very low min LR for fine convergence")
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--capacity", default="large", choices=["tiny", "base", "large"])
    parser.add_argument("--train-hr-patch-size", type=int, default=256, help="Large patch for better context")
    parser.add_argument("--loss-mode", default="l1", choices=["l1", "mse", "charbonnier", "huber"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--use-amp", action="store_true", help="Use mixed precision")
    parser.add_argument("--scheduler", default="cosine_warm_restarts",
                        choices=["cosine", "cosine_warm_restarts", "none"])
    parser.add_argument("--restart-period", type=int, default=50, help="Epochs before LR restart (warm restarts)")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.9999, help="Higher EMA for stability")
    parser.add_argument("--save-every", type=int, default=25, help="Save checkpoint every N epochs")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    return parser.parse_args()


class ModelEMA:
    """Exponential Moving Average for model weights."""

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.state_dict().items():
            if param.is_floating_point():
                self.shadow[name] = param.detach().clone()

    def update(self, model: nn.Module) -> None:
        for name, param in model.state_dict().items():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1 - self.decay)

    def apply_shadow(self, model: nn.Module) -> None:
        for name, param in model.state_dict().items():
            if name in self.shadow:
                self.backup[name] = param.detach().clone()
        model.load_state_dict({**model.state_dict(), **self.shadow}, strict=False)

    def restore(self, model: nn.Module) -> None:
        model.load_state_dict({**model.state_dict(), **self.backup}, strict=False)
        self.backup.clear()


def build_loss(mode: str) -> nn.Module:
    if mode == "l1":
        return nn.L1Loss()
    elif mode == "mse":
        return nn.MSELoss()
    elif mode == "huber":
        return nn.HuberLoss(delta=0.1)
    elif mode == "charbonnier":
        class CharbonnierLoss(nn.Module):
            def __init__(self, eps=1e-6):
                super().__init__()
                self.eps = eps
            def forward(self, pred, target):
                return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps ** 2))
        return CharbonnierLoss()
    raise ValueError(f"Unknown loss mode: {mode}")


def build_scheduler(optimizer, args, total_epochs):
    if args.scheduler == "none":
        return None
    elif args.scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=args.min_learning_rate)
    elif args.scheduler == "cosine_warm_restarts":
        # Warm restarts help escape local minima
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=args.restart_period,
            T_mult=2,  # Double period after each restart
            eta_min=args.min_learning_rate
        )
    raise ValueError(f"Unknown scheduler: {args.scheduler}")


def quick_bicubic_baseline(loader, scale, device, max_batches=20):
    """Compute bicubic baseline PSNR."""
    total_psnr = 0.0
    total_ssim = 0.0
    seen = 0

    with torch.no_grad():
        for batch_idx, (lr_imgs, hr_imgs) in enumerate(loader):
            if batch_idx >= max_batches:
                break
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            # Use only first channel for bicubic
            lr_base = lr_imgs[:, :1]
            bicubic_up = F.interpolate(lr_base, size=hr_imgs.shape[-2:], mode="bicubic", align_corners=False)
            bicubic_up = bicubic_up.clamp(0, 1)

            total_psnr += calc_psnr(bicubic_up, hr_imgs, data_range=1.0, clamp_min=0.0, clamp_max=1.0)
            total_ssim += calc_ssim(bicubic_up, hr_imgs, data_range=1.0, clamp_min=0.0, clamp_max=1.0)
            seen += 1

    return total_psnr / max(seen, 1), total_ssim / max(seen, 1)


def filter_split_leakage(train_pairs, val_pairs):
    """Remove validation samples that overlap with training."""
    train_hashes = {file_md5(hr_path) for _, hr_path in train_pairs}
    return [pair for pair in val_pairs if file_md5(pair[1]) not in train_hashes]


def train_epoch(model, loader, criterion, optimizer, device, use_amp, grad_clip, ema, scaler):
    model.train()
    total_loss = 0.0
    seen = 0

    progress = tqdm(loader, desc="Train", leave=False)
    for lr_imgs, hr_imgs in progress:
        lr_imgs = lr_imgs.to(device, non_blocking=True)
        hr_imgs = hr_imgs.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            sr = model(lr_imgs)
            # Apply sigmoid for [0,1] output
            sr = torch.sigmoid(sr)
            loss = criterion(sr, hr_imgs)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if ema is not None:
            ema.update(model)

        total_loss += loss.item()
        seen += 1
        progress.set_postfix(loss=f"{loss.item():.5f}")

    return total_loss / max(seen, 1)


def validate(model, loader, criterion, device, max_batches=None):
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_rmse = 0.0
    total_corr = 0.0
    seen = 0

    with torch.no_grad():
        for batch_idx, (lr_imgs, hr_imgs) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            lr_imgs = lr_imgs.to(device, non_blocking=True)
            hr_imgs = hr_imgs.to(device, non_blocking=True)

            sr = model(lr_imgs)
            sr = torch.sigmoid(sr).clamp(0, 1)

            loss = criterion(sr, hr_imgs)
            total_loss += loss.item()
            total_psnr += calc_psnr(sr, hr_imgs, data_range=1.0, clamp_min=0.0, clamp_max=1.0)
            total_ssim += calc_ssim(sr, hr_imgs, data_range=1.0, clamp_min=0.0, clamp_max=1.0)
            total_rmse += calc_rmse(sr, hr_imgs, clamp_min=0.0, clamp_max=1.0)
            total_corr += calc_correlation(sr, hr_imgs, clamp_min=0.0, clamp_max=1.0)
            seen += 1

    n = max(seen, 1)
    return total_loss / n, total_psnr / n, total_ssim / n, total_rmse / n, total_corr / n


def main():
    args = parse_args()

    print("=" * 70)
    print("MAXIMUM PSNR TRAINING")
    print("=" * 70)
    print(f"Model: {args.model} | Capacity: {args.capacity}")
    print(f"Epochs: {args.epochs} | Batch size: {args.batch_size}")
    print(f"LR: {args.learning_rate} -> {args.min_learning_rate}")
    print(f"Loss: {args.loss_mode} | Scheduler: {args.scheduler}")
    print(f"Patch size: {args.train_hr_patch_size} | EMA decay: {args.ema_decay}")
    print(f"Device: {args.device} | AMP: {args.use_amp}")
    print("=" * 70)

    # Setup directories
    directories = resolve_split_dirs(args.dataset_root, "train", "val")

    # Determine input mode
    spec = MODEL_SPECS[args.model]
    input_mode = spec.default_input_mode
    in_channels = infer_input_channels(input_mode)
    target_mode = "solar_equalized" if input_mode == "solar_features" else "grayscale"

    print(f"Input mode: {input_mode} ({in_channels} channels)")

    # Build datasets - NO batch limits, full dataset every epoch
    train_dataset = SolarSRDataset(
        directories["train_lr"],
        directories["train_hr"],
        input_mode=input_mode,
        target_mode=target_mode,
        augment=True,
        hr_patch_size=args.train_hr_patch_size,
        random_crop=True,
    )

    val_dataset = SolarSRDataset(
        directories["val_lr"],
        directories["val_hr"],
        input_mode=input_mode,
        target_mode=target_mode,
        augment=False,
        hr_patch_size=None,
        random_crop=False,
    )

    # Filter leakage
    original_val_size = len(val_dataset.pairs)
    val_dataset.pairs = filter_split_leakage(train_dataset.pairs, val_dataset.pairs)
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)} (filtered {original_val_size - len(val_dataset)})")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
    )

    # Bicubic baseline
    bicubic_psnr, bicubic_ssim = quick_bicubic_baseline(val_loader, scale=4, device=args.device)
    print(f"Bicubic baseline: PSNR={bicubic_psnr:.3f} | SSIM={bicubic_ssim:.4f}")

    # Build model
    model = build_model(
        args.model,
        in_channels=in_channels,
        out_channels=1,
        scale=4,
        capacity=args.capacity,
    ).to(args.device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}")

    # Loss, optimizer, scheduler
    criterion = build_loss(args.loss_mode)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.99),
    )
    scheduler = build_scheduler(optimizer, args, args.epochs)

    # EMA
    ema = ModelEMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None

    # AMP scaler
    scaler = torch.amp.GradScaler("cuda", enabled=args.use_amp)

    # Setup save directory
    save_root = Path(args.save_root) if args.save_root else resolve_project_root() / "checkpoints" / "psnr_max"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = save_root / f"{args.model}_{args.capacity}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args)
    config["bicubic_psnr"] = bicubic_psnr
    config["param_count"] = param_count
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    # Resume if specified
    start_epoch = 1
    best_psnr = 0.0
    history = []

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if ema is not None and "ema_state_dict" in checkpoint:
            ema.shadow = checkpoint["ema_state_dict"]
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_psnr = checkpoint.get("best_psnr", 0.0)
        history = checkpoint.get("history", [])
        print(f"Resumed from epoch {start_epoch - 1}, best PSNR: {best_psnr:.3f}")

    print(f"\nStarting training from epoch {start_epoch}...")
    print(f"Checkpoints will be saved to: {run_dir}")
    print("-" * 70)

    # Training loop - NO EARLY STOPPING
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.perf_counter()

        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer,
            args.device, args.use_amp, args.grad_clip, ema, scaler
        )

        # Validate with EMA weights if available
        if ema is not None:
            ema.apply_shadow(model)

        val_loss, val_psnr, val_ssim, val_rmse, val_corr = validate(
            model, val_loader, criterion, args.device
        )

        if ema is not None:
            ema.restore(model)

        # Update scheduler
        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.perf_counter() - epoch_start

        # Record history
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_psnr": val_psnr,
            "val_ssim": val_ssim,
            "val_rmse": val_rmse,
            "val_corr": val_corr,
            "lr": current_lr,
        }
        history.append(record)

        # Print progress
        psnr_gain = val_psnr - bicubic_psnr
        is_best = val_psnr > best_psnr
        best_marker = " *BEST*" if is_best else ""

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train: {train_loss:.5f} | Val: {val_loss:.5f} | "
            f"PSNR: {val_psnr:.3f} (+{psnr_gain:.2f}) | SSIM: {val_ssim:.4f} | "
            f"LR: {current_lr:.2e} | {epoch_time:.1f}s{best_marker}"
        )

        # Save best model
        if is_best:
            best_psnr = val_psnr
            if ema is not None:
                ema.apply_shadow(model)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "ema_state_dict": ema.shadow if ema else None,
                "best_psnr": best_psnr,
                "record": record,
                "history": history,
            }, run_dir / "best_model.pt")
            if ema is not None:
                ema.restore(model)

        # Periodic checkpoint
        if epoch % args.save_every == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "ema_state_dict": ema.shadow if ema else None,
                "best_psnr": best_psnr,
                "history": history,
            }, run_dir / f"checkpoint_epoch_{epoch:04d}.pt")

        # Save history
        (run_dir / "history.json").write_text(json.dumps({"history": history}, indent=2))

    print("=" * 70)
    print(f"Training complete! Best PSNR: {best_psnr:.3f}")
    print(f"Bicubic baseline: {bicubic_psnr:.3f}")
    print(f"Improvement over bicubic: +{best_psnr - bicubic_psnr:.3f} dB")
    print(f"Checkpoints saved to: {run_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
