from __future__ import annotations

import csv
import json
import os
import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import SolarSRDataset, infer_input_channels, resolve_project_root, resolve_split_dirs
from .losses import ReconstructionLoss
from .metrics import calc_psnr, calc_ssim, selection_score
from .registry import MODEL_SPECS, build_discriminator, build_model, candidate_order, suggest_capacity


@dataclass
class TrainConfig:
    model_name: str = "diffusion_sr"
    scale: int = 4
    epochs: int = 50
    batch_size: int = 4
    learning_rate: float = 2e-4
    discriminator_learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    patience: int = 10
    min_delta: float = 1e-4
    scheduler_patience: int = 4
    scheduler_factor: float = 0.5
    num_workers: int = -1
    apply_clahe: bool = False
    input_mode: str = "auto"
    capacity: str = "auto"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = True
    seed: int = 42
    train_split: str = "training"
    val_split: str = "validation"
    max_val_batches: int | None = 200
    diffusion_eval_steps: int = 8
    diffusion_metric_batches: int = 2
    adv_weight: float = 5e-3
    gan_warmup_epochs: int = 2
    save_root: str | None = None
    strict_gpu: bool = True
    target_range: str = "zero_one"
    output_activation: str = "auto"


@dataclass
class TrainingSummary:
    model_name: str
    family: str
    capacity: str
    input_mode: str
    best_epoch: int
    best_val_loss: float
    best_psnr: float
    best_ssim: float
    selection_score: float
    fit_diagnosis: str
    fit_recommendation: str
    run_dir: str
    checkpoint_path: str
    history_path: str
    parameter_count: int


class EarlyStopper:
    def __init__(self, patience: int, min_delta: float = 0.0, mode: str = "max") -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best: float | None = None
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        if self.best is None:
            self.best = value
            return False

        improved = value > self.best + self.min_delta if self.mode == "max" else value < self.best - self.min_delta
        if improved:
            self.best = value
            self.bad_epochs = 0
            return False

        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def resolve_device(requested_device: str, strict_gpu: bool = True) -> str:
    requested = str(requested_device).lower()
    if requested.startswith("cuda"):
        if torch.cuda.is_available():
            return requested
        message = (
            "CUDA was requested but torch.cuda.is_available() is False. "
            "Install a CUDA-enabled PyTorch build in the same environment used to launch training."
        )
        if strict_gpu:
            raise RuntimeError(message)
        print(f"[WARN] {message} Falling back to CPU.")
        return "cpu"
    return requested


def resolve_num_workers(config: TrainConfig) -> int:
    if config.num_workers >= 0:
        resolved_workers = config.num_workers
    else:
        cpu_count = os.cpu_count() or 4
        if str(config.device).startswith("cuda") and torch.cuda.is_available():
            resolved_workers = max(2, min(8, cpu_count // 2))
        else:
            resolved_workers = max(0, min(4, cpu_count // 2))

    # On Windows, CUDA DataLoader worker processes can fail with WinError 1455
    # because each spawned process loads large CUDA DLLs. Keep workers in the
    # main process by default for stability unless explicitly overridden.
    allow_windows_cuda_workers = os.environ.get("SOLARRES_WINDOWS_CUDA_WORKERS", "0") == "1"
    if (
        os.name == "nt"
        and str(config.device).startswith("cuda")
        and torch.cuda.is_available()
        and resolved_workers > 0
        and not allow_windows_cuda_workers
    ):
        print(
            "[WARN] Windows + CUDA detected. Forcing num_workers=0 to avoid "
            "WinError 1455 (paging file too small) in spawned DataLoader workers. "
            "Set SOLARRES_WINDOWS_CUDA_WORKERS=1 to override."
        )
        return 0
    return resolved_workers


def print_runtime_info(device: str, num_workers: int, use_amp: bool) -> None:
    print(f"Runtime | torch={torch.__version__} | device={device} | amp={use_amp} | workers={num_workers}")
    if device.startswith("cuda") and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"CUDA | available=True | gpu={gpu_name} | count={torch.cuda.device_count()}")
    else:
        print("CUDA | available=False")


def quick_bicubic_baseline(
    loader: DataLoader,
    input_mode: str,
    clamp_min: float | None,
    clamp_max: float | None,
    metric_data_range: float,
    max_batches: int = 12,
) -> tuple[float, float]:
    total_psnr = 0.0
    total_ssim = 0.0
    seen = 0
    for batch_index, (lr_imgs, hr_imgs) in enumerate(loader):
        if batch_index >= max_batches:
            break
        lr_base = lr_imgs[:, :1] if input_mode == "solar_features" else lr_imgs[:, :1]
        up = F.interpolate(lr_base, size=hr_imgs.shape[-2:], mode="bicubic", align_corners=False)
        total_psnr += calc_psnr(up, hr_imgs, data_range=metric_data_range, clamp_min=clamp_min, clamp_max=clamp_max)
        total_ssim += calc_ssim(up, hr_imgs, data_range=metric_data_range, clamp_min=clamp_min, clamp_max=clamp_max)
        seen += 1
    if seen == 0:
        return 0.0, 0.0
    return total_psnr / seen, total_ssim / seen


def resolve_target_range(config: TrainConfig) -> tuple[float | None, float | None, float]:
    if config.target_range == "zero_one":
        return 0.0, 1.0, 1.0
    if config.target_range == "minus_one_one":
        return -1.0, 1.0, 2.0
    if config.target_range == "none":
        return None, None, 1.0
    raise ValueError(f"Unsupported target_range: {config.target_range}")


def apply_output_activation(prediction: torch.Tensor, config: TrainConfig) -> torch.Tensor:
    mode = config.output_activation
    if mode == "auto":
        mode = "sigmoid" if config.target_range == "zero_one" else ("tanh" if config.target_range == "minus_one_one" else "identity")
    if mode == "sigmoid":
        return torch.sigmoid(prediction)
    if mode == "tanh":
        return torch.tanh(prediction)
    if mode == "identity":
        return prediction
    raise ValueError(f"Unsupported output_activation: {config.output_activation}")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def amp_context(device: str, use_amp: bool):
    if use_amp and device.startswith("cuda") and torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def build_grad_scaler(device: str, use_amp: bool) -> torch.amp.GradScaler | torch.cuda.amp.GradScaler:
    enabled = use_amp and device.startswith("cuda") and torch.cuda.is_available()
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        from torch.cuda.amp import GradScaler as LegacyGradScaler

        return LegacyGradScaler(enabled=enabled)


def diagnose_fit(history: list[dict[str, float]]) -> tuple[str, str]:
    if len(history) < 4:
        return "insufficient_history", "Run more epochs before reading fit quality too aggressively."

    train_losses = np.array([row["train_loss"] for row in history], dtype=np.float64)
    val_losses = np.array([row["val_loss"] for row in history], dtype=np.float64)
    best_epoch = int(np.argmin(val_losses))
    train_improvement = (train_losses[0] - train_losses[-1]) / max(train_losses[0], 1e-8)
    val_improvement = (val_losses[0] - val_losses[-1]) / max(val_losses[0], 1e-8)
    gap = (val_losses[-1] - train_losses[-1]) / max(abs(val_losses[-1]), 1e-8)

    if best_epoch < len(val_losses) - 1 and val_losses[-1] > val_losses[best_epoch] * 1.05 and gap > 0.15:
        return (
            "overfitting_risk",
            "Validation degraded after the best epoch. Reduce capacity, shorten patience, or increase augmentation.",
        )
    if train_improvement < 0.2 and val_improvement < 0.15:
        return (
            "underfitting_risk",
            "Both train and validation losses are moving too slowly. Increase capacity or train longer with a smaller LR decay.",
        )
    return "well_fit", "The validation curve looks healthy relative to training."


def _save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _save_checkpoint(path: Path, payload: dict) -> None:
    torch.save(payload, path)


def _make_run_dir(config: TrainConfig) -> Path:
    root = Path(config.save_root) if config.save_root else resolve_project_root() / "checkpoints" / "sr_suite"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / f"{config.model_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _build_dataloaders(config: TrainConfig, dataset_root: str | Path | None = None):
    directories = resolve_split_dirs(dataset_root, config.train_split, config.val_split)
    input_mode = config.input_mode if config.input_mode != "auto" else MODEL_SPECS[config.model_name].default_input_mode
    target_mode = "solar_equalized" if input_mode == "solar_features" else "grayscale"
    pin_memory = config.device.startswith("cuda") and torch.cuda.is_available()
    num_workers = resolve_num_workers(config)

    train_dataset = SolarSRDataset(
        directories["train_lr"],
        directories["train_hr"],
        input_mode=input_mode,
        target_mode=target_mode,
        apply_clahe=config.apply_clahe,
        augment=True,
    )
    val_dataset = SolarSRDataset(
        directories["val_lr"],
        directories["val_hr"],
        input_mode=input_mode,
        target_mode=target_mode,
        apply_clahe=config.apply_clahe,
        augment=False,
    )

    train_kwargs = {
        "batch_size": config.batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    val_kwargs = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        train_kwargs["prefetch_factor"] = 4
        val_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_dataset, **train_kwargs)
    val_loader = DataLoader(val_dataset, **val_kwargs)
    return train_loader, val_loader, input_mode, num_workers


def _train_pixel_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: ReconstructionLoss,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    device: str,
    use_amp: bool,
    grad_clip_norm: float,
) -> float:
    model.train()
    scaler = build_grad_scaler(device, use_amp)
    running_loss = 0.0

    progress = tqdm(loader, desc="Train", leave=False)
    for lr_imgs, hr_imgs in progress:
        lr_imgs = lr_imgs.to(device, non_blocking=True)
        hr_imgs = hr_imgs.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with amp_context(device, use_amp):
            sr = apply_output_activation(model(lr_imgs), config)
            loss, _ = criterion(sr, hr_imgs)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / max(len(loader), 1)


def _train_gan_epoch(
    generator: nn.Module,
    discriminator: nn.Module,
    loader: DataLoader,
    criterion: ReconstructionLoss,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    config: TrainConfig,
    device: str,
    adv_weight: float,
) -> float:
    generator.train()
    discriminator.train()
    bce = nn.BCEWithLogitsLoss()
    running_loss = 0.0

    progress = tqdm(loader, desc="Train", leave=False)
    for lr_imgs, hr_imgs in progress:
        lr_imgs = lr_imgs.to(device, non_blocking=True)
        hr_imgs = hr_imgs.to(device, non_blocking=True)

        optimizer_d.zero_grad(set_to_none=True)
        with torch.no_grad():
            fake_imgs = apply_output_activation(generator(lr_imgs), config)
        real_logits = discriminator(hr_imgs)
        fake_logits = discriminator(fake_imgs.detach())
        real_targets = torch.full_like(real_logits, 0.9)
        fake_targets = torch.zeros_like(fake_logits)
        d_loss = 0.5 * (bce(real_logits, real_targets) + bce(fake_logits, fake_targets))
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
        optimizer_d.step()

        optimizer_g.zero_grad(set_to_none=True)
        fake_imgs = apply_output_activation(generator(lr_imgs), config)
        fake_logits = discriminator(fake_imgs)
        recon_loss, _ = criterion(fake_imgs, hr_imgs)
        g_adv = bce(fake_logits, torch.ones_like(fake_logits))
        g_loss = recon_loss + adv_weight * g_adv
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        optimizer_g.step()

        running_loss += g_loss.item()
        progress.set_postfix(g_loss=f"{g_loss.item():.4f}", d_loss=f"{d_loss.item():.4f}")

    return running_loss / max(len(loader), 1)


def _train_diffusion_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    use_amp: bool,
    grad_clip_norm: float,
) -> float:
    model.train()
    scaler = build_grad_scaler(device, use_amp)
    running_loss = 0.0

    progress = tqdm(loader, desc="Train", leave=False)
    for lr_imgs, hr_imgs in progress:
        lr_imgs = lr_imgs.to(device, non_blocking=True)
        hr_imgs = hr_imgs.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with amp_context(device, use_amp):
            loss = model.training_loss(lr_imgs, hr_imgs)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / max(len(loader), 1)


def _validate_reconstruction(
    model: nn.Module,
    loader: DataLoader,
    criterion: ReconstructionLoss,
    config: TrainConfig,
    clamp_min: float | None,
    clamp_max: float | None,
    metric_data_range: float,
    device: str,
    max_batches: int | None = None,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    seen = 0

    with torch.no_grad():
        for batch_index, (lr_imgs, hr_imgs) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            lr_imgs = lr_imgs.to(device, non_blocking=True)
            hr_imgs = hr_imgs.to(device, non_blocking=True)
            sr = apply_output_activation(model(lr_imgs), config)
            loss, _ = criterion(sr, hr_imgs)
            total_loss += loss.item()
            total_psnr += calc_psnr(sr, hr_imgs, data_range=metric_data_range, clamp_min=clamp_min, clamp_max=clamp_max)
            total_ssim += calc_ssim(sr, hr_imgs, data_range=metric_data_range, clamp_min=clamp_min, clamp_max=clamp_max)
            seen += 1

    if seen == 0:
        return float("inf"), 0.0, 0.0
    return total_loss / seen, total_psnr / seen, total_ssim / seen


def _validate_diffusion(
    model: nn.Module,
    loader: DataLoader,
    clamp_min: float | None,
    clamp_max: float | None,
    metric_data_range: float,
    device: str,
    eval_steps: int,
    metric_batches: int,
    max_batches: int | None = None,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    loss_batches = 0
    metric_seen = 0

    with torch.no_grad():
        for batch_index, (lr_imgs, hr_imgs) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            lr_imgs = lr_imgs.to(device, non_blocking=True)
            hr_imgs = hr_imgs.to(device, non_blocking=True)
            loss = model.training_loss(lr_imgs, hr_imgs)
            total_loss += loss.item()
            loss_batches += 1

            if metric_seen < metric_batches:
                sr = model.sample(lr_imgs, hr_imgs.shape[-2:], sample_steps=eval_steps)
                total_psnr += calc_psnr(sr, hr_imgs, data_range=metric_data_range, clamp_min=clamp_min, clamp_max=clamp_max)
                total_ssim += calc_ssim(sr, hr_imgs, data_range=metric_data_range, clamp_min=clamp_min, clamp_max=clamp_max)
                metric_seen += 1

    avg_loss = total_loss / max(loss_batches, 1)
    avg_psnr = total_psnr / max(metric_seen, 1)
    avg_ssim = total_ssim / max(metric_seen, 1)
    return avg_loss, avg_psnr, avg_ssim


def fit_model(
    config: TrainConfig,
    dataset_root: str | Path | None = None,
) -> TrainingSummary:
    seed_everything(config.seed)
    resolved_device = resolve_device(config.device, strict_gpu=config.strict_gpu)
    config = replace(config, device=resolved_device)
    train_loader, val_loader, input_mode, resolved_workers = _build_dataloaders(config, dataset_root)
    print_runtime_info(config.device, resolved_workers, config.use_amp)
    train_size = len(train_loader.dataset)
    capacity = config.capacity if config.capacity != "auto" else suggest_capacity(config.model_name, train_size)
    family = MODEL_SPECS[config.model_name].family
    in_channels = infer_input_channels(input_mode)
    run_dir = _make_run_dir(config)
    checkpoint_path = run_dir / "best_model.pt"
    history_path = run_dir / "history.json"

    effective_config = replace(config, input_mode=input_mode, capacity=capacity, num_workers=resolved_workers)
    _save_json(run_dir / "config.json", asdict(effective_config))

    clamp_min, clamp_max, metric_data_range = resolve_target_range(config)
    bicubic_psnr, bicubic_ssim = quick_bicubic_baseline(
        val_loader,
        input_mode=input_mode,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
        metric_data_range=metric_data_range,
        max_batches=12,
    )
    print(f"Sanity | bicubic baseline PSNR={bicubic_psnr:.3f} SSIM={bicubic_ssim:.4f}")
    if bicubic_psnr < 12.0:
        print("[WARN] Very low bicubic baseline detected. Check LR-HR pairing, normalization, and data range settings.")

    criterion = ReconstructionLoss(
        ssim_data_range=metric_data_range,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
    ).to(config.device)
    model = build_model(config.model_name, in_channels=in_channels, out_channels=1, scale=config.scale, capacity=capacity).to(config.device)
    parameter_count = count_parameters(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=config.scheduler_factor, patience=config.scheduler_patience)

    discriminator = None
    optimizer_d = None
    if family == "gan":
        discriminator = build_discriminator(config.model_name, in_channels=1).to(config.device)
        optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=config.discriminator_learning_rate, weight_decay=config.weight_decay)

    early_stopper = EarlyStopper(config.patience, min_delta=config.min_delta, mode="max")
    history: list[dict[str, float]] = []
    best_record: dict[str, float] | None = None
    best_score = float("-inf")

    for epoch in range(1, config.epochs + 1):
        if family == "pixel":
            train_loss = _train_pixel_epoch(model, train_loader, criterion, optimizer, config, config.device, config.use_amp, config.grad_clip_norm)
            val_loss, val_psnr, val_ssim = _validate_reconstruction(
                model, val_loader, criterion, config, clamp_min, clamp_max, metric_data_range, config.device, config.max_val_batches
            )
        elif family == "gan":
            current_adv_weight = config.adv_weight if epoch > config.gan_warmup_epochs else 0.0
            train_loss = _train_gan_epoch(
                model, discriminator, train_loader, criterion, optimizer, optimizer_d, config, config.device, current_adv_weight
            )
            val_loss, val_psnr, val_ssim = _validate_reconstruction(
                model, val_loader, criterion, config, clamp_min, clamp_max, metric_data_range, config.device, config.max_val_batches
            )
        else:
            train_loss = _train_diffusion_epoch(model, train_loader, optimizer, config.device, config.use_amp, config.grad_clip_norm)
            val_loss, val_psnr, val_ssim = _validate_diffusion(
                model,
                val_loader,
                clamp_min,
                clamp_max,
                metric_data_range,
                config.device,
                config.diffusion_eval_steps,
                config.diffusion_metric_batches,
                config.max_val_batches,
            )

        scheduler.step(val_loss)
        score = selection_score(val_psnr, val_ssim, val_loss)
        record = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_psnr": float(val_psnr),
            "val_ssim": float(val_ssim),
            "selection_score": float(score),
        }
        history.append(record)

        print(
            f"Epoch {epoch}/{config.epochs} | train={train_loss:.4f} | val={val_loss:.4f} | "
            f"PSNR={val_psnr:.3f} | SSIM={val_ssim:.4f} | score={score:.3f}"
        )

        if score > best_score + config.min_delta:
            best_score = score
            best_record = record
            payload = {
                "model_name": config.model_name,
                "family": family,
                "capacity": capacity,
                "input_mode": input_mode,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "record": record,
                "config": asdict(effective_config),
            }
            if discriminator is not None and optimizer_d is not None:
                payload["discriminator_state_dict"] = discriminator.state_dict()
                payload["optimizer_d_state_dict"] = optimizer_d.state_dict()
            _save_checkpoint(checkpoint_path, payload)

        if early_stopper.step(score):
            print("Early stopping triggered on validation score.")
            break

    if best_record is None:
        raise RuntimeError("Training finished without producing a valid checkpoint.")

    _save_json(history_path, {"history": history})
    fit_diagnosis, fit_recommendation = diagnose_fit(history)
    summary = TrainingSummary(
        model_name=config.model_name,
        family=family,
        capacity=capacity,
        input_mode=input_mode,
        best_epoch=int(best_record["epoch"]),
        best_val_loss=float(best_record["val_loss"]),
        best_psnr=float(best_record["val_psnr"]),
        best_ssim=float(best_record["val_ssim"]),
        selection_score=float(best_record["selection_score"]),
        fit_diagnosis=fit_diagnosis,
        fit_recommendation=fit_recommendation,
        run_dir=str(run_dir),
        checkpoint_path=str(checkpoint_path),
        history_path=str(history_path),
        parameter_count=parameter_count,
    )
    _save_json(run_dir / "summary.json", asdict(summary))
    return summary


def benchmark_models(
    base_config: TrainConfig,
    model_names: list[str] | None = None,
    dataset_root: str | Path | None = None,
) -> list[TrainingSummary]:
    models = model_names or candidate_order(diffusion_centric=True)
    benchmark_root = Path(base_config.save_root) if base_config.save_root else resolve_project_root() / "checkpoints" / "sr_benchmark"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    benchmark_dir = benchmark_root / timestamp
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[TrainingSummary] = []
    for model_name in models:
        run_config = replace(base_config, model_name=model_name, save_root=str(benchmark_dir))
        summaries.append(fit_model(run_config, dataset_root=dataset_root))

    summaries.sort(key=lambda item: item.selection_score, reverse=True)

    leaderboard = [asdict(summary) for summary in summaries]
    _save_json(benchmark_dir / "leaderboard.json", {"leaderboard": leaderboard})
    with (benchmark_dir / "leaderboard.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=leaderboard[0].keys())
        writer.writeheader()
        writer.writerows(leaderboard)

    return summaries
