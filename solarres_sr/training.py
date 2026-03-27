from __future__ import annotations

import csv
import json
import os
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import SolarSRDataset, infer_input_channels, resolve_project_root, resolve_split_dirs
from .data import file_md5
from .losses import ReconstructionLoss
from .metrics import calc_correlation, calc_psnr, calc_rmse, calc_ssim, selection_score
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
    optimizer: str = "auto"
    scheduler: str = "auto"
    warmup_epochs: int = 2
    min_learning_rate: float = 1e-6
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    grad_clip_norm: float = 1.0
    ema_decay: float = 0.999
    use_ema_for_eval: bool = True
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
    diffusion_sampler: str = "ddpm"
    diffusion_stabilize_eval: bool = True
    diffusion_clamp_pred_x0: bool = True
    diffusion_self_condition: bool = True
    diffusion_recon_weight: float = 0.2
    diffusion_guide_weight: float = 0.1
    diffusion_x0_weight: float = 0.25
    diffusion_timestep_bias: float = 1.5
    adv_weight: float = 5e-3
    gan_warmup_epochs: int = 2
    save_root: str | None = None
    strict_gpu: bool = True
    target_range: str = "zero_one"
    output_activation: str = "auto"
    loss_pixel_mode: str = "charbonnier"
    loss_pixel_weight: float = 1.0
    loss_ssim_weight: float = 0.15
    loss_edge_weight: float = 0.1
    loss_fft_weight: float = 0.0
    train_hr_patch_size: int | None = 192
    deduplicate_splits_by_hr_hash: bool = True
    profile_timing: bool = True
    max_train_batches: int | None = None
    selection_metric: str = "score"
    verbose_debug: bool = False


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
    best_rmse: float
    best_correlation: float
    bicubic_psnr: float
    selection_score: float
    selected_metric: str
    selected_metric_value: float
    beat_bicubic: bool
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


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}
        for name, tensor in model.state_dict().items():
            if tensor.is_floating_point():
                self.shadow[name] = tensor.detach().clone()

    def update(self, model: nn.Module) -> None:
        if not self.shadow:
            return
        with torch.no_grad():
            state = model.state_dict()
            one_minus = 1.0 - self.decay
            for name, shadow_tensor in self.shadow.items():
                current = state[name].detach()
                shadow_tensor.mul_(self.decay).add_(current, alpha=one_minus)

    def apply_shadow(self, model: nn.Module) -> None:
        self.backup = {}
        state = model.state_dict()
        for name, shadow_tensor in self.shadow.items():
            self.backup[name] = state[name].detach().clone()
            state[name].copy_(shadow_tensor)

    def restore(self, model: nn.Module) -> None:
        if not self.backup:
            return
        state = model.state_dict()
        for name, backup_tensor in self.backup.items():
            state[name].copy_(backup_tensor)
        self.backup = {}

    def state_dict(self) -> dict[str, object]:
        return {
            "decay": self.decay,
            "shadow": {name: tensor.detach().cpu() for name, tensor in self.shadow.items()},
        }


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


def tensor_stats(name: str, tensor: torch.Tensor) -> str:
    tensor = tensor.detach()
    return (
        f"{name}: shape={tuple(tensor.shape)} min={tensor.min().item():.4f} "
        f"max={tensor.max().item():.4f} mean={tensor.mean().item():.4f} std={tensor.std().item():.4f}"
    )


def verify_lr_hr_alignment(lr_tensor: torch.Tensor, hr_tensor: torch.Tensor, expected_scale: int) -> None:
    if lr_tensor.ndim != 4 or hr_tensor.ndim != 4:
        raise RuntimeError(
            f"Expected batched tensors with shape [N,C,H,W], got lr={tuple(lr_tensor.shape)} hr={tuple(hr_tensor.shape)}"
        )
    lr_h, lr_w = lr_tensor.shape[-2:]
    hr_h, hr_w = hr_tensor.shape[-2:]
    if lr_h <= 0 or lr_w <= 0:
        raise RuntimeError(f"Invalid LR spatial shape lr={tuple(lr_tensor.shape)}")
    if hr_h != lr_h * expected_scale or hr_w != lr_w * expected_scale:
        raise RuntimeError(
            "LR/HR scale mismatch detected. "
            f"expected_scale={expected_scale} lr_hw=({lr_h},{lr_w}) hr_hw=({hr_h},{hr_w})"
        )


def verify_model_device(model: nn.Module, device: str) -> None:
    devices = {str(parameter.device) for parameter in model.parameters()}
    if len(devices) != 1:
        raise RuntimeError(f"Model parameters are split across devices. Found devices={devices}")
    only_device = next(iter(devices))
    if str(device).startswith("cuda"):
        ok = only_device.startswith("cuda")
    else:
        ok = only_device == str(device)
    if not ok:
        raise RuntimeError(f"Model parameters are not fully on requested device={device}. Found devices={devices}")
    print(f"Model device check | parameters_on={only_device}")


def filter_split_leakage_by_hr_hash(train_pairs: list[tuple[Path, Path]], val_pairs: list[tuple[Path, Path]]) -> list[tuple[Path, Path]]:
    train_hashes = {file_md5(hr_path) for _, hr_path in train_pairs}
    return [pair for pair in val_pairs if file_md5(pair[1]) not in train_hashes]


def quick_bicubic_baseline(
    loader: DataLoader,
    input_mode: str,
    clamp_min: float | None,
    clamp_max: float | None,
    metric_data_range: float,
    expected_scale: int,
    max_batches: int = 12,
) -> tuple[float, float, float, float]:
    total_psnr = 0.0
    total_ssim = 0.0
    total_rmse = 0.0
    total_correlation = 0.0
    seen = 0
    for batch_index, (lr_imgs, hr_imgs) in enumerate(loader):
        if batch_index >= max_batches:
            break
        if batch_index == 0:
            verify_lr_hr_alignment(lr_imgs, hr_imgs, expected_scale=expected_scale)
        lr_base = lr_imgs[:, :1] if input_mode == "solar_features" else lr_imgs[:, :1]
        up = F.interpolate(lr_base, size=hr_imgs.shape[-2:], mode="bicubic", align_corners=False)
        total_psnr += calc_psnr(up, hr_imgs, data_range=metric_data_range, clamp_min=clamp_min, clamp_max=clamp_max)
        total_ssim += calc_ssim(up, hr_imgs, data_range=metric_data_range, clamp_min=clamp_min, clamp_max=clamp_max)
        total_rmse += calc_rmse(up, hr_imgs, clamp_min=clamp_min, clamp_max=clamp_max)
        total_correlation += calc_correlation(up, hr_imgs, clamp_min=clamp_min, clamp_max=clamp_max)
        seen += 1
    if seen == 0:
        return 0.0, 0.0, 0.0, 0.0
    return total_psnr / seen, total_ssim / seen, total_rmse / seen, total_correlation / seen


def resolve_target_range(config: TrainConfig) -> tuple[float | None, float | None, float]:
    if config.target_range == "zero_one":
        return 0.0, 1.0, 1.0
    if config.target_range == "minus_one_one":
        return -1.0, 1.0, 2.0
    if config.target_range == "none":
        return None, None, 1.0
    raise ValueError(f"Unsupported target_range: {config.target_range}")


def compute_selection_metric(
    model_name: str,
    psnr: float,
    ssim_value: float,
    val_loss: float,
    bicubic_psnr: float,
    chosen_metric: str,
) -> float:
    metric = chosen_metric or "score"
    family = MODEL_SPECS[model_name].family
    if metric == "score":
        if family == "diffusion":
            # Diffusion val_loss is noise-prediction loss (different scale/meaning
            # from reconstruction losses), so do not subtract it for model ranking.
            return float(psnr + 20.0 * ssim_value)
        return float(selection_score(psnr, ssim_value, val_loss))
    if metric == "psnr":
        return float(psnr)
    if metric == "ssim":
        return float(ssim_value)
    if metric == "bicubic_gap_psnr":
        # Compare by reconstruction quality relative to bicubic, independent of
        # diffusion noise-prediction loss scale.
        return float(psnr - bicubic_psnr)
    if family == "diffusion":
        raise ValueError(f"Unsupported diffusion selection metric: {metric}")
    raise ValueError(f"Unsupported selection metric: {metric}")


def summary_sort_key(summary: TrainingSummary) -> tuple[float, float, float]:
    return (
        float(summary.selected_metric_value),
        float(summary.best_psnr),
        float(summary.best_ssim),
    )


def _should_print_tensor_stats(debug_state: dict[str, bool] | None, key: str, verbose_debug: bool) -> bool:
    if verbose_debug:
        return True
    if debug_state is None:
        return True
    return not bool(debug_state.get(key, False))


def _mark_tensor_stats_printed(debug_state: dict[str, bool] | None, key: str) -> None:
    if debug_state is None:
        return
    debug_state[key] = True


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


def snapshot_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()}


def resolve_optimizer_name(config: TrainConfig, family: str) -> str:
    if config.optimizer != "auto":
        return config.optimizer
    if family in {"pixel", "diffusion", "gan"}:
        return "adam"
    return "adamw"


def build_optimizer(
    params,
    config: TrainConfig,
    family: str,
    lr: float,
) -> torch.optim.Optimizer:
    optimizer_name = resolve_optimizer_name(config, family)
    if optimizer_name == "adam":
        return torch.optim.Adam(
            params,
            lr=lr,
            betas=(config.adam_beta1, config.adam_beta2),
            weight_decay=config.weight_decay,
        )
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=lr,
            betas=(config.adam_beta1, config.adam_beta2),
            weight_decay=config.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def resolve_scheduler_name(config: TrainConfig, family: str) -> str:
    if config.scheduler != "auto":
        return config.scheduler
    if family in {"pixel", "diffusion"}:
        return "cosine"
    return "plateau"


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    family: str,
):
    scheduler_name = resolve_scheduler_name(config, family)
    if scheduler_name == "none":
        return None, scheduler_name
    if scheduler_name == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
            min_lr=config.min_learning_rate,
        )
        return scheduler, scheduler_name
    if scheduler_name == "cosine":
        effective_epochs = max(1, config.epochs - max(config.warmup_epochs, 0))
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=effective_epochs,
            eta_min=config.min_learning_rate,
        )
        return scheduler, scheduler_name
    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def apply_warmup_lr(optimizer: torch.optim.Optimizer, base_lr: float, epoch: int, warmup_epochs: int) -> None:
    if warmup_epochs <= 0 or epoch > warmup_epochs:
        return
    # Linear warmup from 1/(warmup_epochs+1) to 1.0 over warmup_epochs
    # Epoch 1 gets base_lr * 1/(warmup_epochs+1), last warmup epoch gets base_lr * warmup_epochs/(warmup_epochs+1)
    # After warmup, LR returns to normal scheduler control
    warmup_scale = float(epoch) / float(warmup_epochs + 1)
    warmup_lr = max(base_lr * warmup_scale, 1e-8)  # Ensure non-zero LR
    for param_group in optimizer.param_groups:
        param_group["lr"] = warmup_lr


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    if not optimizer.param_groups:
        return 0.0
    return float(optimizer.param_groups[0].get("lr", 0.0))


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
        hr_patch_size=config.train_hr_patch_size,
        random_crop=True,
    )
    val_dataset = SolarSRDataset(
        directories["val_lr"],
        directories["val_hr"],
        input_mode=input_mode,
        target_mode=target_mode,
        apply_clahe=config.apply_clahe,
        augment=False,
        hr_patch_size=None,
        random_crop=False,
    )

    if config.deduplicate_splits_by_hr_hash:
        original_val_size = len(val_dataset.pairs)
        val_dataset.pairs = filter_split_leakage_by_hr_hash(train_dataset.pairs, val_dataset.pairs)
        removed = original_val_size - len(val_dataset.pairs)
        print(f"Split leakage filter | removed_from_val={removed} | val_size={len(val_dataset.pairs)}")

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
    scaler: torch.amp.GradScaler | torch.cuda.amp.GradScaler,
    ema: ModelEMA | None = None,
    max_batches: int | None = None,
    debug_state: dict[str, bool] | None = None,
    verbose_debug: bool = False,
) -> tuple[float, dict[str, float]]:
    model.train()
    running_loss = 0.0
    data_time = 0.0
    forward_time = 0.0
    backward_time = 0.0
    batch_end = time.perf_counter()
    printed_stats = False

    progress = tqdm(loader, desc="Train", leave=False)
    seen = 0
    for batch_index, (lr_imgs, hr_imgs) in enumerate(progress):
        if max_batches is not None and batch_index >= max_batches:
            break
        data_time += time.perf_counter() - batch_end
        lr_imgs = lr_imgs.to(device, non_blocking=True)
        hr_imgs = hr_imgs.to(device, non_blocking=True)
        if not printed_stats:
            verify_lr_hr_alignment(lr_imgs, hr_imgs, expected_scale=config.scale)
        optimizer.zero_grad(set_to_none=True)

        forward_start = time.perf_counter()
        with amp_context(device, use_amp):
            sr = apply_output_activation(model(lr_imgs), config)
            loss, _ = criterion(sr, hr_imgs)
        forward_time += time.perf_counter() - forward_start

        backward_start = time.perf_counter()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update(model)
        backward_time += time.perf_counter() - backward_start

        running_loss += loss.item()
        seen += 1
        progress.set_postfix(loss=f"{loss.item():.4f}")
        if not printed_stats and _should_print_tensor_stats(debug_state, "train_batch_stats", verbose_debug):
            print(tensor_stats("train.lr", lr_imgs))
            print(tensor_stats("train.hr", hr_imgs))
            print(tensor_stats("train.pred", sr))
            printed_stats = True
            _mark_tensor_stats_printed(debug_state, "train_batch_stats")
        batch_end = time.perf_counter()

    steps = max(seen, 1)
    return running_loss / steps, {
        "data_time": data_time / steps,
        "forward_time": forward_time / steps,
        "backward_time": backward_time / steps,
    }


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
    scaler: torch.amp.GradScaler | torch.cuda.amp.GradScaler,
    ema: ModelEMA | None = None,
    max_batches: int | None = None,
    debug_state: dict[str, bool] | None = None,
    verbose_debug: bool = False,
) -> tuple[float, dict[str, float]]:
    generator.train()
    discriminator.train()
    bce = nn.BCEWithLogitsLoss()
    running_loss = 0.0
    data_time = 0.0
    forward_time = 0.0
    backward_time = 0.0
    batch_end = time.perf_counter()
    printed_stats = False
    use_amp = config.use_amp

    # Consistent label smoothing: 0.9 for real, 0.0 for fake
    # Generator tries to make discriminator output 0.9 (same as real target)
    real_label = 0.9
    fake_label = 0.0

    progress = tqdm(loader, desc="Train", leave=False)
    seen = 0
    for batch_index, (lr_imgs, hr_imgs) in enumerate(progress):
        if max_batches is not None and batch_index >= max_batches:
            break
        data_time += time.perf_counter() - batch_end
        lr_imgs = lr_imgs.to(device, non_blocking=True)
        hr_imgs = hr_imgs.to(device, non_blocking=True)
        if not printed_stats:
            verify_lr_hr_alignment(lr_imgs, hr_imgs, expected_scale=config.scale)

        forward_start = time.perf_counter()

        # Discriminator step with AMP
        optimizer_d.zero_grad(set_to_none=True)
        with amp_context(device, use_amp):
            with torch.no_grad():
                fake_imgs = apply_output_activation(generator(lr_imgs), config)
            real_logits = discriminator(hr_imgs)
            fake_logits = discriminator(fake_imgs.detach())
            real_targets = torch.full_like(real_logits, real_label)
            fake_targets = torch.full_like(fake_logits, fake_label)
            d_loss = 0.5 * (bce(real_logits, real_targets) + bce(fake_logits, fake_targets))

        scaler.scale(d_loss).backward()
        scaler.unscale_(optimizer_d)
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
        scaler.step(optimizer_d)
        scaler.update()

        # Generator step with AMP
        optimizer_g.zero_grad(set_to_none=True)
        with amp_context(device, use_amp):
            fake_imgs = apply_output_activation(generator(lr_imgs), config)
            fake_logits = discriminator(fake_imgs)
            recon_loss, _ = criterion(fake_imgs, hr_imgs)
            # Generator wants discriminator to think fake is real (use real_label for consistency)
            g_adv = bce(fake_logits, torch.full_like(fake_logits, real_label))
            g_loss = recon_loss + adv_weight * g_adv

        forward_time += time.perf_counter() - forward_start

        backward_start = time.perf_counter()
        scaler.scale(g_loss).backward()
        scaler.unscale_(optimizer_g)
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        scaler.step(optimizer_g)
        scaler.update()

        if ema is not None:
            ema.update(generator)
        backward_time += time.perf_counter() - backward_start

        running_loss += g_loss.item()
        seen += 1
        progress.set_postfix(g_loss=f"{g_loss.item():.4f}", d_loss=f"{d_loss.item():.4f}")
        if not printed_stats and _should_print_tensor_stats(debug_state, "train_batch_stats", verbose_debug):
            print(tensor_stats("train.lr", lr_imgs))
            print(tensor_stats("train.hr", hr_imgs))
            print(tensor_stats("train.pred", fake_imgs))
            printed_stats = True
            _mark_tensor_stats_printed(debug_state, "train_batch_stats")
        batch_end = time.perf_counter()

    steps = max(seen, 1)
    return running_loss / steps, {
        "data_time": data_time / steps,
        "forward_time": forward_time / steps,
        "backward_time": backward_time / steps,
    }


def _train_diffusion_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: ReconstructionLoss,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    clamp_min: float | None,
    clamp_max: float | None,
    device: str,
    use_amp: bool,
    grad_clip_norm: float,
    expected_scale: int,
    scaler: torch.amp.GradScaler | torch.cuda.amp.GradScaler,
    ema: ModelEMA | None = None,
    max_batches: int | None = None,
    debug_state: dict[str, bool] | None = None,
    verbose_debug: bool = False,
) -> tuple[float, dict[str, float]]:
    model.train()
    running_loss = 0.0
    data_time = 0.0
    forward_time = 0.0
    backward_time = 0.0
    batch_end = time.perf_counter()
    printed_stats = False

    progress = tqdm(loader, desc="Train", leave=False)
    seen = 0
    for batch_index, (lr_imgs, hr_imgs) in enumerate(progress):
        if max_batches is not None and batch_index >= max_batches:
            break
        data_time += time.perf_counter() - batch_end
        lr_imgs = lr_imgs.to(device, non_blocking=True)
        hr_imgs = hr_imgs.to(device, non_blocking=True)
        if not printed_stats:
            verify_lr_hr_alignment(lr_imgs, hr_imgs, expected_scale=expected_scale)
        optimizer.zero_grad(set_to_none=True)

        forward_start = time.perf_counter()
        with amp_context(device, use_amp):
            loss = model.training_loss(
                lr_imgs,
                hr_imgs,
                reconstruction_criterion=criterion,
                recon_weight=config.diffusion_recon_weight,
                guide_weight=config.diffusion_guide_weight,
                x0_weight=config.diffusion_x0_weight,
                timestep_bias=config.diffusion_timestep_bias,
                clamp_range=(clamp_min, clamp_max),
            )
        forward_time += time.perf_counter() - forward_start

        backward_start = time.perf_counter()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update(model)
        backward_time += time.perf_counter() - backward_start

        running_loss += loss.item()
        seen += 1
        progress.set_postfix(loss=f"{loss.item():.4f}")
        if not printed_stats and _should_print_tensor_stats(debug_state, "train_batch_stats", verbose_debug):
            print(tensor_stats("train.lr", lr_imgs))
            print(tensor_stats("train.hr", hr_imgs))
            printed_stats = True
            _mark_tensor_stats_printed(debug_state, "train_batch_stats")
        batch_end = time.perf_counter()

    steps = max(seen, 1)
    return running_loss / steps, {
        "data_time": data_time / steps,
        "forward_time": forward_time / steps,
        "backward_time": backward_time / steps,
    }


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
    debug_state: dict[str, bool] | None = None,
    verbose_debug: bool = False,
) -> tuple[float, float, float, float, float, float]:
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_rmse = 0.0
    total_correlation = 0.0
    seen = 0
    start_time = time.perf_counter()

    with torch.no_grad():
        for batch_index, (lr_imgs, hr_imgs) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            lr_imgs = lr_imgs.to(device, non_blocking=True)
            hr_imgs = hr_imgs.to(device, non_blocking=True)
            if batch_index == 0:
                verify_lr_hr_alignment(lr_imgs, hr_imgs, expected_scale=config.scale)
            sr = apply_output_activation(model(lr_imgs), config)
            loss, _ = criterion(sr, hr_imgs)
            total_loss += loss.item()
            total_psnr += calc_psnr(sr, hr_imgs, data_range=metric_data_range, clamp_min=clamp_min, clamp_max=clamp_max)
            total_ssim += calc_ssim(sr, hr_imgs, data_range=metric_data_range, clamp_min=clamp_min, clamp_max=clamp_max)
            total_rmse += calc_rmse(sr, hr_imgs, clamp_min=clamp_min, clamp_max=clamp_max)
            total_correlation += calc_correlation(sr, hr_imgs, clamp_min=clamp_min, clamp_max=clamp_max)
            seen += 1
            if batch_index == 0 and _should_print_tensor_stats(debug_state, "val_batch_stats", verbose_debug):
                print(tensor_stats("val.lr", lr_imgs))
                print(tensor_stats("val.hr", hr_imgs))
                print(tensor_stats("val.pred", sr))
                print(f"Metric range | data_range={metric_data_range} clamp_min={clamp_min} clamp_max={clamp_max}")
                _mark_tensor_stats_printed(debug_state, "val_batch_stats")

    if seen == 0:
        return float("inf"), 0.0, 0.0, 0.0, 0.0, time.perf_counter() - start_time
    return (
        total_loss / seen,
        total_psnr / seen,
        total_ssim / seen,
        total_rmse / seen,
        total_correlation / seen,
        time.perf_counter() - start_time,
    )


def _validate_diffusion(
    model: nn.Module,
    loader: DataLoader,
    criterion: ReconstructionLoss,
    config: TrainConfig,
    clamp_min: float | None,
    clamp_max: float | None,
    metric_data_range: float,
    device: str,
    expected_scale: int,
    eval_steps: int,
    metric_batches: int,
    max_batches: int | None = None,
    debug_state: dict[str, bool] | None = None,
    verbose_debug: bool = False,
) -> tuple[float, float, float, float, float, float]:
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_rmse = 0.0
    total_correlation = 0.0
    loss_batches = 0
    metric_seen = 0
    start_time = time.perf_counter()

    stabilize_output = bool(config.diffusion_stabilize_eval and config.target_range == "zero_one")
    clamp_pred_x0 = bool(stabilize_output and config.diffusion_clamp_pred_x0)
    output_clamp = (0.0, 1.0) if stabilize_output else None

    with torch.no_grad():
        for batch_index, (lr_imgs, hr_imgs) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            lr_imgs = lr_imgs.to(device, non_blocking=True)
            hr_imgs = hr_imgs.to(device, non_blocking=True)
            if batch_index == 0:
                verify_lr_hr_alignment(lr_imgs, hr_imgs, expected_scale=expected_scale)
            loss = model.training_loss(
                lr_imgs,
                hr_imgs,
                reconstruction_criterion=criterion,
                recon_weight=config.diffusion_recon_weight,
                guide_weight=config.diffusion_guide_weight,
                x0_weight=config.diffusion_x0_weight,
                timestep_bias=config.diffusion_timestep_bias,
                clamp_range=(clamp_min, clamp_max),
            )
            total_loss += loss.item()
            loss_batches += 1

            if metric_seen < metric_batches:
                sr = model.sample(
                    lr_imgs,
                    hr_imgs.shape[-2:],
                    sample_steps=eval_steps,
                    sampler=config.diffusion_sampler,
                    clamp_output=stabilize_output,
                    output_clamp=output_clamp,
                    clamp_pred_x0=clamp_pred_x0,
                )
                total_psnr += calc_psnr(sr, hr_imgs, data_range=metric_data_range, clamp_min=clamp_min, clamp_max=clamp_max)
                total_ssim += calc_ssim(sr, hr_imgs, data_range=metric_data_range, clamp_min=clamp_min, clamp_max=clamp_max)
                total_rmse += calc_rmse(sr, hr_imgs, clamp_min=clamp_min, clamp_max=clamp_max)
                total_correlation += calc_correlation(sr, hr_imgs, clamp_min=clamp_min, clamp_max=clamp_max)
                metric_seen += 1
                if metric_seen == 1 and _should_print_tensor_stats(debug_state, "val_batch_stats", verbose_debug):
                    print(tensor_stats("val.lr", lr_imgs))
                    print(tensor_stats("val.hr", hr_imgs))
                    print(tensor_stats("val.pred", sr))
                    print(f"Metric range | data_range={metric_data_range} clamp_min={clamp_min} clamp_max={clamp_max}")
                    print(
                        "Diffusion eval | "
                        f"sampler={config.diffusion_sampler} "
                        f"eval_steps={eval_steps} "
                        f"metric_batches={metric_batches} "
                        f"stabilize_output={stabilize_output} "
                        f"clamp_pred_x0={clamp_pred_x0}"
                    )
                    _mark_tensor_stats_printed(debug_state, "val_batch_stats")

    avg_loss = total_loss / max(loss_batches, 1)
    avg_psnr = total_psnr / max(metric_seen, 1)
    avg_ssim = total_ssim / max(metric_seen, 1)
    avg_rmse = total_rmse / max(metric_seen, 1)
    avg_correlation = total_correlation / max(metric_seen, 1)
    return avg_loss, avg_psnr, avg_ssim, avg_rmse, avg_correlation, time.perf_counter() - start_time


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
    bicubic_psnr, bicubic_ssim, bicubic_rmse, bicubic_correlation = quick_bicubic_baseline(
        val_loader,
        input_mode=input_mode,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
        metric_data_range=metric_data_range,
        expected_scale=config.scale,
        max_batches=12,
    )
    print(
        "Sanity | "
        f"bicubic baseline PSNR={bicubic_psnr:.3f} "
        f"SSIM={bicubic_ssim:.4f} "
        f"RMSE={bicubic_rmse:.4f} "
        f"Corr={bicubic_correlation:.4f}"
    )
    if bicubic_psnr < 12.0:
        print("[WARN] Very low bicubic baseline detected. Check LR-HR pairing, normalization, and data range settings.")

    criterion = ReconstructionLoss(
        pixel_mode=config.loss_pixel_mode,
        pixel_weight=config.loss_pixel_weight,
        ssim_weight=config.loss_ssim_weight,
        edge_weight=config.loss_edge_weight,
        fft_weight=config.loss_fft_weight,
        ssim_data_range=metric_data_range,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
    ).to(config.device)
    model = build_model(
        config.model_name,
        in_channels=in_channels,
        out_channels=1,
        scale=config.scale,
        capacity=capacity,
        diffusion_self_condition=config.diffusion_self_condition,
    ).to(config.device)
    parameter_count = count_parameters(model)
    verify_model_device(model, config.device)
    print(f"Model params | trainable={parameter_count}")
    print(f"Patch/scale | train_hr_patch_size={config.train_hr_patch_size} scale={config.scale}")

    optimizer = build_optimizer(model.parameters(), config, family=family, lr=config.learning_rate)
    scheduler, scheduler_name = build_scheduler(optimizer, config, family=family)
    print(
        "Optimization | "
        f"optimizer={resolve_optimizer_name(config, family)} "
        f"scheduler={scheduler_name} "
        f"warmup_epochs={config.warmup_epochs} "
        f"ema_decay={config.ema_decay}"
    )

    discriminator = None
    optimizer_d = None
    if family == "gan":
        discriminator = build_discriminator(config.model_name, in_channels=1).to(config.device)
        optimizer_d = build_optimizer(
            discriminator.parameters(),
            config,
            family=family,
            lr=config.discriminator_learning_rate,
        )

    ema: ModelEMA | None = None
    if config.ema_decay > 0.0:
        ema = ModelEMA(model, decay=config.ema_decay)

    early_stopper = EarlyStopper(config.patience, min_delta=config.min_delta, mode="max")
    history: list[dict[str, float]] = []
    best_record: dict[str, float] | None = None
    best_score = float("-inf")
    debug_state = {"train_batch_stats": False, "val_batch_stats": False}

    # Create scaler once to preserve state across epochs
    scaler = build_grad_scaler(config.device, config.use_amp)

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()
        apply_warmup_lr(optimizer, config.learning_rate, epoch, config.warmup_epochs)
        if optimizer_d is not None:
            apply_warmup_lr(optimizer_d, config.discriminator_learning_rate, epoch, config.warmup_epochs)
        if family == "pixel":
            train_loss, train_timing = _train_pixel_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                config,
                config.device,
                config.use_amp,
                config.grad_clip_norm,
                scaler,
                ema=ema,
                max_batches=config.max_train_batches,
                debug_state=debug_state,
                verbose_debug=config.verbose_debug,
            )
        elif family == "gan":
            current_adv_weight = config.adv_weight if epoch > config.gan_warmup_epochs else 0.0
            train_loss, train_timing = _train_gan_epoch(
                model,
                discriminator,
                train_loader,
                criterion,
                optimizer,
                optimizer_d,
                config,
                config.device,
                current_adv_weight,
                scaler,
                ema=ema,
                max_batches=config.max_train_batches,
                debug_state=debug_state,
                verbose_debug=config.verbose_debug,
            )
        else:
            train_loss, train_timing = _train_diffusion_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                config,
                clamp_min,
                clamp_max,
                config.device,
                config.use_amp,
                config.grad_clip_norm,
                config.scale,
                scaler,
                ema=ema,
                max_batches=config.max_train_batches,
                debug_state=debug_state,
                verbose_debug=config.verbose_debug,
            )

        ema_applied = False
        if ema is not None and config.use_ema_for_eval:
            ema.apply_shadow(model)
            ema_applied = True
        try:
            if family in {"pixel", "gan"}:
                val_loss, val_psnr, val_ssim, val_rmse, val_correlation, val_time = _validate_reconstruction(
                    model,
                    val_loader,
                    criterion,
                    config,
                    clamp_min,
                    clamp_max,
                    metric_data_range,
                    config.device,
                    config.max_val_batches,
                    debug_state=debug_state,
                    verbose_debug=config.verbose_debug,
                )
            else:
                val_loss, val_psnr, val_ssim, val_rmse, val_correlation, val_time = _validate_diffusion(
                    model,
                    val_loader,
                    criterion,
                    config,
                    clamp_min,
                    clamp_max,
                    metric_data_range,
                    config.device,
                    config.scale,
                    config.diffusion_eval_steps,
                    config.diffusion_metric_batches,
                    config.max_val_batches,
                    debug_state=debug_state,
                    verbose_debug=config.verbose_debug,
                )
        finally:
            if ema_applied:
                ema.restore(model)

        epoch_time = time.perf_counter() - epoch_start
        if config.device.startswith("cuda") and torch.cuda.is_available():
            peak_mem_mb = torch.cuda.max_memory_allocated() / (1024**2)
            torch.cuda.reset_peak_memory_stats()
        else:
            peak_mem_mb = 0.0

        if scheduler is not None:
            if scheduler_name == "plateau":
                scheduler.step(val_loss)
            elif scheduler_name == "cosine" and epoch > config.warmup_epochs:
                scheduler.step()
        score = compute_selection_metric(
            model_name=config.model_name,
            psnr=val_psnr,
            ssim_value=val_ssim,
            val_loss=val_loss,
            bicubic_psnr=bicubic_psnr,
            chosen_metric=config.selection_metric,
        )
        beat_bicubic = bool(val_psnr >= bicubic_psnr)
        record = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_psnr": float(val_psnr),
            "val_ssim": float(val_ssim),
            "val_rmse": float(val_rmse),
            "val_correlation": float(val_correlation),
            "lr": current_lr(optimizer),
            "selection_score": float(score),
            "selected_metric": config.selection_metric,
            "selected_metric_value": float(score),
            "bicubic_psnr": float(bicubic_psnr),
            "beat_bicubic": beat_bicubic,
        }
        history.append(record)

        print(
            f"Epoch {epoch}/{config.epochs} | train={train_loss:.4f} | val={val_loss:.4f} | "
            f"PSNR={val_psnr:.3f} | SSIM={val_ssim:.4f} | RMSE={val_rmse:.4f} | "
            f"Corr={val_correlation:.4f} | lr={current_lr(optimizer):.2e} | "
            f"{config.selection_metric}={score:.3f}"
        )
        if family == "diffusion":
            stabilize_output = bool(config.diffusion_stabilize_eval and config.target_range == "zero_one")
            print(
                "Diffusion summary | "
                f"sampler={config.diffusion_sampler} "
                f"eval_steps={config.diffusion_eval_steps} "
                f"metric_batches={config.diffusion_metric_batches} "
                f"self_condition={config.diffusion_self_condition} "
                f"guide_w={config.diffusion_guide_weight:.2f} "
                f"recon_w={config.diffusion_recon_weight:.2f} "
                f"x0_w={config.diffusion_x0_weight:.2f} "
                f"t_bias={config.diffusion_timestep_bias:.2f} "
                f"stabilize_output={stabilize_output} "
                f"beat_bicubic={beat_bicubic}"
            )
        if config.profile_timing:
            print(
                "Timing | "
                f"data={train_timing['data_time']:.4f}s/b "
                f"fwd={train_timing['forward_time']:.4f}s/b "
                f"bwd={train_timing['backward_time']:.4f}s/b "
                f"val_total={val_time:.2f}s "
                f"epoch_total={epoch_time:.2f}s "
                f"gpu_peak_mem={peak_mem_mb:.1f}MB"
            )

        if score > best_score + config.min_delta:
            best_score = score
            best_record = record
            if ema is not None and config.use_ema_for_eval:
                ema.apply_shadow(model)
                model_state = snapshot_state_dict(model)
                ema.restore(model)
            else:
                model_state = snapshot_state_dict(model)
            payload = {
                "model_name": config.model_name,
                "family": family,
                "capacity": capacity,
                "input_mode": input_mode,
                "model_state_dict": model_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
                "record": record,
                "config": asdict(effective_config),
            }
            if discriminator is not None and optimizer_d is not None:
                payload["discriminator_state_dict"] = snapshot_state_dict(discriminator)
                payload["optimizer_d_state_dict"] = optimizer_d.state_dict()
            if ema is not None:
                payload["ema_state_dict"] = ema.state_dict()
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
        best_rmse=float(best_record["val_rmse"]),
        best_correlation=float(best_record["val_correlation"]),
        bicubic_psnr=float(bicubic_psnr),
        selection_score=float(best_record["selection_score"]),
        selected_metric=config.selection_metric,
        selected_metric_value=float(best_record.get("selected_metric_value", best_record["selection_score"])),
        beat_bicubic=bool(best_record.get("beat_bicubic", float(best_record["val_psnr"]) >= bicubic_psnr)),
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

    summaries.sort(key=summary_sort_key, reverse=True)

    leaderboard = [asdict(summary) for summary in summaries]
    _save_json(benchmark_dir / "leaderboard.json", {"leaderboard": leaderboard})
    with (benchmark_dir / "leaderboard.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=leaderboard[0].keys())
        writer.writeheader()
        writer.writerows(leaderboard)

    return summaries


def tuning_trial_overrides(model_name: str) -> list[dict[str, object]]:
    presets: dict[str, list[dict[str, object]]] = {
        "srcnn": [
            {
                "capacity": "base",
                "learning_rate": 2e-4,
                "batch_size": 8,
                "train_hr_patch_size": 192,
                "loss_ssim_weight": 0.12,
                "loss_edge_weight": 0.08,
                "loss_fft_weight": 0.02,
                "ema_decay": 0.995,
            },
            {
                "capacity": "large",
                "learning_rate": 1e-4,
                "batch_size": 6,
                "train_hr_patch_size": 224,
                "loss_ssim_weight": 0.15,
                "loss_edge_weight": 0.10,
                "loss_fft_weight": 0.03,
                "ema_decay": 0.999,
            },
        ],
        "rlfb_esa": [
            {
                "capacity": "base",
                "learning_rate": 2e-4,
                "batch_size": 6,
                "train_hr_patch_size": 192,
                "loss_ssim_weight": 0.10,
                "loss_edge_weight": 0.08,
                "loss_fft_weight": 0.02,
            },
            {
                "capacity": "large",
                "learning_rate": 1.2e-4,
                "batch_size": 4,
                "train_hr_patch_size": 224,
                "loss_ssim_weight": 0.14,
                "loss_edge_weight": 0.10,
                "loss_fft_weight": 0.03,
            },
        ],
        "edsr": [
            {
                "capacity": "base",
                "learning_rate": 1.5e-4,
                "batch_size": 4,
                "train_hr_patch_size": 192,
                "loss_ssim_weight": 0.12,
                "loss_edge_weight": 0.10,
                "loss_fft_weight": 0.03,
                "ema_decay": 0.999,
            },
            {
                "capacity": "large",
                "learning_rate": 1e-4,
                "batch_size": 3,
                "train_hr_patch_size": 160,
                "loss_ssim_weight": 0.16,
                "loss_edge_weight": 0.12,
                "loss_fft_weight": 0.04,
                "ema_decay": 0.999,
            },
        ],
        "rcan": [
            {
                "capacity": "base",
                "learning_rate": 1e-4,
                "batch_size": 3,
                "train_hr_patch_size": 160,
                "loss_ssim_weight": 0.16,
                "loss_edge_weight": 0.10,
                "loss_fft_weight": 0.04,
                "ema_decay": 0.999,
            },
            {
                "capacity": "large",
                "learning_rate": 8e-5,
                "batch_size": 2,
                "train_hr_patch_size": 128,
                "loss_ssim_weight": 0.18,
                "loss_edge_weight": 0.12,
                "loss_fft_weight": 0.05,
                "ema_decay": 0.999,
            },
        ],
        "srgan": [
            {
                "capacity": "tiny",
                "learning_rate": 1e-4,
                "discriminator_learning_rate": 1e-4,
                "batch_size": 2,
                "train_hr_patch_size": 128,
                "adv_weight": 3e-3,
                "scheduler": "plateau",
                "loss_ssim_weight": 0.05,
                "loss_edge_weight": 0.03,
                "loss_fft_weight": 0.0,
                "ema_decay": 0.995,
            },
            {
                "capacity": "base",
                "learning_rate": 8e-5,
                "discriminator_learning_rate": 8e-5,
                "batch_size": 2,
                "train_hr_patch_size": 128,
                "adv_weight": 1e-3,
                "scheduler": "plateau",
                "loss_ssim_weight": 0.08,
                "loss_edge_weight": 0.04,
                "loss_fft_weight": 0.0,
                "ema_decay": 0.995,
            },
        ],
        "esrgan": [
            {
                "capacity": "tiny",
                "learning_rate": 1e-4,
                "discriminator_learning_rate": 1e-4,
                "batch_size": 2,
                "train_hr_patch_size": 128,
                "adv_weight": 3e-3,
                "scheduler": "plateau",
                "loss_ssim_weight": 0.06,
                "loss_edge_weight": 0.03,
                "loss_fft_weight": 0.0,
                "ema_decay": 0.995,
            },
            {
                "capacity": "base",
                "learning_rate": 8e-5,
                "discriminator_learning_rate": 8e-5,
                "batch_size": 1,
                "train_hr_patch_size": 128,
                "adv_weight": 1e-3,
                "scheduler": "plateau",
                "loss_ssim_weight": 0.08,
                "loss_edge_weight": 0.04,
                "loss_fft_weight": 0.0,
                "ema_decay": 0.995,
            },
        ],
        "diffusion_sr": [
            {
                "capacity": "tiny",
                "learning_rate": 8e-5,
                "batch_size": 2,
                "train_hr_patch_size": 160,
                "scheduler": "cosine",
                "diffusion_eval_steps": 24,
                "diffusion_metric_batches": 6,
                "ema_decay": 0.9995,
                "patience": 16,
                "diffusion_recon_weight": 0.15,
                "diffusion_guide_weight": 0.08,
                "diffusion_x0_weight": 0.20,
                "diffusion_timestep_bias": 1.4,
            },
            {
                "capacity": "base",
                "learning_rate": 6e-5,
                "batch_size": 1,
                "train_hr_patch_size": 128,
                "scheduler": "cosine",
                "diffusion_eval_steps": 32,
                "diffusion_metric_batches": 8,
                "ema_decay": 0.9995,
                "patience": 18,
                "diffusion_recon_weight": 0.22,
                "diffusion_guide_weight": 0.10,
                "diffusion_x0_weight": 0.30,
                "diffusion_timestep_bias": 1.7,
            },
        ],
        "swinir": [
            {
                "capacity": "tiny",
                "learning_rate": 1e-4,
                "batch_size": 4,
                "train_hr_patch_size": 128,
                "loss_ssim_weight": 0.14,
                "loss_edge_weight": 0.10,
                "loss_fft_weight": 0.03,
                "ema_decay": 0.999,
            },
            {
                "capacity": "base",
                "learning_rate": 6e-5,
                "batch_size": 2,
                "train_hr_patch_size": 128,
                "loss_ssim_weight": 0.18,
                "loss_edge_weight": 0.12,
                "loss_fft_weight": 0.04,
                "ema_decay": 0.999,
            },
        ],
    }
    if model_name not in presets:
        raise KeyError(f"No tuning presets for model: {model_name}")
    return presets[model_name]


def should_final_fit(summary: TrainingSummary, bicubic_psnr: float, args: dict[str, object]) -> tuple[bool, str]:
    reasons: list[str] = []
    model_name = summary.model_name

    if bool(args.get("quick_only", False)):
        reasons.append("quick-only mode enabled")

    final_models_set = args.get("final_models_set")
    if final_models_set is not None and model_name not in final_models_set:
        reasons.append("not included in --final-models")

    topk_final = int(args.get("topk_final", 0) or 0)
    topk_set = args.get("topk_set")
    if topk_final > 0 and (topk_set is None or model_name not in topk_set):
        reasons.append(f"outside top-{topk_final} by selected metric")

    skip_margin = args.get("skip_final_below_bicubic_margin")
    if skip_margin is not None:
        margin = float(skip_margin)
        threshold = bicubic_psnr - margin
        if summary.best_psnr < threshold:
            reasons.append(
                f"quick PSNR {summary.best_psnr:.3f} < bicubic {bicubic_psnr:.3f} - margin {margin:.3f}"
            )

    if bool(args.get("diffusion_final_only_if_competitive", False)) and MODEL_SPECS[model_name].family == "diffusion":
        small_margin = float(args.get("diffusion_competitive_margin", 0.1))
        competitive_vs_bicubic = summary.best_psnr >= (bicubic_psnr - small_margin)
        ranked_topk = bool(topk_set is not None and model_name in topk_set)
        if not competitive_vs_bicubic and not ranked_topk:
            reasons.append(
                "diffusion not competitive "
                f"(PSNR {summary.best_psnr:.3f} < bicubic {bicubic_psnr:.3f} - {small_margin:.3f}, "
                "and not in top-k)"
            )

    if reasons:
        return False, "; ".join(reasons)
    return True, "eligible"


def finetune_all_models(
    base_config: TrainConfig,
    model_names: list[str] | None = None,
    dataset_root: str | Path | None = None,
    quick_epochs: int = 8,
    quick_max_train_batches: int | None = 800,
    quick_max_val_batches: int | None = 80,
    final_epochs: int | None = None,
    *,
    quick_only: bool = False,
    topk_final: int = 0,
    final_models: list[str] | None = None,
    skip_final_below_bicubic_margin: float | None = None,
    selection_metric: str | None = None,
    diffusion_final_only_if_competitive: bool = False,
    diffusion_competitive_margin: float = 0.1,
) -> list[TrainingSummary]:
    models = model_names or candidate_order(diffusion_centric=True)
    metric_name = selection_metric or base_config.selection_metric or "score"
    if metric_name not in {"score", "psnr", "ssim", "bicubic_gap_psnr"}:
        raise ValueError(f"Unsupported selection metric: {metric_name}")
    if topk_final < 0:
        raise ValueError("--topk-final must be >= 0.")

    base_config = replace(base_config, selection_metric=metric_name)
    tuning_root = Path(base_config.save_root) if base_config.save_root else resolve_project_root() / "checkpoints" / "sr_finetune_suite"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_dir = tuning_root / timestamp
    suite_dir.mkdir(parents=True, exist_ok=True)

    quick_records: list[dict[str, object]] = []
    best_trial_config: dict[str, TrainConfig] = {}
    best_quick_summaries: list[TrainingSummary] = []

    for model_name in models:
        trial_overrides = tuning_trial_overrides(model_name)
        model_trial_summaries: list[tuple[TrainingSummary, TrainConfig]] = []
        print(f"[TUNE] Quick search | model={model_name} | trials={len(trial_overrides)}")
        for trial_index, overrides in enumerate(trial_overrides, start=1):
            trial_name = f"{model_name}_trial{trial_index}"
            trial_save_root = suite_dir / "quick" / model_name / trial_name
            trial_kwargs = dict(overrides)
            if base_config.capacity != "auto":
                trial_kwargs["capacity"] = base_config.capacity
            if base_config.input_mode != "auto":
                trial_kwargs["input_mode"] = base_config.input_mode
            if base_config.optimizer != "auto":
                trial_kwargs["optimizer"] = base_config.optimizer
            if base_config.scheduler != "auto":
                trial_kwargs["scheduler"] = base_config.scheduler
            if base_config.loss_pixel_mode != "charbonnier":
                trial_kwargs["loss_pixel_mode"] = base_config.loss_pixel_mode
            run_config = replace(
                base_config,
                model_name=model_name,
                epochs=quick_epochs,
                save_root=str(trial_save_root),
                max_train_batches=quick_max_train_batches,
                max_val_batches=quick_max_val_batches,
                selection_metric=metric_name,
                **trial_kwargs,
            )
            summary = fit_model(run_config, dataset_root=dataset_root)
            model_trial_summaries.append((summary, run_config))
            quick_records.append(
                {
                    "model_name": model_name,
                    "trial_index": trial_index,
                    "trial_config": asdict(run_config),
                    "summary": asdict(summary),
                }
            )

        model_trial_summaries.sort(key=lambda item: summary_sort_key(item[0]), reverse=True)
        best_summary, best_config = model_trial_summaries[0]
        best_trial_config[model_name] = best_config
        best_quick_summaries.append(best_summary)
        print(
            f"[TUNE] Best quick trial | model={model_name} | "
            f"{metric_name}={best_summary.selected_metric_value:.3f} | "
            f"PSNR={best_summary.best_psnr:.3f} | SSIM={best_summary.best_ssim:.4f} | "
            f"Bicubic={best_summary.bicubic_psnr:.3f}"
        )

    quick_leaderboard = sorted(best_quick_summaries, key=summary_sort_key, reverse=True)
    quick_rank_by_model = {summary.model_name: index + 1 for index, summary in enumerate(quick_leaderboard)}
    final_models_set = set(final_models) if final_models else None
    topk_set = set(summary.model_name for summary in quick_leaderboard[:topk_final]) if topk_final > 0 else None

    print("[TUNE] Quick leaderboard")
    for index, summary in enumerate(quick_leaderboard, start=1):
        gap = summary.best_psnr - summary.bicubic_psnr
        print(
            f"  {index}. {summary.model_name} | {metric_name}={summary.selected_metric_value:.3f} | "
            f"PSNR={summary.best_psnr:.3f} | SSIM={summary.best_ssim:.4f} | "
            f"bicubic={summary.bicubic_psnr:.3f} | gap={gap:+.3f}"
        )
    if quick_leaderboard:
        bicubic_values = [summary.bicubic_psnr for summary in quick_leaderboard]
        print(
            "[TUNE] Bicubic baseline | "
            f"mean={float(np.mean(bicubic_values)):.3f} "
            f"min={float(np.min(bicubic_values)):.3f} "
            f"max={float(np.max(bicubic_values)):.3f}"
        )

    gating_args = {
        "quick_only": quick_only,
        "topk_final": topk_final,
        "topk_set": topk_set,
        "final_models_set": final_models_set,
        "skip_final_below_bicubic_margin": skip_final_below_bicubic_margin,
        "diffusion_final_only_if_competitive": diffusion_final_only_if_competitive,
        "diffusion_competitive_margin": diffusion_competitive_margin,
    }
    eligibility: list[dict[str, object]] = []
    eligible_summaries: list[TrainingSummary] = []
    skipped_summaries: list[tuple[TrainingSummary, str]] = []
    for summary in quick_leaderboard:
        allow, reason = should_final_fit(summary, summary.bicubic_psnr, gating_args)
        eligibility.append(
            {
                "model_name": summary.model_name,
                "allow_final_fit": allow,
                "reason": reason,
                "quick_rank": quick_rank_by_model[summary.model_name],
            }
        )
        if allow:
            eligible_summaries.append(summary)
        else:
            skipped_summaries.append((summary, reason))

    print("[TUNE] Final-fit eligibility")
    if eligible_summaries:
        for summary in eligible_summaries:
            print(f"  eligible: {summary.model_name} (rank={quick_rank_by_model[summary.model_name]})")
    else:
        print("  eligible: none")
    if skipped_summaries:
        for summary, reason in skipped_summaries:
            print(f"  skipped: {summary.model_name} | {reason}")

    has_routing_filters = (
        topk_final > 0
        or final_models_set is not None
        or skip_final_below_bicubic_margin is not None
        or diffusion_final_only_if_competitive
    )
    if not has_routing_filters:
        family_priority = {"pixel": 0, "gan": 1, "diffusion": 2}
        final_plan = sorted(
            eligible_summaries,
            key=lambda summary: (
                family_priority.get(MODEL_SPECS[summary.model_name].family, 3),
                -summary.selected_metric_value,
                -summary.best_psnr,
                summary.model_name,
            ),
        )
    else:
        eligible_names = {item.model_name for item in eligible_summaries}
        final_plan = [summary for summary in quick_leaderboard if summary.model_name in eligible_names]

    print("[TUNE] Final-fit plan")
    if final_plan:
        for index, summary in enumerate(final_plan, start=1):
            print(f"  {index}. {summary.model_name} (rank={quick_rank_by_model[summary.model_name]})")
    else:
        print("  no final fits scheduled")

    _save_json(
        suite_dir / "quick_search_results.json",
        {
            "selection_metric": metric_name,
            "trials": quick_records,
            "quick_leaderboard": [asdict(summary) for summary in quick_leaderboard],
            "eligibility": eligibility,
            "final_plan": [summary.model_name for summary in final_plan],
        },
    )

    final_summaries: list[TrainingSummary] = []
    target_epochs = final_epochs if final_epochs is not None else base_config.epochs
    for summary in final_plan:
        model_name = summary.model_name
        cfg = best_trial_config[model_name]
        final_save_root = suite_dir / "final" / model_name
        final_config = replace(
            cfg,
            model_name=model_name,
            epochs=target_epochs,
            save_root=str(final_save_root),
            max_train_batches=base_config.max_train_batches,
            max_val_batches=base_config.max_val_batches,
            selection_metric=metric_name,
        )
        print(f"[TUNE] Final fit | model={model_name} | epochs={target_epochs}")
        final_summaries.append(fit_model(final_config, dataset_root=dataset_root))

    final_summaries.sort(key=summary_sort_key, reverse=True)
    output_summaries = final_summaries if final_summaries else quick_leaderboard
    leaderboard = [asdict(summary) for summary in output_summaries]
    _save_json(suite_dir / "leaderboard.json", {"leaderboard": leaderboard})
    if leaderboard:
        with (suite_dir / "leaderboard.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=leaderboard[0].keys())
            writer.writeheader()
            writer.writerows(leaderboard)
    _save_json(
        suite_dir / "best_fit.json",
        {
            "best_model": leaderboard[0] if leaderboard else None,
            "suite_dir": str(suite_dir),
            "models": models,
            "quick_only": quick_only,
            "selection_metric": metric_name,
        },
    )
    return output_summaries
