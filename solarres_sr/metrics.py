from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from pytorch_msssim import ssim


def calc_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    clamp_min: float | None = 0.0,
    clamp_max: float | None = 1.0,
) -> float:
    """Calculate PSNR averaged across batch samples (standard approach).

    Computes per-sample PSNR and averages, rather than computing PSNR
    on global MSE, which matches the standard reporting in SR papers.
    """
    if clamp_min is not None or clamp_max is not None:
        low = -float("inf") if clamp_min is None else clamp_min
        high = float("inf") if clamp_max is None else clamp_max
        pred = pred.clamp(low, high)
        target = target.clamp(low, high)

    # Compute per-sample MSE (mean over spatial dims, not batch)
    batch_size = pred.shape[0]
    if batch_size == 0:
        return 0.0

    # Reshape to (B, -1) and compute MSE per sample
    pred_flat = pred.reshape(batch_size, -1)
    target_flat = target.reshape(batch_size, -1)
    mse_per_sample = ((pred_flat - target_flat) ** 2).mean(dim=1)

    # Compute PSNR per sample, then average
    # Avoid log(0) by clamping MSE
    mse_per_sample = mse_per_sample.clamp_min(1e-12)
    psnr_per_sample = 20.0 * math.log10(data_range) - 10.0 * torch.log10(mse_per_sample)

    return float(psnr_per_sample.mean().item())


def calc_rmse(
    pred: torch.Tensor,
    target: torch.Tensor,
    clamp_min: float | None = 0.0,
    clamp_max: float | None = 1.0,
) -> float:
    if clamp_min is not None or clamp_max is not None:
        low = -float("inf") if clamp_min is None else clamp_min
        high = float("inf") if clamp_max is None else clamp_max
        pred = pred.clamp(low, high)
        target = target.clamp(low, high)
    mse = F.mse_loss(pred, target)
    return float(torch.sqrt(mse).item())


def calc_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    clamp_min: float | None = 0.0,
    clamp_max: float | None = 1.0,
) -> float:
    if clamp_min is not None or clamp_max is not None:
        low = -float("inf") if clamp_min is None else clamp_min
        high = float("inf") if clamp_max is None else clamp_max
        pred = pred.clamp(low, high)
        target = target.clamp(low, high)
    return float(ssim(pred, target, data_range=data_range, size_average=True).item())


def calc_correlation(
    pred: torch.Tensor,
    target: torch.Tensor,
    clamp_min: float | None = 0.0,
    clamp_max: float | None = 1.0,
) -> float:
    if clamp_min is not None or clamp_max is not None:
        low = -float("inf") if clamp_min is None else clamp_min
        high = float("inf") if clamp_max is None else clamp_max
        pred = pred.clamp(low, high)
        target = target.clamp(low, high)

    pred_flat = pred.reshape(pred.shape[0], -1).float()
    target_flat = target.reshape(target.shape[0], -1).float()
    pred_centered = pred_flat - pred_flat.mean(dim=1, keepdim=True)
    target_centered = target_flat - target_flat.mean(dim=1, keepdim=True)
    numerator = (pred_centered * target_centered).sum(dim=1)
    denominator = torch.sqrt(
        pred_centered.square().sum(dim=1) * target_centered.square().sum(dim=1)
    ).clamp_min(1e-12)
    return float((numerator / denominator).mean().item())


def selection_score(psnr: float, ssim_value: float, val_loss: float) -> float:
    if math.isinf(psnr):
        psnr = 99.0
    return float(psnr + 20.0 * ssim_value - val_loss)
