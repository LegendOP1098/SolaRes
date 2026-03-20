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
    if clamp_min is not None or clamp_max is not None:
        low = -float("inf") if clamp_min is None else clamp_min
        high = float("inf") if clamp_max is None else clamp_max
        pred = pred.clamp(low, high)
        target = target.clamp(low, high)
    mse = F.mse_loss(pred, target)
    if mse.item() <= 1e-12:
        return float("inf")
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse.item())


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


def selection_score(psnr: float, ssim_value: float, val_loss: float) -> float:
    if math.isinf(psnr):
        psnr = 99.0
    return float(psnr + 20.0 * ssim_value - val_loss)
