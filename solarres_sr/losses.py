from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps**2))


class SobelEdgeLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        kernel_x = torch.tensor(
            [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]], dtype=torch.float32
        )
        kernel_y = torch.tensor(
            [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]], dtype=torch.float32
        )
        self.register_buffer("kernel_x", kernel_x.unsqueeze(1))
        self.register_buffer("kernel_y", kernel_y.unsqueeze(1))

    def _gradients(self, tensor: torch.Tensor) -> torch.Tensor:
        channels = tensor.shape[1]
        kernel_x = self.kernel_x.repeat(channels, 1, 1, 1)
        kernel_y = self.kernel_y.repeat(channels, 1, 1, 1)
        grad_x = F.conv2d(tensor, kernel_x, padding=1, groups=channels)
        grad_y = F.conv2d(tensor, kernel_y, padding=1, groups=channels)
        return torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self._gradients(pred), self._gradients(target))


class FFTMagnitudeLoss(nn.Module):
    def __init__(self, log_magnitude: bool = True) -> None:
        super().__init__()
        self.log_magnitude = log_magnitude

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # FFT in float32 avoids ComplexHalf operator gaps under AMP.
        pred_fft = torch.fft.rfftn(pred.float(), dim=(-2, -1))
        target_fft = torch.fft.rfftn(target.float(), dim=(-2, -1))
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        if self.log_magnitude:
            pred_mag = torch.log1p(pred_mag)
            target_mag = torch.log1p(target_mag)
        return F.l1_loss(pred_mag, target_mag)


class ReconstructionLoss(nn.Module):
    def __init__(
        self,
        pixel_mode: str = "charbonnier",
        pixel_weight: float = 1.0,
        ssim_weight: float = 0.15,
        edge_weight: float = 0.1,
        fft_weight: float = 0.0,
        ssim_data_range: float = 1.0,
        clamp_min: float | None = 0.0,
        clamp_max: float | None = 1.0,
    ) -> None:
        super().__init__()
        self.pixel_mode = pixel_mode
        self.pixel_weight = pixel_weight
        self.ssim_weight = ssim_weight
        self.edge_weight = edge_weight
        self.fft_weight = fft_weight
        self.ssim_data_range = ssim_data_range
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        if pixel_mode == "charbonnier":
            self.pixel = CharbonnierLoss()
        elif pixel_mode == "l1":
            self.pixel = nn.L1Loss()
        elif pixel_mode == "mse":
            self.pixel = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported pixel_mode: {pixel_mode}")
        self.edge = SobelEdgeLoss()
        self.fft = FFTMagnitudeLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        pixel_loss = self.pixel(pred, target)
        if self.clamp_min is not None or self.clamp_max is not None:
            low = -float("inf") if self.clamp_min is None else self.clamp_min
            high = float("inf") if self.clamp_max is None else self.clamp_max
            pred_ssim = pred.clamp(low, high)
            target_ssim = target.clamp(low, high)
        else:
            pred_ssim = pred
            target_ssim = target
        ssim_loss = 1.0 - ssim(pred_ssim, target_ssim, data_range=self.ssim_data_range, size_average=True)
        edge_loss = self.edge(pred, target)
        fft_loss = self.fft(pred, target)
        total = (
            self.pixel_weight * pixel_loss
            + self.ssim_weight * ssim_loss
            + self.edge_weight * edge_loss
            + self.fft_weight * fft_loss
        )
        return total, {
            "pixel": float(pixel_loss.item()),
            "ssim": float(ssim_loss.item()),
            "edge": float(edge_loss.item()),
            "fft": float(fft_loss.item()),
            "total": float(total.item()),
        }
