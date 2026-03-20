from __future__ import annotations

import math

import torch
import torch.nn as nn


def default_conv(in_channels: int, out_channels: int, kernel_size: int = 3, bias: bool = True) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=bias)


def norm_groups(channels: int, preferred: int = 8) -> int:
    for groups in reversed(range(1, preferred + 1)):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, res_scale: float = 1.0) -> None:
        super().__init__()
        self.body = nn.Sequential(
            default_conv(channels, channels, 3),
            nn.ReLU(inplace=True),
            default_conv(channels, channels, 3),
        )
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x) * self.res_scale


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.attn(x)


class PixelShuffleUpsampler(nn.Sequential):
    def __init__(self, channels: int, scale: int, activation: type[nn.Module] | None = None) -> None:
        if scale not in {2, 3, 4, 8}:
            raise ValueError(f"Unsupported scale factor: {scale}")
        modules: list[nn.Module] = []
        if scale == 3:
            modules.extend([default_conv(channels, channels * 9, 3), nn.PixelShuffle(3)])
            if activation is not None:
                modules.append(activation(inplace=True) if activation is nn.ReLU else activation())
        else:
            steps = int(math.log2(scale))
            for _ in range(steps):
                modules.extend([default_conv(channels, channels * 4, 3), nn.PixelShuffle(2)])
                if activation is not None:
                    modules.append(activation(inplace=True) if activation is nn.ReLU else activation())
        super().__init__(*modules)
