from __future__ import annotations

import torch
import torch.nn as nn

from .common import ChannelAttention, PixelShuffleUpsampler, default_conv


class RCAB(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, res_scale: float = 1.0) -> None:
        super().__init__()
        self.body = nn.Sequential(
            default_conv(channels, channels, 3),
            nn.ReLU(inplace=True),
            default_conv(channels, channels, 3),
            ChannelAttention(channels, reduction=reduction),
        )
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x) * self.res_scale


class ResidualGroup(nn.Module):
    def __init__(self, channels: int, num_blocks: int, reduction: int = 16, res_scale: float = 1.0) -> None:
        super().__init__()
        modules = [RCAB(channels, reduction=reduction, res_scale=res_scale) for _ in range(num_blocks)]
        modules.append(default_conv(channels, channels, 3))
        self.body = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class RCAN(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        num_features: int = 64,
        num_groups: int = 6,
        num_blocks: int = 10,
        reduction: int = 16,
        scale: int = 4,
    ) -> None:
        super().__init__()
        self.head = default_conv(in_channels, num_features, 3)
        self.body = nn.Sequential(
            *[
                ResidualGroup(
                    channels=num_features,
                    num_blocks=num_blocks,
                    reduction=reduction,
                    res_scale=1.0,
                )
                for _ in range(num_groups)
            ]
        )
        self.body_conv = default_conv(num_features, num_features, 3)
        self.upsampler = PixelShuffleUpsampler(num_features, scale, activation=nn.ReLU)
        self.tail = default_conv(num_features, out_channels, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        head = self.head(x)
        body = self.body_conv(self.body(head)) + head
        out = self.upsampler(body)
        return self.tail(out)
