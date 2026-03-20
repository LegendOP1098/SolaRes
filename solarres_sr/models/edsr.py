from __future__ import annotations

import torch
import torch.nn as nn

from .common import PixelShuffleUpsampler, ResidualBlock, default_conv


class EDSR(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        num_features: int = 64,
        num_blocks: int = 16,
        res_scale: float = 0.1,
        scale: int = 4,
    ) -> None:
        super().__init__()
        self.head = default_conv(in_channels, num_features, 3)
        self.body = nn.Sequential(*[ResidualBlock(num_features, res_scale=res_scale) for _ in range(num_blocks)])
        self.body_conv = default_conv(num_features, num_features, 3)
        self.upsampler = PixelShuffleUpsampler(num_features, scale, activation=nn.ReLU)
        self.tail = default_conv(num_features, out_channels, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        head = self.head(x)
        body = self.body_conv(self.body(head)) + head
        out = self.upsampler(body)
        return self.tail(out)
