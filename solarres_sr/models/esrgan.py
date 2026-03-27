from __future__ import annotations

import torch
import torch.nn as nn

from .common import default_conv


class DenseResidualBlock(nn.Module):
    def __init__(self, channels: int, growth_channels: int = 32, res_scale: float = 0.2) -> None:
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = default_conv(channels, growth_channels, 3)
        self.conv2 = default_conv(channels + growth_channels, growth_channels, 3)
        self.conv3 = default_conv(channels + 2 * growth_channels, growth_channels, 3)
        self.conv4 = default_conv(channels + 3 * growth_channels, growth_channels, 3)
        self.conv5 = default_conv(channels + 4 * growth_channels, channels, 3)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = self.act(self.conv3(torch.cat([x, x1, x2], dim=1)))
        x4 = self.act(self.conv4(torch.cat([x, x1, x2, x3], dim=1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], dim=1))
        return x + x5 * self.res_scale


class RRDB(nn.Module):
    def __init__(self, channels: int, growth_channels: int = 32, res_scale: float = 0.2) -> None:
        super().__init__()
        # Each DenseResidualBlock already applies res_scale internally.
        # We use a smaller scale (res_scale^0.5) per block to achieve similar overall scaling
        # without double-scaling the features.
        block_scale = res_scale ** 0.5
        self.body = nn.Sequential(
            DenseResidualBlock(channels, growth_channels, block_scale),
            DenseResidualBlock(channels, growth_channels, block_scale),
            DenseResidualBlock(channels, growth_channels, block_scale),
        )
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Don't apply res_scale again - blocks already scale their residuals
        return x + self.body(x)


class RRDBNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        num_features: int = 64,
        num_blocks: int = 12,
        growth_channels: int = 32,
        scale: int = 4,
    ) -> None:
        super().__init__()
        self.head = default_conv(in_channels, num_features, 3)
        self.body = nn.Sequential(*[RRDB(num_features, growth_channels=growth_channels) for _ in range(num_blocks)])
        self.body_conv = default_conv(num_features, num_features, 3)

        upsample_layers: list[nn.Module] = []
        current_scale = scale
        while current_scale > 1:
            upsample_layers.extend(
                [
                    nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1),
                    nn.PixelShuffle(2),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            current_scale //= 2
        self.upsampler = nn.Sequential(*upsample_layers)
        self.tail = nn.Sequential(
            default_conv(num_features, num_features, 3),
            nn.LeakyReLU(0.2, inplace=True),
            default_conv(num_features, out_channels, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        head = self.head(x)
        body = self.body_conv(self.body(head)) + head
        out = self.upsampler(body)
        return self.tail(out)
