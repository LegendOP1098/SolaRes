from __future__ import annotations

import torch
import torch.nn as nn

from .common import default_conv


class SRResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            default_conv(channels, channels, 3),
            nn.PReLU(channels),
            default_conv(channels, channels, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class UpsampleBlock(nn.Sequential):
    def __init__(self, channels: int) -> None:
        super().__init__(
            nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU(channels),
        )


class SRGANGenerator(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        num_features: int = 64,
        num_blocks: int = 16,
        scale: int = 4,
    ) -> None:
        super().__init__()
        self.head = nn.Sequential(default_conv(in_channels, num_features, 9), nn.PReLU(num_features))
        self.body = nn.Sequential(*[SRResBlock(num_features) for _ in range(num_blocks)])
        self.body_conv = default_conv(num_features, num_features, 3)
        upsample_layers = []
        current_scale = scale
        while current_scale > 1:
            upsample_layers.append(UpsampleBlock(num_features))
            current_scale //= 2
        self.upsampler = nn.Sequential(*upsample_layers)
        self.tail = nn.Sequential(default_conv(num_features, out_channels, 9))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        head = self.head(x)
        body = self.body_conv(self.body(head)) + head
        out = self.upsampler(body)
        return self.tail(out)


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 64) -> None:
        super().__init__()
        channels = [base_channels, base_channels, base_channels * 2, base_channels * 2, base_channels * 4, base_channels * 4]
        strides = [1, 2, 1, 2, 1, 2]
        layers: list[nn.Module] = []
        previous = in_channels
        for index, (channel, stride) in enumerate(zip(channels, strides)):
            layers.append(nn.Conv2d(previous, channel, kernel_size=3, stride=stride, padding=1))
            if index > 0:
                layers.append(nn.BatchNorm2d(channel))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            previous = channel
        layers.extend(
            [
                nn.Conv2d(previous, base_channels * 8, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(base_channels * 8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(base_channels * 8, 1, kernel_size=3, stride=1, padding=1),
            ]
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
