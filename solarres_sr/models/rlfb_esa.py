from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import default_conv


class RLFB(nn.Module):
    def __init__(self, channels: int = 64) -> None:
        super().__init__()
        self.conv1 = default_conv(channels, channels, 3)
        self.conv2 = default_conv(channels, channels, 3)
        self.conv3 = default_conv(channels, channels, 3)
        self.conv_reduce = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.conv1(x))
        out = self.act(self.conv2(out))
        out = self.act(self.conv3(out))
        out = self.conv_reduce(out)
        return x + out


class ESA(nn.Module):
    def __init__(self, channels: int = 64) -> None:
        super().__init__()
        reduced = max(channels // 4, 16)
        self.reduce = nn.Conv2d(channels, reduced, kernel_size=1)
        self.conv1 = nn.Conv2d(reduced, reduced, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(reduced, reduced, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(reduced, reduced, kernel_size=3, padding=1)
        self.expand = nn.Conv2d(reduced, channels, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=7, stride=3, padding=3)
        self.sigmoid = nn.Sigmoid()
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reduced = self.reduce(x)
        branch = self.act(self.conv1(reduced))
        branch = self.pool(branch)
        branch = self.act(self.conv2(branch))
        branch = self.conv3(branch)
        branch = F.interpolate(branch, size=x.shape[-2:], mode="bilinear", align_corners=False)
        branch = self.expand(branch)
        return x * self.sigmoid(branch)


class RLFBESANet(nn.Module):
    def __init__(
        self,
        in_channels: int = 10,
        out_channels: int = 1,
        num_features: int = 64,
        num_rlfb: int = 12,
        scale: int = 4,
    ) -> None:
        super().__init__()
        self.shallow = default_conv(in_channels, num_features, 3)
        self.rlfb_blocks = nn.Sequential(*[RLFB(num_features) for _ in range(num_rlfb)])
        self.esa = ESA(num_features)
        self.fuse = nn.Conv2d(num_features, num_features, kernel_size=1)

        upsample_layers: list[nn.Module] = []
        current_scale = scale
        while current_scale > 1:
            upsample_layers.extend(
                [
                    nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1),
                    nn.PixelShuffle(2),
                    nn.ReLU(inplace=True),
                ]
            )
            current_scale //= 2
        self.upsampler = nn.Sequential(*upsample_layers)
        self.reconstruction = default_conv(num_features, out_channels, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shallow = self.shallow(x)
        feat = self.rlfb_blocks(shallow)
        feat = self.esa(feat)
        feat = self.fuse(feat + shallow)
        feat = self.upsampler(feat)
        return self.reconstruction(feat)
