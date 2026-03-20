from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SRCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        scale: int = 4,
        features: tuple[int, int] = (64, 32),
    ) -> None:
        super().__init__()
        self.scale = scale
        self.conv1 = nn.Conv2d(in_channels, features[0], kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(features[0], features[1], kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(features[1], out_channels, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return self.conv3(x)
