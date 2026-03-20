from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import norm_groups


def sinusoidal_time_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half_dim = dim // 2
    exponent = -math.log(10000.0) * torch.arange(half_dim, device=timesteps.device) / max(half_dim - 1, 1)
    emb = timesteps.float().unsqueeze(1) * torch.exp(exponent).unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-4, 0.999)


def extract(buffer: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    values = buffer.gather(0, timesteps)
    return values.reshape(shape[0], *((1,) * (len(shape) - 1)))


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(norm_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(norm_groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.time_proj(time_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


class ConditionalUNet(nn.Module):
    def __init__(self, in_channels: int, condition_channels: int, base_channels: int = 32) -> None:
        super().__init__()
        time_dim = base_channels * 4
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(inplace=True),
            nn.Linear(time_dim, time_dim),
        )
        self.in_conv = nn.Conv2d(in_channels + condition_channels, base_channels, kernel_size=3, padding=1)

        self.enc1 = ResBlock(base_channels, base_channels, time_dim)
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.enc2 = ResBlock(base_channels * 2, base_channels * 2, time_dim)
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1)

        self.mid1 = ResBlock(base_channels * 4, base_channels * 4, time_dim)
        self.mid2 = ResBlock(base_channels * 4, base_channels * 4, time_dim)

        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1))
        self.dec2 = ResBlock(base_channels * 4, base_channels * 2, time_dim)
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), nn.Conv2d(base_channels * 2, base_channels, 3, padding=1))
        self.dec1 = ResBlock(base_channels * 2, base_channels, time_dim)

        self.out_norm = nn.GroupNorm(norm_groups(base_channels), base_channels)
        self.out_conv = nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)
        self.act = nn.SiLU(inplace=True)

    def forward(self, noisy: torch.Tensor, condition: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if condition.shape[-2:] != noisy.shape[-2:]:
            condition = F.interpolate(condition, size=noisy.shape[-2:], mode="bilinear", align_corners=False)

        time_emb = sinusoidal_time_embedding(timesteps, self.time_dim)
        time_emb = self.time_mlp(time_emb)

        x = self.in_conv(torch.cat([noisy, condition], dim=1))
        enc1 = self.enc1(x, time_emb)
        enc2 = self.enc2(self.down1(enc1), time_emb)

        mid = self.mid2(self.mid1(self.down2(enc2), time_emb), time_emb)

        dec2 = self.up2(mid)
        dec2 = self.dec2(torch.cat([dec2, enc2], dim=1), time_emb)
        dec1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat([dec1, enc1], dim=1), time_emb)
        return self.out_conv(self.act(self.out_norm(dec1)))


class DiffusionSR(nn.Module):
    def __init__(
        self,
        condition_channels: int = 10,
        out_channels: int = 1,
        base_channels: int = 32,
        timesteps: int = 400,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.timesteps = timesteps
        self.unet = ConditionalUNet(out_channels, condition_channels, base_channels=base_channels)

        betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_cumprod", alpha_cumprod)
        self.register_buffer("sqrt_alpha_cumprod", torch.sqrt(alpha_cumprod))
        self.register_buffer("sqrt_one_minus_alpha_cumprod", torch.sqrt(1.0 - alpha_cumprod))

    def _base_image(self, condition: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        base = condition[:, :1]
        if base.shape[-2:] != output_size:
            base = F.interpolate(base, size=output_size, mode="bicubic", align_corners=False)
        return base

    def q_sample(self, clean: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        mean = extract(self.sqrt_alpha_cumprod, timesteps, clean.shape) * clean
        std = extract(self.sqrt_one_minus_alpha_cumprod, timesteps, clean.shape)
        return mean + std * noise

    def predict_noise(self, noisy_residual: torch.Tensor, condition: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return self.unet(noisy_residual, condition, timesteps)

    def training_loss(self, condition: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        timesteps = torch.randint(0, self.timesteps, (hr.shape[0],), device=hr.device, dtype=torch.long)
        base = self._base_image(condition, hr.shape[-2:])
        residual = hr - base
        noise = torch.randn_like(residual)
        noisy_residual = self.q_sample(residual, timesteps, noise)
        predicted_noise = self.predict_noise(noisy_residual, condition, timesteps)
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        output_size: tuple[int, int],
        sample_steps: int = 50,
    ) -> torch.Tensor:
        batch_size = condition.shape[0]
        device = condition.device
        residual = torch.randn(batch_size, self.out_channels, *output_size, device=device)
        base = self._base_image(condition, output_size)

        schedule = torch.linspace(self.timesteps - 1, 0, steps=sample_steps, device=device)
        schedule = schedule.round().long().unique(sorted=True).flip(0)

        for index, timestep in enumerate(schedule):
            timestep_batch = torch.full((batch_size,), int(timestep.item()), device=device, dtype=torch.long)
            predicted_noise = self.predict_noise(residual, condition, timestep_batch)

            alpha_t = self.alpha_cumprod[timestep]
            sqrt_alpha_t = alpha_t.sqrt()
            sqrt_one_minus_alpha_t = (1.0 - alpha_t).sqrt()
            pred_x0 = (residual - sqrt_one_minus_alpha_t * predicted_noise) / sqrt_alpha_t.clamp(min=1e-6)

            if index == len(schedule) - 1:
                residual = pred_x0
                continue

            next_timestep = schedule[index + 1]
            alpha_next = self.alpha_cumprod[next_timestep]
            residual = alpha_next.sqrt() * pred_x0 + (1.0 - alpha_next).sqrt() * predicted_noise

        return (base + residual).clamp(0.0, 1.0)
