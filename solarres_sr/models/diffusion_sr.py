from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import PixelShuffleUpsampler, ResidualBlock, default_conv, norm_groups


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


class GuideSRHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_features: int,
        num_blocks: int,
        scale: int,
    ) -> None:
        super().__init__()
        self.head = default_conv(in_channels, num_features, 3)
        self.body = nn.Sequential(*[ResidualBlock(num_features, res_scale=0.1) for _ in range(num_blocks)])
        self.body_conv = default_conv(num_features, num_features, 3)
        self.upsampler = PixelShuffleUpsampler(num_features, scale, activation=nn.ReLU)
        self.tail = default_conv(num_features, out_channels, 3)

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        head = self.head(condition)
        body = self.body_conv(self.body(head)) + head
        return self.tail(self.upsampler(body))


class ConditionalUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        condition_channels: int,
        base_channels: int = 32,
        self_condition: bool = True,
    ) -> None:
        super().__init__()
        self.self_condition = bool(self_condition)
        time_dim = base_channels * 4
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(inplace=True),
            nn.Linear(time_dim, time_dim),
        )
        extra_channels = in_channels if self.self_condition else 0
        self.in_conv = nn.Conv2d(in_channels + condition_channels + extra_channels, base_channels, kernel_size=3, padding=1)

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

    def forward(
        self,
        noisy: torch.Tensor,
        condition: torch.Tensor,
        timesteps: torch.Tensor,
        self_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if condition.shape[-2:] != noisy.shape[-2:]:
            condition = F.interpolate(condition, size=noisy.shape[-2:], mode="bilinear", align_corners=False)
        if self.self_condition:
            if self_condition is None:
                self_condition = torch.zeros_like(noisy)
            x_in = torch.cat([noisy, self_condition, condition], dim=1)
        else:
            x_in = torch.cat([noisy, condition], dim=1)

        time_emb = sinusoidal_time_embedding(timesteps, self.time_dim)
        time_emb = self.time_mlp(time_emb)

        x = self.in_conv(x_in)
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
        scale: int = 4,
        self_condition: bool = True,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.timesteps = timesteps
        self.scale = scale
        self.self_condition = bool(self_condition)
        guide_features = max(base_channels * 2, 48)
        guide_blocks = 4 if base_channels <= 24 else (6 if base_channels <= 32 else 8)
        self.guide = GuideSRHead(
            in_channels=condition_channels,
            out_channels=out_channels,
            num_features=guide_features,
            num_blocks=guide_blocks,
            scale=scale,
        )
        self.unet = ConditionalUNet(
            out_channels,
            condition_channels,
            base_channels=base_channels,
            self_condition=self.self_condition,
        )

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

    def guide_image(self, condition: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        base = self._base_image(condition, output_size)
        guide_residual = self.guide(condition)
        if guide_residual.shape[-2:] != output_size:
            guide_residual = F.interpolate(guide_residual, size=output_size, mode="bicubic", align_corners=False)
        return base + guide_residual

    def sample_timesteps(self, batch_size: int, device: torch.device, bias_power: float = 1.0) -> torch.Tensor:
        if bias_power <= 0:
            raise ValueError(f"bias_power must be > 0, got {bias_power}")
        samples = torch.rand(batch_size, device=device)
        if abs(bias_power - 1.0) > 1e-6:
            samples = samples.pow(float(bias_power))
        return torch.clamp((samples * self.timesteps).long(), max=self.timesteps - 1)

    def q_sample(self, clean: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        mean = extract(self.sqrt_alpha_cumprod, timesteps, clean.shape) * clean
        std = extract(self.sqrt_one_minus_alpha_cumprod, timesteps, clean.shape)
        return mean + std * noise

    def predict_noise(
        self,
        noisy_residual: torch.Tensor,
        condition: torch.Tensor,
        timesteps: torch.Tensor,
        self_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.unet(noisy_residual, condition, timesteps, self_condition=self_condition)

    def predict_x0_from_noise(
        self,
        noisy_residual: torch.Tensor,
        predicted_noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        sqrt_alpha = extract(self.sqrt_alpha_cumprod, timesteps, noisy_residual.shape)
        sqrt_one_minus_alpha = extract(self.sqrt_one_minus_alpha_cumprod, timesteps, noisy_residual.shape)
        return (noisy_residual - sqrt_one_minus_alpha * predicted_noise) / sqrt_alpha.clamp(min=1e-6)

    def training_loss(
        self,
        condition: torch.Tensor,
        hr: torch.Tensor,
        reconstruction_criterion: nn.Module | None = None,
        recon_weight: float = 0.0,
        guide_weight: float = 0.0,
        x0_weight: float = 0.0,
        timestep_bias: float = 1.0,
        clamp_range: tuple[float | None, float | None] | None = None,
    ) -> torch.Tensor:
        timesteps = self.sample_timesteps(hr.shape[0], hr.device, bias_power=timestep_bias)
        guide = self.guide_image(condition, hr.shape[-2:])
        residual = hr - guide
        noise = torch.randn_like(residual)
        noisy_residual = self.q_sample(residual, timesteps, noise)
        self_condition = None
        if self.self_condition and torch.rand(1, device=hr.device).item() < 0.5:
            with torch.no_grad():
                initial_noise = self.predict_noise(noisy_residual, condition, timesteps)
                self_condition = self.predict_x0_from_noise(noisy_residual, initial_noise, timesteps).detach()

        predicted_noise = self.predict_noise(noisy_residual, condition, timesteps, self_condition=self_condition)
        predicted_residual = self.predict_x0_from_noise(noisy_residual, predicted_noise, timesteps)
        predicted_hr = guide + predicted_residual
        if clamp_range is not None:
            clamp_min, clamp_max = clamp_range
            if clamp_min is not None or clamp_max is not None:
                low = -float("inf") if clamp_min is None else clamp_min
                high = float("inf") if clamp_max is None else clamp_max
                predicted_hr = predicted_hr.clamp(low, high)

        timestep_weight = 1.0 + (
            1.0 - timesteps.float() / float(max(self.timesteps - 1, 1))
        )
        noise_loss = (((predicted_noise - noise) ** 2).flatten(1).mean(dim=1) * timestep_weight).mean()
        total = noise_loss
        if x0_weight > 0.0:
            x0_loss = ((predicted_residual - residual).abs().flatten(1).mean(dim=1) * timestep_weight).mean()
            total = total + float(x0_weight) * x0_loss
        if reconstruction_criterion is not None and recon_weight > 0.0:
            recon_loss, _ = reconstruction_criterion(predicted_hr, hr)
            total = total + float(recon_weight) * recon_loss
        if reconstruction_criterion is not None and guide_weight > 0.0:
            guide_loss, _ = reconstruction_criterion(guide, hr)
            total = total + float(guide_weight) * guide_loss
        return total

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        output_size: tuple[int, int],
        sample_steps: int = 50,
        sampler: str = "ddpm",
        clamp_output: bool = False,
        output_clamp: tuple[float, float] | None = None,
        clamp_pred_x0: bool = False,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """Sample from the diffusion model.

        Args:
            condition: Conditioning input tensor
            output_size: Target output spatial size (H, W)
            sample_steps: Number of sampling steps
            sampler: Sampling method - "ddpm", "ddim", or "deterministic_fast"
            clamp_output: Whether to clamp final output
            output_clamp: Clamp range (min, max) for output
            clamp_pred_x0: Whether to clamp predicted x0 at each step
            eta: DDIM stochasticity (0=deterministic, 1=DDPM equivalent)
        """
        if sampler not in {"ddpm", "ddim", "deterministic_fast"}:
            raise ValueError(f"Unsupported sampler: {sampler}")

        effective_steps = max(1, int(sample_steps))
        if sampler == "deterministic_fast":
            # Alias for deterministic DDIM with reduced steps
            sampler = "ddim"
            eta = 0.0
            effective_steps = max(4, effective_steps // 2)

        batch_size = condition.shape[0]
        device = condition.device
        residual = torch.randn(batch_size, self.out_channels, *output_size, device=device)
        guide = self.guide_image(condition, output_size)
        self_condition = None

        # Create timestep schedule
        schedule = torch.linspace(self.timesteps - 1, 0, steps=effective_steps, device=device)
        schedule = schedule.round().long().unique(sorted=True).flip(0)

        for index, timestep in enumerate(schedule):
            timestep_batch = torch.full((batch_size,), int(timestep.item()), device=device, dtype=torch.long)
            predicted_noise = self.predict_noise(residual, condition, timestep_batch, self_condition=self_condition)

            alpha_t = self.alpha_cumprod[timestep]
            sqrt_alpha_t = alpha_t.sqrt()
            sqrt_one_minus_alpha_t = (1.0 - alpha_t).sqrt()

            # Predict clean residual x0
            pred_x0 = (residual - sqrt_one_minus_alpha_t * predicted_noise) / sqrt_alpha_t.clamp(min=1e-6)

            if clamp_pred_x0 and output_clamp is not None:
                pred_hr = (guide + pred_x0).clamp(output_clamp[0], output_clamp[1])
                pred_x0 = pred_hr - guide

            if self.self_condition:
                self_condition = pred_x0.detach()

            # Last step - just return pred_x0
            if index == len(schedule) - 1:
                residual = pred_x0
                continue

            next_timestep = schedule[index + 1]
            alpha_next = self.alpha_cumprod[next_timestep]

            if sampler == "ddim":
                # DDIM sampling equation
                # x_{t-1} = sqrt(alpha_{t-1}) * pred_x0 + sqrt(1 - alpha_{t-1} - sigma^2) * direction + sigma * noise
                # sigma = eta * sqrt((1 - alpha_{t-1})/(1 - alpha_t)) * sqrt(1 - alpha_t/alpha_{t-1})

                sigma = 0.0
                if eta > 0:
                    sigma = eta * math.sqrt((1.0 - alpha_next) / (1.0 - alpha_t).clamp(min=1e-8)) * math.sqrt(
                        (1.0 - alpha_t / alpha_next.clamp(min=1e-8)).clamp(min=0)
                    )

                # Direction pointing to x_t
                direction = math.sqrt(max(1.0 - alpha_next - sigma**2, 0)) * predicted_noise

                # Sample x_{t-1}
                residual = alpha_next.sqrt() * pred_x0 + direction
                if sigma > 0:
                    residual = residual + sigma * torch.randn_like(residual)
            else:
                # DDPM sampling (original behavior)
                residual = alpha_next.sqrt() * pred_x0 + (1.0 - alpha_next).sqrt() * predicted_noise

        output = guide + residual
        if clamp_output and output_clamp is not None:
            output = output.clamp(output_clamp[0], output_clamp[1])
        return output
