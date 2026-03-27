from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from .models import DiffusionSR, EDSR, PatchDiscriminator, RCAN, RLFBESANet, RRDBNet, SRCNN, SRGANGenerator, SwinIRNet


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    default_input_mode: str
    description: str


MODEL_SPECS: dict[str, ModelSpec] = {
    "srcnn": ModelSpec("srcnn", "pixel", "grayscale", "Shallow baseline with bicubic pre-upsample."),
    "rlfb_esa": ModelSpec("rlfb_esa", "pixel", "solar_features", "Lightweight feature-distillation model for solar SR."),
    "edsr": ModelSpec("edsr", "pixel", "grayscale", "Residual SR baseline with stronger capacity than SRCNN."),
    "rcan": ModelSpec("rcan", "pixel", "grayscale", "Channel-attention SR model with stronger detail recovery."),
    "swinir": ModelSpec("swinir", "pixel", "grayscale", "Transformer-based SR model with shifted window attention."),
    "srgan": ModelSpec("srgan", "gan", "grayscale", "Perceptual GAN baseline."),
    "esrgan": ModelSpec("esrgan", "gan", "grayscale", "RRDB-based GAN generator for sharper reconstructions."),
    "diffusion_sr": ModelSpec("diffusion_sr", "diffusion", "grayscale", "Conditional residual diffusion model for solar SR."),
}


CAPACITY_PRESETS: dict[str, dict[str, dict[str, int | float | tuple[int, int]]]] = {
    "srcnn": {
        "tiny": {"features": (48, 24)},
        "base": {"features": (64, 32)},
        "large": {"features": (96, 48)},
    },
    "rlfb_esa": {
        "tiny": {"num_features": 48, "num_rlfb": 6},
        "base": {"num_features": 64, "num_rlfb": 12},
        "large": {"num_features": 80, "num_rlfb": 16},
    },
    "edsr": {
        "tiny": {"num_features": 48, "num_blocks": 8, "res_scale": 0.1},
        "base": {"num_features": 64, "num_blocks": 16, "res_scale": 0.1},
        "large": {"num_features": 96, "num_blocks": 24, "res_scale": 0.1},
    },
    "rcan": {
        "tiny": {"num_features": 48, "num_groups": 4, "num_blocks": 6, "reduction": 12},
        "base": {"num_features": 64, "num_groups": 6, "num_blocks": 10, "reduction": 16},
        "large": {"num_features": 96, "num_groups": 8, "num_blocks": 12, "reduction": 16},
    },
    "swinir": {
        "tiny": {"embed_dim": 48, "depths": (4, 4, 4, 4), "num_heads": (4, 4, 4, 4), "window_size": 8},
        "base": {"embed_dim": 60, "depths": (6, 6, 6, 6), "num_heads": (6, 6, 6, 6), "window_size": 8},
        "large": {"embed_dim": 96, "depths": (6, 6, 6, 6, 6, 6), "num_heads": (6, 6, 6, 6, 6, 6), "window_size": 8},
    },
    "srgan": {
        "tiny": {"num_features": 48, "num_blocks": 8},
        "base": {"num_features": 64, "num_blocks": 16},
        "large": {"num_features": 96, "num_blocks": 20},
    },
    "esrgan": {
        "tiny": {"num_features": 48, "num_blocks": 6, "growth_channels": 24},
        "base": {"num_features": 64, "num_blocks": 12, "growth_channels": 32},
        "large": {"num_features": 96, "num_blocks": 18, "growth_channels": 32},
    },
    "diffusion_sr": {
        "tiny": {"base_channels": 24, "timesteps": 300},
        "base": {"base_channels": 32, "timesteps": 400},
        "large": {"base_channels": 48, "timesteps": 500},
    },
}


def list_available_models() -> list[str]:
    return list(MODEL_SPECS)


def candidate_order(diffusion_centric: bool = True) -> list[str]:
    if diffusion_centric:
        return ["diffusion_sr", "swinir", "rcan", "edsr", "rlfb_esa", "esrgan", "srgan", "srcnn"]
    return ["swinir", "rcan", "edsr", "diffusion_sr", "rlfb_esa", "esrgan", "srgan", "srcnn"]


def final_fit_order(model_names: list[str]) -> list[str]:
    family_priority = {"pixel": 0, "gan": 1, "diffusion": 2}
    return sorted(model_names, key=lambda name: (family_priority.get(MODEL_SPECS[name].family, 3), name))


def suggest_capacity(model_name: str, train_size: int) -> str:
    if model_name not in MODEL_SPECS:
        raise KeyError(f"Unknown model: {model_name}")

    if model_name in {"diffusion_sr", "rcan", "swinir"}:
        if train_size < 2500:
            return "tiny"
        if train_size < 10000:
            return "base"
        return "large"
    if model_name in {"edsr", "esrgan", "srgan"}:
        if train_size < 2000:
            return "tiny"
        if train_size < 8000:
            return "base"
        return "large"
    if train_size < 1500:
        return "tiny"
    if train_size < 7000:
        return "base"
    return "large"


def build_model(
    model_name: str,
    in_channels: int,
    out_channels: int = 1,
    scale: int = 4,
    capacity: str = "base",
    diffusion_self_condition: bool = True,
) -> nn.Module:
    if model_name not in MODEL_SPECS:
        raise KeyError(f"Unknown model: {model_name}")
    if capacity not in CAPACITY_PRESETS[model_name]:
        raise KeyError(f"Unknown capacity '{capacity}' for model '{model_name}'")

    kwargs = dict(CAPACITY_PRESETS[model_name][capacity])
    kwargs.update({"scale": scale})

    if model_name == "srcnn":
        return SRCNN(in_channels=in_channels, out_channels=out_channels, **kwargs)
    if model_name == "rlfb_esa":
        return RLFBESANet(in_channels=in_channels, out_channels=out_channels, **kwargs)
    if model_name == "edsr":
        return EDSR(in_channels=in_channels, out_channels=out_channels, **kwargs)
    if model_name == "rcan":
        return RCAN(in_channels=in_channels, out_channels=out_channels, **kwargs)
    if model_name == "swinir":
        return SwinIRNet(in_channels=in_channels, out_channels=out_channels, **kwargs)
    if model_name == "srgan":
        return SRGANGenerator(in_channels=in_channels, out_channels=out_channels, **kwargs)
    if model_name == "esrgan":
        return RRDBNet(in_channels=in_channels, out_channels=out_channels, **kwargs)
    if model_name == "diffusion_sr":
        kwargs.pop("scale", None)
        return DiffusionSR(
            condition_channels=in_channels,
            out_channels=out_channels,
            scale=scale,
            self_condition=diffusion_self_condition,
            **kwargs,
        )
    raise KeyError(f"Unhandled model name: {model_name}")


def build_discriminator(model_name: str, in_channels: int = 1) -> nn.Module:
    if MODEL_SPECS[model_name].family != "gan":
        raise ValueError(f"{model_name} is not a GAN family model.")
    return PatchDiscriminator(in_channels=in_channels)
