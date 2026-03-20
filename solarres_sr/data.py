from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def resolve_project_root(start: str | Path | None = None) -> Path:
    root = Path(start or Path.cwd()).resolve()
    if root.name.lower() == "notebooks":
        root = root.parent
    return root


def find_dataset_root(start: str | Path | None = None) -> Path:
    project_root = resolve_project_root(start)
    candidates = [
        project_root / "Solar Dataset" / "Solar Dataset",
        project_root / "Solar Dataset",
    ]
    dataset_root = next((candidate for candidate in candidates if candidate.exists()), None)
    if dataset_root is None:
        raise FileNotFoundError(
            "Could not find the solar dataset. Expected either "
            "'Solar Dataset/Solar Dataset' or 'Solar Dataset' under the project root."
        )
    return dataset_root


def resolve_split_dirs(
    dataset_root: str | Path | None = None,
    train_split: str = "training",
    val_split: str = "validation",
) -> dict[str, Path]:
    root = find_dataset_root(dataset_root)
    directories = {
        "train_lr": root / train_split / "low_res",
        "train_hr": root / train_split / "high_res",
        "val_lr": root / val_split / "low_res",
        "val_hr": root / val_split / "high_res",
    }
    missing = [str(path) for path in directories.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing dataset directories: {missing}")
    return directories


def infer_input_channels(input_mode: str) -> int:
    if input_mode == "solar_features":
        return 10
    if input_mode == "rgb":
        return 3
    if input_mode == "grayscale":
        return 1
    raise ValueError(f"Unsupported input_mode: {input_mode}")


def _valid_image_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
    )


def pair_image_files(lr_dir: str | Path, hr_dir: str | Path) -> list[tuple[Path, Path]]:
    lr_dir = Path(lr_dir)
    hr_dir = Path(hr_dir)
    if not lr_dir.exists():
        raise FileNotFoundError(f"LR directory not found: {lr_dir}")
    if not hr_dir.exists():
        raise FileNotFoundError(f"HR directory not found: {hr_dir}")

    lr_files = _valid_image_files(lr_dir)
    hr_files = _valid_image_files(hr_dir)
    if not lr_files or not hr_files:
        raise FileNotFoundError(f"No image pairs found in {lr_dir} and {hr_dir}")

    lr_by_stem = {path.stem: path for path in lr_files}
    hr_by_stem = {path.stem: path for path in hr_files}
    shared_stems = sorted(set(lr_by_stem) & set(hr_by_stem))
    if shared_stems:
        return [(lr_by_stem[stem], hr_by_stem[stem]) for stem in shared_stems]

    if len(lr_files) != len(hr_files):
        raise ValueError(
            "LR and HR folders do not share filenames and have different image counts, "
            "so they cannot be paired safely."
        )
    return list(zip(lr_files, hr_files))


def load_grayscale_image(path: str | Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.ascontiguousarray(array)


def load_rgb_image(path: str | Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.transpose(array, (2, 0, 1))


def preprocess_solar_image(
    img_path: str | Path,
    apply_clahe: bool = False,
    levels: int = 2,
) -> torch.Tensor:
    image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")

    image = image.astype(np.float32)
    image /= max(float(image.max()), 1e-8)

    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image_equalized = clahe.apply((image * 255).astype(np.uint8)).astype(np.float32) / 255.0
    else:
        image_equalized = np.log1p(image) / np.log(np.float32(2.0))
    image_equalized = np.ascontiguousarray(image_equalized, dtype=np.float32)

    features: list[np.ndarray] = [image_equalized]

    current = image_equalized.copy()
    laplacian_levels = []
    for _ in range(levels):
        down = cv2.pyrDown(current)
        up = cv2.pyrUp(down, dstsize=current.shape[::-1])
        lap = (current - up).astype(np.float32)
        laplacian_levels.append(cv2.resize(lap, image_equalized.shape[::-1]).astype(np.float32))
        current = down.astype(np.float32)
    features.extend(laplacian_levels)

    lap = cv2.Laplacian(image_equalized, cv2.CV_32F, ksize=3)
    sobelx = cv2.Sobel(image_equalized, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image_equalized, cv2.CV_32F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobelx**2 + sobely**2).astype(np.float32)
    g1 = cv2.GaussianBlur(image_equalized, (3, 3), 0.5)
    g2 = cv2.GaussianBlur(image_equalized, (3, 3), 1.5)
    dog = (g1 - g2).astype(np.float32)

    gabor_responses = []
    for theta in (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4):
        kernel = cv2.getGaborKernel((7, 7), 2.0, theta, 5.0, 0.5, 0, ktype=cv2.CV_32F)
        gabor_responses.append(cv2.filter2D(image_equalized, cv2.CV_32F, kernel).astype(np.float32))

    features.extend([lap, sobel_mag, dog, *gabor_responses])

    stacked = np.stack([np.asarray(feature, dtype=np.float32) for feature in features], axis=0)
    return torch.tensor(stacked, dtype=torch.float32)


def preprocess_solar_equalized_channel(
    img_path: str | Path,
    apply_clahe: bool = False,
) -> torch.Tensor:
    image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")

    image = image.astype(np.float32)
    image /= max(float(image.max()), 1e-8)

    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image_equalized = clahe.apply((image * 255).astype(np.uint8)).astype(np.float32) / 255.0
    else:
        image_equalized = np.log1p(image) / np.log(np.float32(2.0))

    return torch.from_numpy(np.ascontiguousarray(image_equalized, dtype=np.float32)).unsqueeze(0)


def load_input_tensor(
    path: str | Path,
    input_mode: str,
    apply_clahe: bool = False,
) -> torch.Tensor:
    if input_mode == "solar_features":
        return preprocess_solar_image(path, apply_clahe=apply_clahe)
    if input_mode == "grayscale":
        image = load_grayscale_image(path)
        return torch.from_numpy(image).unsqueeze(0)
    if input_mode == "rgb":
        return torch.from_numpy(load_rgb_image(path))
    raise ValueError(f"Unsupported input_mode: {input_mode}")


def load_target_tensor(
    path: str | Path,
    target_mode: str = "grayscale",
    apply_clahe: bool = False,
) -> torch.Tensor:
    if target_mode == "grayscale":
        if apply_clahe:
            return preprocess_solar_equalized_channel(path, apply_clahe=True)
        image = load_grayscale_image(path)
        return torch.from_numpy(image).unsqueeze(0)
    if target_mode == "solar_equalized":
        return preprocess_solar_equalized_channel(path, apply_clahe=apply_clahe)
    raise ValueError(f"Unsupported target_mode: {target_mode}")


def random_augment_pair(lr_tensor: torch.Tensor, hr_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if random.random() < 0.5:
        lr_tensor = torch.flip(lr_tensor, dims=[2])
        hr_tensor = torch.flip(hr_tensor, dims=[2])
    if random.random() < 0.5:
        lr_tensor = torch.flip(lr_tensor, dims=[1])
        hr_tensor = torch.flip(hr_tensor, dims=[1])
    k = random.randint(0, 3)
    if k:
        lr_tensor = torch.rot90(lr_tensor, k=k, dims=[1, 2])
        hr_tensor = torch.rot90(hr_tensor, k=k, dims=[1, 2])
    return lr_tensor.contiguous(), hr_tensor.contiguous()


class SolarSRDataset(Dataset):
    def __init__(
        self,
        lr_dir: str | Path,
        hr_dir: str | Path,
        input_mode: str = "solar_features",
        target_mode: str = "grayscale",
        apply_clahe: bool = False,
        augment: bool = False,
    ) -> None:
        self.pairs = pair_image_files(lr_dir, hr_dir)
        self.input_mode = input_mode
        self.target_mode = target_mode
        self.apply_clahe = apply_clahe
        self.augment = augment

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        lr_path, hr_path = self.pairs[index]
        lr_tensor = load_input_tensor(lr_path, self.input_mode, apply_clahe=self.apply_clahe)
        hr_tensor = load_target_tensor(hr_path, self.target_mode, apply_clahe=self.apply_clahe)
        if self.augment:
            lr_tensor, hr_tensor = random_augment_pair(lr_tensor, hr_tensor)
        return lr_tensor, hr_tensor


def count_images(paths: Iterable[Path]) -> int:
    return sum(1 for _ in paths)
