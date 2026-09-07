"""Leakage-safe fold datasets and fixed three-view construction."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_CONFIG = ROOT / "configs/data.json"
DEFAULT_ASSET_ROOT = ROOT / "artifacts/fold_assignments"
TARGET_NAMES = ("country", "coarse", "a", "b", "c", "fine")
REGIONS = {"coarse": 6, "a": 8, "b": 12, "c": 18, "fine": 20}
ISO_ORDER = ("BY", "DE", "ES", "FI", "FR", "GB", "IS", "IT", "NO", "PL", "SE", "TR")


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_fold_assignments(
    fold: int,
    asset_root: Path = DEFAULT_ASSET_ROOT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if fold not in range(5):
        raise ValueError("fold must be an integer from 0 through 4")
    directory = asset_root / f"fold_{fold}"
    train_path = directory / "training_assignments.csv"
    validation_path = directory / "validation_assignments.csv"
    if not train_path.is_file() or not validation_path.is_file():
        raise FileNotFoundError(f"Missing fold assets in {directory}")
    training = pd.read_csv(train_path)
    validation = pd.read_csv(validation_path)
    _validate_assignment_frames(training, validation, fold)
    return training, validation


def _validate_assignment_frames(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    fold: int,
) -> None:
    required = {
        "filename",
        "country",
        "iso",
        "lat",
        "lng",
        "fold",
        "country_index",
        "coarse",
        "a",
        "b",
        "c",
        "fine",
    }
    for name, frame in (("training", training), ("validation", validation)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} assignments lack columns: {sorted(missing)}")
        if frame.empty or frame.isna().any().any():
            raise ValueError(f"{name} assignments are empty or contain missing values")
        if frame["filename"].duplicated().any():
            raise ValueError(f"{name} filenames are not unique")
        if set(frame["iso"]) - set(ISO_ORDER):
            raise ValueError(f"{name} contains an unexpected country")

        expected_country = frame["iso"].map({iso: i for i, iso in enumerate(ISO_ORDER)})
        actual_country = frame["country_index"].astype(np.int64)
        if not np.array_equal(expected_country.to_numpy(), actual_country.to_numpy()):
            raise ValueError(f"{name} country indices are inconsistent")
        for target, regions in REGIONS.items():
            values = frame[target].to_numpy(dtype=np.int64)
            if not np.array_equal(values // regions, actual_country.to_numpy()):
                raise ValueError(f"{name} {target} labels disagree with countries")

    if set(training["fold"]) == {fold} or (training["fold"] == fold).any():
        raise ValueError("Training assignments contain the validation fold")
    if set(validation["fold"]) != {fold}:
        raise ValueError("Validation assignments do not belong exclusively to the fold")
    train_names = set(training["filename"])
    validation_names = set(validation["filename"])
    if train_names & validation_names:
        raise ValueError("Training and validation assignments overlap")
    if len(training) + len(validation) != 11_758:
        raise ValueError("Fold assignments do not cover 11,758 images")


class ThreeViewTransform:
    """Preserve the full scene plus aligned left/right high-resolution cues."""

    def __init__(
        self, mean: list[float], std: list[float], training: bool, config: dict
    ) -> None:
        if len(mean) != 3 or len(std) != 3 or any(value <= 0 for value in std):
            raise ValueError("Normalization mean/std must contain three valid channels")
        self.mean = mean
        self.std = std
        self.training = bool(training)
        self.global_size = int(config["global_size"])
        self.local_size = int(config["local_size"])
        augmentation = config["augmentation"]
        self.left_range = tuple(
            float(x) for x in augmentation["left_crop_end_fraction_range"]
        )
        self.right_range = tuple(
            float(x) for x in augmentation["right_crop_start_fraction_range"]
        )
        self.validation_left = float(config["validation_left_crop_end_fraction"])
        self.validation_right = float(config["validation_right_crop_start_fraction"])
        self.color_probability = float(augmentation["color_jitter_probability"])
        self.blur_probability = float(augmentation["blur_probability"])
        self.color_jitter = transforms.ColorJitter(
            brightness=float(augmentation["brightness"]),
            contrast=float(augmentation["contrast"]),
            saturation=float(augmentation["saturation"]),
            hue=float(augmentation["hue"]),
        )
        if float(augmentation["horizontal_flip_probability"]) != 0.0:
            raise ValueError("Horizontal flips are forbidden by the project contract")
        if float(augmentation["grayscale_probability"]) != 0.0:
            raise ValueError(
                "Grayscale augmentation is forbidden by the project contract"
            )
        if float(augmentation["random_erasing_probability"]) != 0.0:
            raise ValueError("Random erasing is forbidden by the project contract")

    def _tensor(self, image: Image.Image, size: int) -> Tensor:
        resized = TF.resize(
            image,
            [size, size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        value = TF.pil_to_tensor(resized).to(dtype=torch.float32).div_(255.0)
        return TF.normalize(value, mean=self.mean, std=self.std)

    def __call__(self, image: Image.Image) -> tuple[Tensor, Tensor, Tensor]:
        if image.mode != "RGB":
            image = image.convert("RGB")
        if self.training:
            if torch.rand(()).item() < self.color_probability:
                image = self.color_jitter(image)
            if torch.rand(()).item() < self.blur_probability:
                image = TF.gaussian_blur(image, kernel_size=[5, 5], sigma=[0.1, 1.0])
            left_end_fraction = float(torch.empty(()).uniform_(*self.left_range).item())
            right_start_fraction = float(
                torch.empty(()).uniform_(*self.right_range).item()
            )
        else:
            left_end_fraction = self.validation_left
            right_start_fraction = self.validation_right

        width, height = image.size
        left_end = max(1, min(width, int(round(width * left_end_fraction))))
        right_start = max(0, min(width - 1, int(round(width * right_start_fraction))))
        left = image.crop((0, 0, left_end, height))
        right = image.crop((right_start, 0, width, height))
        return (
            self._tensor(image, self.global_size),
            self._tensor(left, self.local_size),
            self._tensor(right, self.local_size),
        )


class GeoMatchFoldDataset(Dataset):
    def __init__(
        self,
        rows: pd.DataFrame,
        image_directory: Path,
        transform: ThreeViewTransform,
    ) -> None:
        self.rows = rows.reset_index(drop=True).copy()
        self.image_directory = image_directory.resolve()
        self.transform = transform
        if not self.image_directory.is_dir():
            raise FileNotFoundError(self.image_directory)
        unsafe = [name for name in self.rows["filename"] if Path(name).name != name]
        if unsafe:
            raise ValueError(f"Unsafe image filenames found: {unsafe[:3]}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        row = self.rows.iloc[index]
        image_path = self.image_directory / row["filename"]
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source)
            if image.size != (512, 512):
                raise ValueError(f"Unexpected image size {image.size}: {image_path}")
            image = image.convert("RGB")
            global_view, left_view, right_view = self.transform(image)

        item: dict[str, Tensor | str] = {
            "filename": str(row["filename"]),
            "global_view": global_view,
            "left_view": left_view,
            "right_view": right_view,
            "coordinates": torch.tensor([row["lat"], row["lng"]], dtype=torch.float64),
            "country": torch.tensor(int(row["country_index"]), dtype=torch.long),
        }
        for target in ("coarse", "a", "b", "c", "fine"):
            item[target] = torch.tensor(int(row[target]), dtype=torch.long)
        return item


def seed_worker(_: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(
    dataset: Dataset,
    batch_size: int,
    training: bool,
    workers: int,
    seed: int,
    drop_last: bool | None = None,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        drop_last=training if drop_last is None else drop_last,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def build_fold_datasets(
    fold: int,
    image_directory: Path,
    normalization_path: Path,
    data_config_path: Path = DEFAULT_DATA_CONFIG,
    asset_root: Path = DEFAULT_ASSET_ROOT,
) -> tuple[GeoMatchFoldDataset, GeoMatchFoldDataset]:
    config = read_json(data_config_path)
    normalization = read_json(normalization_path)
    if int(normalization["fold"]) != fold:
        raise ValueError("Normalization artifact belongs to a different fold")
    if normalization["fit_policy"] != "fold != k training images only":
        raise ValueError("Unexpected normalization fit policy")
    training, validation = load_fold_assignments(fold, asset_root=asset_root)
    train_transform = ThreeViewTransform(
        normalization["mean"], normalization["std"], training=True, config=config
    )
    validation_transform = ThreeViewTransform(
        normalization["mean"], normalization["std"], training=False, config=config
    )
    return (
        GeoMatchFoldDataset(training, image_directory, train_transform),
        GeoMatchFoldDataset(validation, image_directory, validation_transform),
    )
