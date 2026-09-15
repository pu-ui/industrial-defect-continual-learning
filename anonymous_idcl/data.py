from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Sample:
    path: str
    class_index: int
    is_anomaly: int


def read_manifest(path: str | Path) -> list[Sample]:
    """Read a simple CSV manifest: path,class_index,is_anomaly."""
    rows: list[Sample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
        if header != ["path", "class_index", "is_anomaly"]:
            raise ValueError("Expected CSV header: path,class_index,is_anomaly")
        for line in handle:
            image_path, class_index, is_anomaly = line.strip().split(",")
            rows.append(Sample(image_path, int(class_index), int(is_anomaly)))
    return rows


class ImageDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Sample],
        resize: int,
        image_size: int,
        train: bool,
        label_overrides: dict[str, int] | None = None,
    ):
        self.samples = list(samples)
        self.label_overrides = label_overrides or {}
        ops = [transforms.Resize(resize), transforms.CenterCrop(image_size)]
        if train:
            ops.extend(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(5),
                    transforms.ColorJitter(brightness=0.08, contrast=0.08),
                ]
            )
        ops.extend(
            [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
        )
        self.transform = transforms.Compose(ops)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            tensor = self.transform(image.convert("RGB"))
        class_index = self.label_overrides.get(sample.path, sample.class_index)
        return {
            "image": tensor,
            "class_index": torch.tensor(class_index, dtype=torch.long),
            "is_anomaly": torch.tensor(sample.is_anomaly, dtype=torch.long),
            "path": sample.path,
        }
