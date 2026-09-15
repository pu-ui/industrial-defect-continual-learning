from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class MethodConfig:
    seed: int = 0
    resize: int = 256
    image_size: int = 224
    num_workers: int = 4
    risk_batch_size: int = 8
    classifier_batch_size: int = 32

    patchcore_backbone: str = "wideresnet50"
    patchcore_layers: tuple[str, ...] = ("layer2", "layer3")
    pretrain_embed_dimension: int = 1024
    target_embed_dimension: int = 1024
    patch_size: int = 3
    coreset_ratio: float = 0.1
    faiss_on_gpu: bool = False

    alpha: float = 0.01
    target_recall: float = 0.95
    beta: float = 0.5
    defect_quantile: float = 0.05
    safety_margin: float = 0.05

    initial_epochs: int = 50
    continual_epochs: int = 20
    early_stopping_patience: int = 5
    initial_learning_rate: float = 1e-3
    continual_learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    focal_gamma: float = 2.0
    kd_temperature: float = 2.0
    kd_weight: float = 0.5

    device: str = "cuda"
    pretrained: bool = True
    use_coordinate_attention: bool = True
    use_pecl: bool = True
    use_kd: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MethodConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            values: dict[str, Any] = yaml.safe_load(handle) or {}
        if "patchcore_layers" in values:
            values["patchcore_layers"] = tuple(values["patchcore_layers"])
        return cls(**values)

    def resolved_device(self) -> str:
        import torch

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return self.device
