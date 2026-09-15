from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import MethodConfig
from .data import ImageDataset, Sample


@dataclass(frozen=True)
class Selection:
    sample: Sample
    score: float
    action: str
    training_label: int | None


class PatchCoreRiskModel:
    """PatchCore-based risk scorer used for sample selection.

    The PatchCore dependency is imported lazily so the supervised modules can be
    inspected without installing the full anomaly-detection backend.
    """

    def __init__(self, config: MethodConfig):
        import patchcore.backbones
        import patchcore.common
        import patchcore.patchcore
        import patchcore.sampler

        self.config = config
        self.device = torch.device(config.resolved_device())
        backbone = patchcore.backbones.load(config.patchcore_backbone)
        backbone.name = config.patchcore_backbone
        backbone.seed = config.seed
        sampler = patchcore.sampler.ApproximateGreedyCoresetSampler(
            config.coreset_ratio, self.device
        )
        nn_method = patchcore.common.FaissNN(config.faiss_on_gpu, 8)
        self.model = patchcore.patchcore.PatchCore(self.device)
        self.model.load(
            backbone=backbone,
            layers_to_extract_from=config.patchcore_layers,
            device=self.device,
            input_shape=(3, config.image_size, config.image_size),
            pretrain_embed_dimension=config.pretrain_embed_dimension,
            target_embed_dimension=config.target_embed_dimension,
            patchsize=config.patch_size,
            featuresampler=sampler,
            anomaly_score_num_nn=1,
            nn_method=nn_method,
        )

    def _loader(self, samples: Sequence[Sample], shuffle: bool = False) -> DataLoader:
        dataset = ImageDataset(
            samples, self.config.resize, self.config.image_size, train=False
        )
        return DataLoader(
            dataset,
            batch_size=self.config.risk_batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.device.type == "cuda",
        )

    def fit(self, normal_samples: Sequence[Sample]) -> None:
        self.model.fit(self._loader(normal_samples))

    def score(self, samples: Sequence[Sample]) -> np.ndarray:
        scores, _, _, _ = self.model.predict(self._loader(samples))
        return np.asarray(scores, dtype=np.float64)

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save_to_path(str(directory))


def initial_high_threshold(normal_scores: Sequence[float], alpha: float) -> float:
    return float(np.quantile(np.asarray(normal_scores), 1.0 - alpha))


def updated_high_threshold(
    defect_scores: Sequence[float], target_recall: float, fallback: float
) -> float:
    if not defect_scores:
        return fallback
    return float(
        np.quantile(np.asarray(defect_scores), 1.0 - target_recall, method="lower")
    )


def safe_threshold(
    normal_scores: Sequence[float],
    defect_scores: Sequence[float],
    beta: float,
    defect_quantile: float,
    margin: float,
) -> float:
    normal_limit = float(np.quantile(np.asarray(normal_scores), beta))
    if not defect_scores:
        return normal_limit
    defect_limit = float(np.quantile(np.asarray(defect_scores), defect_quantile)) - margin
    return min(normal_limit, defect_limit)


def select_samples(
    samples: Sequence[Sample],
    scores: Sequence[float],
    high_threshold: float,
    low_threshold: float,
) -> list[Selection]:
    selections: list[Selection] = []
    for sample, score in zip(samples, scores):
        value = float(score)
        if value >= high_threshold:
            selections.append(Selection(sample, value, "reviewed", None))
        elif value <= low_threshold:
            selections.append(Selection(sample, value, "pseudo_normal", 0))
        else:
            selections.append(Selection(sample, value, "deferred", None))
    return selections


def apply_review_labels(
    selections: Iterable[Selection], labels_by_path: dict[str, int]
) -> list[Selection]:
    return [
        replace(item, training_label=labels_by_path[item.sample.path])
        if item.action == "reviewed" and item.sample.path in labels_by_path
        else item
        for item in selections
    ]


def selection_label_overrides(selections: Iterable[Selection]) -> dict[str, int]:
    return {
        item.sample.path: int(item.training_label)
        for item in selections
        if item.training_label is not None
    }
