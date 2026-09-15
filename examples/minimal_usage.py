from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anonymous_idcl.config import MethodConfig
from anonymous_idcl.model import CoordinateAttentionResNet50, parameter_counts
from anonymous_idcl.risk import (
    initial_high_threshold,
    safe_threshold,
    select_samples,
)


def main() -> None:
    config = MethodConfig.from_yaml("configs/example_config.yaml")

    model = CoordinateAttentionResNet50(
        num_classes=6,
        pretrained=False,
        use_coordinate_attention=config.use_coordinate_attention,
    )
    model.freeze_for_continual_learning()
    print("parameter counts:", parameter_counts(model))

    normal_scores = [0.10, 0.12, 0.15, 0.17, 0.21, 0.26]
    defect_scores = [0.50, 0.67, 0.72, 0.81]
    high_threshold = initial_high_threshold(normal_scores, config.alpha)
    low_threshold = safe_threshold(
        normal_scores,
        defect_scores,
        config.beta,
        config.defect_quantile,
        config.safety_margin,
    )
    print("high threshold:", high_threshold)
    print("low threshold:", low_threshold)

    class DummySample:
        def __init__(self, path: str, class_index: int, is_anomaly: int):
            self.path = path
            self.class_index = class_index
            self.is_anomaly = is_anomaly

    samples = [
        DummySample("sample_001.png", 0, 0),
        DummySample("sample_002.png", 1, 1),
        DummySample("sample_003.png", 0, 0),
    ]
    selections = select_samples(samples, [0.11, 0.75, 0.32], high_threshold, low_threshold)
    for item in selections:
        print(item.sample.path, item.score, item.action, item.training_label)

    with torch.no_grad():
        logits = model(torch.randn(2, 3, config.image_size, config.image_size))
    print("logit shape:", tuple(logits.shape))


if __name__ == "__main__":
    main()
