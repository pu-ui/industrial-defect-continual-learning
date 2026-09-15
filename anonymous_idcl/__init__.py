"""Core modules for risk-aware parameter-efficient continual learning."""

from .config import MethodConfig
from .model import CoordinateAttention, CoordinateAttentionResNet50, parameter_counts
from .risk import (
    Selection,
    initial_high_threshold,
    safe_threshold,
    select_samples,
    updated_high_threshold,
)
from .training import FocalLoss, distillation_loss, freeze_batch_norm_statistics

__all__ = [
    "MethodConfig",
    "CoordinateAttention",
    "CoordinateAttentionResNet50",
    "FocalLoss",
    "Selection",
    "distillation_loss",
    "freeze_batch_norm_statistics",
    "initial_high_threshold",
    "parameter_counts",
    "safe_threshold",
    "select_samples",
    "updated_high_threshold",
]
