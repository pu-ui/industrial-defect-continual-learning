from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


class FocalLoss(nn.Module):
    def __init__(self, class_weights: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probabilities = F.log_softmax(logits, dim=1)
        probabilities = log_probabilities.exp()
        indices = torch.arange(targets.shape[0], device=targets.device)
        target_log_probabilities = log_probabilities[indices, targets]
        target_probabilities = probabilities[indices, targets]
        target_weights = self.class_weights[targets]
        return (
            -target_weights
            * (1.0 - target_probabilities).pow(self.gamma)
            * target_log_probabilities
        ).mean()


def class_weights(labels: Sequence[int], num_classes: int) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes)
    weights = np.zeros(num_classes, dtype=np.float32)
    present = counts > 0
    weights[present] = len(labels) / (num_classes * counts[present])
    return torch.from_numpy(weights)


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    return F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    ) * (temperature * temperature)


@dataclass
class TrainingResult:
    epochs_run: int
    best_loss: float
    history: list[float]


def freeze_batch_norm_statistics(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


def train_classifier(
    model: nn.Module,
    loader: DataLoader,
    labels: Sequence[int],
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    focal_gamma: float,
    patience: int,
    old_model: nn.Module | None = None,
    kd_temperature: float = 2.0,
    kd_weight: float = 0.5,
) -> TrainingResult:
    if not labels:
        raise ValueError("Cannot train the classifier without selected samples")
    model.to(device)
    if old_model is not None:
        old_model.to(device).eval()
    weights = class_weights(labels, model.fc.out_features).to(device)
    criterion = FocalLoss(weights, focal_gamma)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs)
    )
    best_loss = float("inf")
    best_state = None
    stale_epochs = 0
    history: list[float] = []

    for _ in range(epochs):
        model.train()
        freeze_batch_norm_statistics(model)
        running_loss = 0.0
        sample_count = 0
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["class_index"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            if old_model is not None:
                with torch.no_grad():
                    teacher_logits = old_model(images)
                loss = loss + kd_weight * distillation_loss(
                    logits, teacher_logits, kd_temperature
                )
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach()) * images.shape[0]
            sample_count += images.shape[0]
        scheduler.step()
        epoch_loss = running_loss / max(1, sample_count)
        history.append(epoch_loss)
        if epoch_loss < best_loss - 1e-6:
            best_loss = epoch_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainingResult(len(history), best_loss, history)
