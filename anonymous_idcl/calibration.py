from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class TemperatureScaler(nn.Module):
    """Single-parameter confidence calibration for classification logits."""

    def __init__(self, initial_temperature: float = 1.0):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(initial_temperature)))

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp_min(1e-6)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    max_iter: int = 50,
) -> TemperatureScaler:
    scaler = TemperatureScaler().to(logits.device)
    optimizer = torch.optim.LBFGS([scaler.log_temperature], lr=0.1, max_iter=max_iter)

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(scaler(logits), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return scaler
