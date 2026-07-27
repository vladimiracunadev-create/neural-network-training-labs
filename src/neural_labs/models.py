"""Compatibility facade for the domain model registry.

New code should use :mod:`neural_labs.core.registry` and domain modules directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

# Importing domain modules performs explicit model registrations.
from .domains.tabular.models import LinearClassifier, TabularMLP  # noqa: F401
from .domains.text.models import InspectableTextTransformer as TextTransformer
from .domains.text.models import MaskedTextRNN as TextRNN
from .domains.time_series.models import LSTMRegressor
from .domains.vision.models import ConvClassifier as SmallCNN
from .domains.vision.models import DCGANDiscriminator as Discriminator
from .domains.vision.models import DCGANGenerator as Generator
from .domains.reinforcement.models import DQN, DuelingDQN
from .core.registry import MODEL_REGISTRY, ensure_module


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 12):
        super().__init__()
        hidden = max(32, input_dim // 2)
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, latent_dim), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden), nn.ReLU(), nn.Linear(hidden, input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class SensorFusionNet(nn.Module):
    def __init__(self, classes: int):
        super().__init__()
        self.accelerometer = nn.Sequential(
            nn.Conv1d(6, 32, 5, padding=2), nn.ReLU(),
            nn.Conv1d(32, 48, 5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.gyroscope = nn.Sequential(
            nn.Conv1d(3, 24, 5, padding=2), nn.ReLU(),
            nn.Conv1d(24, 32, 5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Linear(80, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        accelerometer = torch.cat([x[:, 0:3], x[:, 6:9]], dim=1)
        gyroscope = x[:, 3:6]
        acc_features = self.accelerometer(accelerometer).squeeze(-1)
        gyro_features = self.gyroscope(gyroscope).squeeze(-1)
        return self.head(torch.cat([acc_features, gyro_features], dim=1))


@MODEL_REGISTRY.register("autoencoder", domain="anomaly", description="Autoencoder denso para anomalías")
def _create_autoencoder(*, input_shape, num_classes, metadata=None, **kwargs):
    del num_classes, metadata
    input_dim = int(torch.tensor(input_shape).prod().item())
    return Autoencoder(input_dim, latent_dim=int(kwargs.get("latent_dim", 12)))


@MODEL_REGISTRY.register("sensor_fusion", domain="multimodal", description="Fusión de acelerómetro y giroscopio")
def _create_sensor_fusion(*, input_shape, num_classes, metadata=None, **kwargs):
    del input_shape, metadata, kwargs
    return SensorFusionNet(int(num_classes or 2))


@dataclass
class ModelSpec:
    architecture: str
    input_shape: tuple[int, ...]
    num_classes: int | None
    kwargs: dict[str, Any]


def build_model(
    architecture: str,
    input_shape: tuple[int, ...],
    num_classes: int | None,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> nn.Module:
    aliases = {
        "distillation_cnn": "distillation_student",
    }
    resolved = aliases.get(architecture, architecture)
    model = MODEL_REGISTRY.create(
        resolved,
        input_shape=tuple(input_shape),
        num_classes=num_classes,
        metadata=metadata or {},
        **kwargs,
    )
    return ensure_module(model)


__all__ = [
    "Autoencoder",
    "DQN",
    "DuelingDQN",
    "Discriminator",
    "Generator",
    "LinearClassifier",
    "LSTMRegressor",
    "ModelSpec",
    "SensorFusionNet",
    "SmallCNN",
    "TabularMLP",
    "TextRNN",
    "TextTransformer",
    "build_model",
]
