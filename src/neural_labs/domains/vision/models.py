from __future__ import annotations

import torch
from torch import nn

from ...core.registry import MODEL_REGISTRY


class ConvClassifier(nn.Module):
    def __init__(self, channels: int, classes: int, dropout: float = 0.2, batch_norm: bool = True, width: int = 32):
        super().__init__()

        def block(inp: int, out: int) -> list[nn.Module]:
            layers: list[nn.Module] = [nn.Conv2d(inp, out, 3, padding=1, bias=not batch_norm)]
            if batch_norm:
                layers.append(nn.BatchNorm2d(out))
            layers.extend([nn.ReLU(inplace=True), nn.MaxPool2d(2)])
            return layers

        self.features = nn.Sequential(
            *block(channels, width),
            *block(width, width * 2),
            *block(width * 2, width * 4),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(width * 4, classes))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))


class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim: int = 64, channels: int = 1, base: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.network = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, base * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(base * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(base * 2, base, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(True),
            nn.ConvTranspose2d(base, channels, 4, 2, 3, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim == 2:
            z = z[:, :, None, None]
        return self.network(z)


class DCGANDiscriminator(nn.Module):
    def __init__(self, channels: int = 1, base: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(channels, base, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base, base * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 2, base * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base * 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class MobileStudent(nn.Module):
    def __init__(self, channels: int, classes: int):
        super().__init__()

        def depthwise(inp: int, out: int, stride: int = 1):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride=stride, padding=1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU6(inplace=True),
                nn.Conv2d(inp, out, 1, bias=False),
                nn.BatchNorm2d(out),
                nn.ReLU6(inplace=True),
            )

        self.features = nn.Sequential(
            nn.Conv2d(channels, 24, 3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU6(inplace=True),
            depthwise(24, 48, 2),
            depthwise(48, 96, 2),
            depthwise(96, 128, 2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(128, classes))

    def forward(self, x):
        return self.head(self.features(x))


class TeacherCNN(ConvClassifier):
    def __init__(self, channels: int, classes: int):
        super().__init__(channels, classes, dropout=0.25, batch_norm=True, width=64)


for _name in ("cnn", "cnn_export", "augmentation_comparison"):
    MODEL_REGISTRY.register(_name, domain="vision", description="CNN convolucional para clasificación")(
        lambda *, input_shape, num_classes, metadata=None, **kwargs: ConvClassifier(
            int(input_shape[0]),
            int(num_classes or 2),
            dropout=float(kwargs.get("dropout", 0.2)),
            batch_norm=bool(kwargs.get("batch_norm", True)),
            width=int(kwargs.get("width", 32)),
        )
    )


@MODEL_REGISTRY.register("distillation_student", domain="vision", description="Estudiante móvil con convoluciones depthwise")
def create_student(*, input_shape, num_classes, metadata=None, **kwargs):
    del metadata, kwargs
    return MobileStudent(int(input_shape[0]), int(num_classes or 2))


@MODEL_REGISTRY.register("distillation_teacher", domain="vision", description="Profesor CNN de mayor capacidad")
def create_teacher(*, input_shape, num_classes, metadata=None, **kwargs):
    del metadata, kwargs
    return TeacherCNN(int(input_shape[0]), int(num_classes or 2))


@MODEL_REGISTRY.register("transfer_resnet18", domain="vision", description="ResNet-18 para fine-tuning y transferencia")
def create_transfer_resnet18(*, input_shape, num_classes, metadata=None, **kwargs):
    del input_shape, metadata, kwargs
    try:
        from torchvision.models import resnet18
    except ImportError as exc:
        raise RuntimeError('Instale el extra vision: pip install -e ".[vision]"') from exc
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, int(num_classes or 2))
    return model
