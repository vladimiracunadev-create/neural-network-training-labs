from __future__ import annotations

import torch
from torch import nn


class SimCLREncoder(nn.Module):
    def __init__(self, representation_dim: int = 128, projection_dim: int = 64):
        super().__init__()
        try:
            from torchvision.models import resnet18
        except ImportError as exc:
            raise RuntimeError('Instale el extra vision: pip install -e ".[vision]"') from exc
        backbone = resnet18(weights=None)
        features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.representation = nn.Linear(features, representation_dim)
        self.projector = nn.Sequential(
            nn.Linear(representation_dim, representation_dim), nn.ReLU(), nn.Linear(representation_dim, projection_dim)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.representation(self.backbone(x))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        representation = self.encode(x)
        projection = nn.functional.normalize(self.projector(representation), dim=1)
        return representation, projection


def nt_xent_loss(first: torch.Tensor, second: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    if first.shape != second.shape:
        raise ValueError("Las dos vistas deben producir embeddings con la misma forma.")
    batch = first.shape[0]
    embeddings = torch.cat([first, second], dim=0)
    similarity = embeddings @ embeddings.T / temperature
    mask = torch.eye(2 * batch, dtype=torch.bool, device=embeddings.device)
    similarity = similarity.masked_fill(mask, -torch.inf)
    targets = torch.cat([torch.arange(batch, 2 * batch), torch.arange(0, batch)]).to(embeddings.device)
    return nn.functional.cross_entropy(similarity, targets)
