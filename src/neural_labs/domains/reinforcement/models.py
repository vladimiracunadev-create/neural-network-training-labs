from __future__ import annotations

import torch
from torch import nn

from ...core.registry import MODEL_REGISTRY


class DQN(nn.Module):
    def __init__(self, state_dim: int = 4, actions: int = 2):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, actions))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DuelingDQN(nn.Module):
    def __init__(self, state_dim: int = 4, actions: int = 2):
        super().__init__()
        self.features = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.value = nn.Linear(128, 1)
        self.advantage = nn.Linear(128, actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        value = self.value(features)
        advantage = self.advantage(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


@MODEL_REGISTRY.register("dqn_inventory", domain="reinforcement", description="Double Dueling DQN para inventario")
def create_inventory_dqn(*, input_shape, num_classes, metadata=None, **kwargs):
    del input_shape, num_classes, metadata
    return DuelingDQN(state_dim=int(kwargs.get("state_dim", 4)), actions=int(kwargs.get("actions", 4)))
