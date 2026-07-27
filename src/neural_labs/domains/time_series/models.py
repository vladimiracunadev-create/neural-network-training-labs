from __future__ import annotations

import torch
from torch import nn

from ...core.registry import MODEL_REGISTRY


class LSTMRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 96):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, dropout=0.15, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        output, _ = self.lstm(x)
        return self.head(output[:, -1]).squeeze(-1)


class TemporalConvNet(nn.Module):
    def __init__(self, input_dim: int, channels: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(input_dim, channels, 3, padding=2, dilation=1),
            nn.ReLU(),
            nn.Conv1d(channels, channels, 3, padding=4, dilation=2),
            nn.ReLU(),
            nn.Conv1d(channels, channels, 3, padding=8, dilation=4),
            nn.ReLU(),
        )
        self.head = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        features = self.network(x.transpose(1, 2))[..., : x.shape[1]]
        return self.head(features[:, :, -1]).squeeze(-1)


@MODEL_REGISTRY.register("lstm_regression", domain="time_series", description="LSTM para pronóstico temporal")
def create_lstm(*, input_shape, num_classes, metadata=None, **kwargs):
    del num_classes, metadata
    return LSTMRegressor(int(input_shape[-1]), hidden_dim=int(kwargs.get("hidden_dim", 96)))


@MODEL_REGISTRY.register("tcn_regression", domain="time_series", description="Red convolucional temporal dilatada")
def create_tcn(*, input_shape, num_classes, metadata=None, **kwargs):
    del num_classes, metadata
    return TemporalConvNet(int(input_shape[-1]), channels=int(kwargs.get("channels", 64)))
