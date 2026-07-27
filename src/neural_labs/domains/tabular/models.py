from __future__ import annotations

from typing import Iterable

from torch import nn

from ...core.registry import MODEL_REGISTRY


class LinearClassifier(nn.Module):
    def __init__(self, input_dim: int, classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1 if classes == 2 else classes)

    def forward(self, x):
        output = self.linear(x)
        return output.squeeze(-1) if output.shape[-1] == 1 else output


class TabularMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        classes: int | None,
        hidden: Iterable[int] = (128, 64),
        dropout: float = 0.2,
        activation: str = "relu",
        regression: bool = False,
        batch_norm: bool = False,
    ):
        super().__init__()
        activations = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh, "silu": nn.SiLU}
        activation_cls = activations.get(activation, nn.ReLU)
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden:
            layers.append(nn.Linear(previous, int(width)))
            if batch_norm:
                layers.append(nn.BatchNorm1d(int(width)))
            layers.append(activation_cls())
            if dropout:
                layers.append(nn.Dropout(float(dropout)))
            previous = int(width)
        output_dim = 1 if regression or classes == 2 else int(classes or 1)
        layers.append(nn.Linear(previous, output_dim))
        self.network = nn.Sequential(*layers)
        self.squeeze = output_dim == 1

    def forward(self, x):
        output = self.network(x)
        return output.squeeze(-1) if self.squeeze else output


@MODEL_REGISTRY.register("linear", domain="tabular", description="Clasificador lineal PyTorch")
def create_linear(*, input_shape, num_classes, metadata=None, **kwargs):
    del metadata, kwargs
    return LinearClassifier(int(input_shape[-1]), int(num_classes or 2))


for _name in (
    "mlp",
    "mlp_optuna",
    "mlp_explainability",
    "mlp_calibration",
    "capstone_mlp",
    "activation_comparison",
    "optimizer_comparison",
    "regularization_comparison",
    "fedavg_mlp",
):
    MODEL_REGISTRY.register(_name, domain="tabular", description="MLP configurable para datos tabulares")(
        lambda *, input_shape, num_classes, metadata=None, **kwargs: TabularMLP(
            int(input_shape[-1]),
            num_classes,
            hidden=kwargs.get("hidden", (128, 64)),
            dropout=float(kwargs.get("dropout", 0.2)),
            activation=str(kwargs.get("activation", "relu")),
            regression=bool(kwargs.get("regression", False)),
            batch_norm=bool(kwargs.get("batch_norm", False)),
        )
    )
