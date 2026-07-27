from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score


def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    task: str,
    seed: int = 42,
    samples: int = 500,
    confidence: float = 0.95,
) -> dict[str, Any]:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if len(y_true) != len(y_pred) or len(y_true) < 2:
        return {"warning": "No hay suficientes observaciones para bootstrap."}
    rng = np.random.default_rng(seed)
    size = len(y_true)
    functions = (
        {
            "mae": lambda a, b: mean_absolute_error(a, b),
            "rmse": lambda a, b: mean_squared_error(a, b) ** 0.5,
            "r2": lambda a, b: r2_score(a, b),
        }
        if task == "regression"
        else {
            "accuracy": lambda a, b: accuracy_score(a, b),
            "macro_f1": lambda a, b: f1_score(a, b, average="macro", zero_division=0),
        }
    )
    distributions: dict[str, list[float]] = {name: [] for name in functions}
    for _ in range(max(50, samples)):
        indices = rng.integers(0, size, size=size)
        true_sample = y_true[indices]
        pred_sample = y_pred[indices]
        for name, function in functions.items():
            try:
                distributions[name].append(float(function(true_sample, pred_sample)))
            except ValueError:
                continue
    alpha = (1.0 - confidence) / 2.0
    result: dict[str, Any] = {"confidence": confidence, "bootstrap_samples": max(50, samples)}
    for name, values in distributions.items():
        if not values:
            continue
        result[name] = {
            "estimate": float(functions[name](y_true, y_pred)),
            "lower": float(np.quantile(values, alpha)),
            "upper": float(np.quantile(values, 1.0 - alpha)),
        }
    return result
