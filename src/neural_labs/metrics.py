from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    auc,
)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray | None = None) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    binary = len(np.unique(y_true)) == 2
    average = "binary" if binary else "macro"
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision" if binary else "macro_precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall" if binary else "macro_recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1" if binary else "macro_f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }
    if probabilities is not None and binary:
        scores = probabilities[:, 1] if probabilities.ndim == 2 else probabilities.reshape(-1)
        metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
        precision, recall, _ = precision_recall_curve(y_true, scores)
        metrics["pr_auc"] = float(auc(recall, precision))
        metrics["brier"] = float(brier_score_loss(y_true, scores))
    return metrics


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    nonzero = np.abs(y_true) > 1e-8
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mape": float(mean_absolute_percentage_error(y_true[nonzero], y_pred[nonzero])) if nonzero.any() else float("nan"),
        "r2": float(r2_score(y_true, y_pred)),
    }


def expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    y_true = np.asarray(y_true)
    probs = np.asarray(probabilities)
    if probs.ndim == 1:
        confidence = np.maximum(probs, 1 - probs)
        predictions = (probs >= 0.5).astype(int)
    else:
        confidence = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
    correct = predictions == y_true
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            value += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(value)


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, path: Path, labels: list[str] | None = None) -> None:
    matrix = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix)
    fig.colorbar(image, ax=ax)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Valor real")
    ax.set_title("Matriz de confusión")
    if labels and len(labels) == matrix.shape[0]:
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)), labels)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_history(history: dict[str, list[float]], csv_path: Path, image_path: Path) -> None:
    import pandas as pd

    frame = pd.DataFrame(history)
    frame.to_csv(csv_path, index=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    for column in frame.columns:
        ax.plot(frame.index + 1, frame[column], label=column)
    ax.set_xlabel("Época")
    ax.set_title("Historial de entrenamiento")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(image_path, dpi=160)
    plt.close(fig)
