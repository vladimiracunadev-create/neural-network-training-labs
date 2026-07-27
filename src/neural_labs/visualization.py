from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .datasets import ArrayDataset, DataBundle


def _first_batch(dataset: Any, maximum: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(dataset, ArrayDataset):
        return dataset.features[:maximum].cpu(), dataset.targets[:maximum].cpu()
    loader = DataLoader(dataset, batch_size=maximum, shuffle=False, num_workers=0)
    features, targets = next(iter(loader))
    return features.cpu(), targets.cpu()


def dataset_overview(bundle: DataBundle, maximum: int = 512) -> tuple[pd.DataFrame, plt.Figure]:
    """Create a task-aware overview without fitting anything on validation or test."""
    counts = {
        "train": len(bundle.train_ids),
        "validation": len(bundle.validation_ids),
        "test": len(bundle.test_ids),
    }
    table = pd.DataFrame(
        {
            "campo": ["dataset", "tarea", "forma_entrada", "clases", "train", "validation", "test"],
            "valor": [
                bundle.dataset_name,
                bundle.task,
                str(bundle.input_shape),
                bundle.num_classes,
                counts["train"],
                counts["validation"],
                counts["test"],
            ],
        }
    )

    if bundle.task == "node_classification":
        graph = bundle.train
        figure, axis = plt.subplots(figsize=(8, 4.5))
        labels = graph.y.detach().cpu().numpy()
        values, frequencies = np.unique(labels, return_counts=True)
        axis.bar(values.astype(str), frequencies)
        axis.set_title("Distribución de clases en la red de citas")
        axis.set_xlabel("Clase")
        axis.set_ylabel("Nodos")
        figure.tight_layout()
        return table, figure

    features, targets = _first_batch(bundle.train, maximum)
    feature_array = features.detach().cpu().numpy()
    target_array = targets.detach().cpu().numpy().reshape(-1)

    if bundle.task == "reinforcement_learning":
        demand = np.asarray(bundle.raw.get("train_demand", target_array), dtype=float).reshape(-1)
        figure, axis = plt.subplots(figsize=(10, 4.5))
        axis.plot(np.arange(len(demand)), demand)
        axis.set_title("Demanda real de entrenamiento en orden cronológico")
        axis.set_xlabel("Día")
        axis.set_ylabel("Unidades")
        figure.tight_layout()
        return table, figure

    is_image = feature_array.ndim == 4 and feature_array.shape[1] in {1, 3}
    if is_image:
        sample_count = min(12, len(feature_array))
        figure, axes = plt.subplots(3, 4, figsize=(8, 6))
        for axis in axes.ravel():
            axis.axis("off")
        for axis, image, target in zip(axes.ravel(), feature_array[:sample_count], target_array[:sample_count]):
            if image.shape[0] == 1:
                axis.imshow(image[0], cmap="gray")
            else:
                image = np.moveaxis(image, 0, -1)
                minimum, maximum_value = float(image.min()), float(image.max())
                normalized = (image - minimum) / max(maximum_value - minimum, 1e-8)
                axis.imshow(np.clip(normalized, 0.0, 1.0))
            label = bundle.class_names[int(target)] if bundle.class_names and int(target) < len(bundle.class_names) else str(int(target))
            axis.set_title(label, fontsize=8)
            axis.axis("off")
        figure.suptitle("Muestras reales de train")
        figure.tight_layout()
        return table, figure

    if feature_array.ndim == 3:
        figure, axis = plt.subplots(figsize=(10, 4.5))
        sequence = feature_array[0]
        if sequence.shape[0] <= 16 and sequence.shape[1] > sequence.shape[0]:
            sequence = sequence.T
        for channel in range(min(4, sequence.shape[-1])):
            axis.plot(sequence[:, channel], label=f"canal_{channel}")
        axis.set_title("Primera secuencia real de entrenamiento")
        axis.set_xlabel("Paso temporal")
        axis.legend()
        figure.tight_layout()
        return table, figure

    figure, axis = plt.subplots(figsize=(8, 4.5))
    if bundle.task == "regression":
        axis.hist(target_array, bins=min(30, max(5, int(np.sqrt(len(target_array))))))
        axis.set_title("Distribución del objetivo en train")
        axis.set_xlabel("Objetivo")
        axis.set_ylabel("Frecuencia")
    else:
        values, frequencies = np.unique(target_array.astype(int), return_counts=True)
        labels = [
            bundle.class_names[int(value)] if bundle.class_names and int(value) < len(bundle.class_names) else str(int(value))
            for value in values
        ]
        axis.bar(labels, frequencies)
        axis.set_title("Distribución de clases en train")
        axis.set_xlabel("Clase")
        axis.set_ylabel("Muestras")
        axis.tick_params(axis="x", rotation=35)
    figure.tight_layout()
    return table, figure


def prediction_errors(run_dir: str | Path, maximum: int = 100) -> pd.DataFrame:
    path = Path(run_dir) / "predictions.csv"
    if not path.exists():
        return pd.DataFrame(columns=["sample_id", "y_true", "y_pred"])
    frame = pd.read_csv(path)
    if "y_true" not in frame or "y_pred" not in frame:
        return frame.head(0)
    errors = frame[frame["y_true"] != frame["y_pred"]].copy()
    probability_columns = [column for column in errors.columns if column.startswith("probability_")]
    if probability_columns:
        errors["confidence"] = errors[probability_columns].max(axis=1)
        errors = errors.sort_values("confidence", ascending=False)
    return errors.head(maximum).reset_index(drop=True)


def run_summary(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    output: dict[str, Any] = {"run_dir": str(root)}
    for name in ["metrics.json", "baseline_metrics.json", "environment.json", "dataset_manifest.json"]:
        path = root / name
        if path.exists():
            output[name.removesuffix(".json")] = json.loads(path.read_text(encoding="utf-8"))
    output["artifacts"] = sorted(path.name for path in root.iterdir() if path.is_file()) if root.exists() else []
    return output
