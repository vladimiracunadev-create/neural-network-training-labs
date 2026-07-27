from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import TensorDataset

from .catalog import ROOT, get_lab
from .config import merged_config
from .datasets import ArrayDataset, DataBundle, prepare_dataset
from .experiments import _train_torch_model
from .models import build_model
from .runtime import create_run_dir, get_device, save_json, seed_everything


def _combine_development_splits(bundle: DataBundle) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(bundle.train, ArrayDataset) or not isinstance(bundle.validation, ArrayDataset):
        raise TypeError("La validación cruzada integrada requiere un dataset tabular ArrayDataset.")
    x = torch.cat([bundle.train.features, bundle.validation.features]).numpy()
    y = torch.cat([bundle.train.targets, bundle.validation.targets]).numpy()
    return x, y


def cross_validate_lab(
    lab_id: str,
    *,
    folds: int = 5,
    split_seed: int = 42,
    training_seeds: list[int] | None = None,
    quick: bool = False,
    config_name: str = "baseline",
    device: str = "cpu",
    output_dir: str | Path = "reports/cross-validation",
) -> dict[str, Any]:
    lab = get_lab(lab_id)
    if lab["task"] not in {"regression", "multiclass_classification", "binary_classification"}:
        raise ValueError("La validación cruzada estándar solo está habilitada para tareas supervisadas tabulares.")
    bundle = prepare_dataset(lab_id, quick=quick, seed=split_seed)
    x, y = _combine_development_splits(bundle)
    seeds = training_seeds or [41, 42, 43]
    splitter = KFold(folds, shuffle=True, random_state=split_seed) if lab["task"] == "regression" else StratifiedKFold(folds, shuffle=True, random_state=split_seed)
    output = ROOT / output_dir / lab_id
    output.mkdir(parents=True, exist_ok=True)
    fold_results = []
    for fold, (train_indices, validation_indices) in enumerate(splitter.split(x, y if lab["task"] != "regression" else None), start=1):
        for training_seed in seeds:
            seed_everything(training_seed)
            train_ids = [f"cv-{fold}-train-{index}" for index in train_indices]
            validation_ids = [f"cv-{fold}-validation-{index}" for index in validation_indices]
            fold_bundle = DataBundle(
                lab_id=lab_id,
                dataset_name=bundle.dataset_name,
                task=bundle.task,
                train=ArrayDataset(x[train_indices], y[train_indices], train_ids),
                validation=ArrayDataset(x[validation_indices], y[validation_indices], validation_ids),
                test=ArrayDataset(x[validation_indices], y[validation_indices], validation_ids),
                input_shape=bundle.input_shape,
                num_classes=bundle.num_classes,
                class_names=bundle.class_names,
                feature_names=bundle.feature_names,
                train_ids=train_ids,
                validation_ids=validation_ids,
                test_ids=validation_ids,
                metadata=bundle.metadata,
            )
            config = merged_config(lab_id, config_name, seed=training_seed, device=device)
            config["_quick"] = quick
            fold_dir = create_run_dir(f"{lab_id}-fold{fold}-seed{training_seed}", output)
            architecture = {"numpy_mlp": "mlp", "numpy_logistic": "linear"}.get(lab["architecture"], lab["architecture"])
            model = build_model(architecture, bundle.input_shape or (1,), bundle.num_classes, {})
            _model, _history, metrics, _true, _pred, _prob = _train_torch_model(
                model,
                fold_bundle,
                config,
                get_device(device),
                fold_dir,
                evaluate_test=True,
            )
            fold_results.append({"fold": fold, "training_seed": training_seed, "metrics": metrics})
    metric = str(lab.get("selection_metric", "accuracy"))
    values = [float(item["metrics"][metric]) for item in fold_results if metric in item["metrics"]]
    summary = {
        "lab_id": lab_id,
        "folds": folds,
        "split_seed": split_seed,
        "training_seeds": seeds,
        "selection_metric": metric,
        "mean": float(np.mean(values)) if values else None,
        "std": float(np.std(values)) if values else None,
        "minimum": float(np.min(values)) if values else None,
        "maximum": float(np.max(values)) if values else None,
        "results": fold_results,
        "test_set_used": False,
    }
    save_json(output / "summary.json", summary)
    return summary


def walk_forward_windows(length: int, *, minimum_train: int, validation_size: int, step: int) -> list[tuple[slice, slice]]:
    if minimum_train <= 0 or validation_size <= 0 or step <= 0:
        raise ValueError("minimum_train, validation_size y step deben ser positivos.")
    windows = []
    train_end = minimum_train
    while train_end + validation_size <= length:
        windows.append((slice(0, train_end), slice(train_end, train_end + validation_size)))
        train_end += step
    return windows
