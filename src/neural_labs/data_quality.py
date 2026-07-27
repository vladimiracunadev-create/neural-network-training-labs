from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import torch

from .datasets import ArrayDataset, DataBundle
from .runtime import sha256_strings


def _dataset_stats(dataset: Any, ids: list[str], task: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "samples": len(dataset),
        "id_fingerprint": sha256_strings(ids),
        "duplicate_ids": len(ids) - len(set(ids)),
    }
    if isinstance(dataset, ArrayDataset):
        features = dataset.features.detach().cpu().numpy()
        targets = dataset.targets.detach().cpu().numpy().reshape(-1)
        result.update(
            {
                "feature_shape": list(features.shape),
                "feature_dtype": str(features.dtype),
                "feature_nan_count": int(np.isnan(features).sum()) if np.issubdtype(features.dtype, np.floating) else 0,
                "feature_inf_count": int(np.isinf(features).sum()) if np.issubdtype(features.dtype, np.floating) else 0,
                "feature_min": float(np.nanmin(features)) if features.size else None,
                "feature_max": float(np.nanmax(features)) if features.size else None,
                "target_nan_count": int(np.isnan(targets).sum()) if np.issubdtype(targets.dtype, np.floating) else 0,
            }
        )
        if task != "regression":
            result["class_distribution"] = {str(key): int(value) for key, value in sorted(Counter(targets.tolist()).items())}
        else:
            result["target_mean"] = float(np.mean(targets))
            result["target_std"] = float(np.std(targets))
    else:
        try:
            first = dataset[0]
            features = first[0] if isinstance(first, (tuple, list)) else first
            if isinstance(features, torch.Tensor):
                result["sample_shape"] = list(features.shape)
                result["sample_dtype"] = str(features.dtype)
        except Exception as exc:
            result["inspection_warning"] = str(exc)
    return result


def quality_report(bundle: DataBundle) -> dict[str, Any]:
    train_ids = set(bundle.train_ids)
    validation_ids = set(bundle.validation_ids)
    test_ids = set(bundle.test_ids)
    overlaps = {
        "train_validation": len(train_ids.intersection(validation_ids)),
        "train_test": len(train_ids.intersection(test_ids)),
        "validation_test": len(validation_ids.intersection(test_ids)),
    }
    return {
        "lab_id": bundle.lab_id,
        "dataset": bundle.dataset_name,
        "task": bundle.task,
        "splits": {
            "train": _dataset_stats(bundle.train, bundle.train_ids, bundle.task),
            "validation": _dataset_stats(bundle.validation, bundle.validation_ids, bundle.task),
            "test": _dataset_stats(bundle.test, bundle.test_ids, bundle.task),
        },
        "overlaps": overlaps,
        "passed": all(value == 0 for value in overlaps.values()),
    }
