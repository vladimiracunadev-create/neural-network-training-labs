from __future__ import annotations

from typing import Any

import numpy as np

from .datasets import ArrayDataset, DataBundle


def _numeric_matrix(dataset: Any) -> np.ndarray | None:
    if not isinstance(dataset, ArrayDataset):
        return None
    values = dataset.features.detach().cpu().numpy()
    if not np.issubdtype(values.dtype, np.floating):
        return None
    if values.ndim != 2 or values.shape[1] > 512:
        return None
    return values.astype(np.float64, copy=False)


def _standardized_mean_difference(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    reference_mean = np.nanmean(reference, axis=0)
    candidate_mean = np.nanmean(candidate, axis=0)
    reference_var = np.nanvar(reference, axis=0, ddof=1)
    candidate_var = np.nanvar(candidate, axis=0, ddof=1)
    pooled = np.sqrt((reference_var + candidate_var) / 2.0)
    return np.divide(candidate_mean - reference_mean, pooled, out=np.zeros_like(pooled), where=pooled > 1e-12)


def drift_report(bundle: DataBundle, *, warning_threshold: float = 0.2) -> dict[str, Any]:
    train = _numeric_matrix(bundle.train)
    validation = _numeric_matrix(bundle.validation)
    test = _numeric_matrix(bundle.test)
    if train is None or validation is None or test is None:
        return {
            "available": False,
            "reason": "El reporte automático se limita a matrices tabulares float con hasta 512 características.",
        }
    names = bundle.feature_names or [f"feature_{index}" for index in range(train.shape[1])]
    comparisons: dict[str, Any] = {}
    for split_name, candidate in (("validation", validation), ("test", test)):
        smd = _standardized_mean_difference(train, candidate)
        order = np.argsort(np.abs(smd))[::-1]
        top = [
            {
                "feature": str(names[index]) if index < len(names) else f"feature_{index}",
                "standardized_mean_difference": float(smd[index]),
                "absolute_smd": float(abs(smd[index])),
            }
            for index in order[: min(20, len(order))]
        ]
        comparisons[split_name] = {
            "max_absolute_smd": float(np.max(np.abs(smd))) if smd.size else 0.0,
            "mean_absolute_smd": float(np.mean(np.abs(smd))) if smd.size else 0.0,
            "features_over_threshold": int(np.sum(np.abs(smd) >= warning_threshold)),
            "top_features": top,
        }
    return {
        "available": True,
        "method": "standardized_mean_difference",
        "warning_threshold": warning_threshold,
        "interpretation": "SMD alto indica cambio de distribución, no necesariamente fuga ni error.",
        "comparisons": comparisons,
    }
