from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error

from .datasets import DataBundle


SENSITIVE_CANDIDATES = ("sex", "gender", "race", "age", "country", "region")


def _groups(series: pd.Series, column: str) -> pd.Series:
    if column.lower() == "age":
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().sum() >= 20:
            try:
                return pd.qcut(numeric, q=4, duplicates="drop").astype(str).fillna("<missing>")
            except ValueError:
                pass
    return series.astype(str).fillna("<missing>")


def _classification_group_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    unique = np.unique(y_true)
    if set(unique.tolist()).issubset({0, 1}) and len(unique) == 2:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metrics.update(
            {
                "selection_rate": float(np.mean(y_pred == 1)),
                "true_positive_rate": float(tp / max(tp + fn, 1)),
                "false_positive_rate": float(fp / max(fp + tn, 1)),
            }
        )
    return metrics


def subgroup_report(
    bundle: DataBundle,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    minimum_size: int = 20,
) -> dict[str, Any]:
    raw_test = bundle.raw.get("x_test") if bundle.raw else None
    if not isinstance(raw_test, pd.DataFrame) or len(raw_test) != len(y_true):
        return {"available": False, "reason": "No hay atributos tabulares crudos alineados con el conjunto de test."}
    columns = [column for column in raw_test.columns if str(column).lower() in SENSITIVE_CANDIDATES]
    if not columns:
        return {"available": False, "reason": "El dataset no expone columnas de segmentación reconocidas."}
    report: dict[str, Any] = {
        "available": True,
        "minimum_group_size": minimum_size,
        "warning": "Es un diagnóstico descriptivo; no demuestra equidad ni causalidad.",
        "columns": {},
    }
    for column in columns:
        groups: dict[str, Any] = {}
        grouped = _groups(raw_test[column], str(column))
        for value, positions in grouped.groupby(grouped, observed=False).indices.items():
            positions = np.asarray(positions, dtype=int)
            if len(positions) < minimum_size:
                continue
            true_group = np.asarray(y_true)[positions]
            pred_group = np.asarray(y_pred)[positions]
            if bundle.task == "regression":
                metrics = {"mae": float(mean_absolute_error(true_group, pred_group))}
            else:
                metrics = _classification_group_metrics(true_group, pred_group)
            groups[str(value)] = {"samples": int(len(positions)), **metrics}
        disparities: dict[str, float] = {}
        metric_names = sorted({key for values in groups.values() for key in values if key != "samples"})
        for metric in metric_names:
            values = [float(group[metric]) for group in groups.values() if metric in group]
            if len(values) >= 2:
                disparities[f"{metric}_max_minus_min"] = max(values) - min(values)
        report["columns"][str(column)] = {"groups": groups, "disparities": disparities}
    return report
