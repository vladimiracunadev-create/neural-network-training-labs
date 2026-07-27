from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np


@dataclass
class PredictionEvent:
    timestamp: str
    model: str
    reference: str
    features_summary: dict[str, float]
    predictions: list[Any]


class PredictionLogger:
    """Append-only JSONL logger with no raw-feature retention by default."""

    def __init__(self, path: Path, *, model: str = "default", reference: str = "champion"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.reference = reference
        self._lock = Lock()

    @staticmethod
    def summarize(features: Any) -> dict[str, float]:
        array = np.asarray(features, dtype=np.float64)
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            return {"count": 0.0, "mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan}
        return {
            "count": float(finite.size),
            "mean": float(finite.mean()),
            "std": float(finite.std()),
            "min": float(finite.min()),
            "max": float(finite.max()),
        }

    def log(self, features: Any, predictions: list[Any]) -> PredictionEvent:
        event = PredictionEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            model=self.model,
            reference=self.reference,
            features_summary=self.summarize(features),
            predictions=predictions,
        )
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(event), ensure_ascii=False, allow_nan=True) + "\n")
        return event

    def read(self, limit: int = 1000) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()[-limit:]
        return [json.loads(line) for line in lines if line.strip()]


def drift_from_summaries(reference: dict[str, float], current: dict[str, float]) -> dict[str, Any]:
    ref_std = max(abs(float(reference.get("std", 0.0))), 1e-8)
    mean_shift = abs(float(current.get("mean", 0.0)) - float(reference.get("mean", 0.0))) / ref_std
    std_ratio = float(current.get("std", 0.0)) / ref_std
    status = "alert" if mean_shift >= 2.0 or std_ratio >= 2.0 or std_ratio <= 0.5 else "ok"
    return {"status": status, "mean_shift_in_reference_std": mean_shift, "std_ratio": std_ratio}


def monitoring_report(log_path: Path, reference_path: Path | None = None, *, limit: int = 1000) -> dict[str, Any]:
    logger = PredictionLogger(log_path)
    events = logger.read(limit=limit)
    if not events:
        return {"events": 0, "status": "no_data", "current": None, "drift": None}
    summaries = [event["features_summary"] for event in events]
    weighted_count = sum(float(item.get("count", 0.0)) for item in summaries)
    current = {
        "count": weighted_count,
        "mean": sum(float(item.get("mean", 0.0)) * float(item.get("count", 0.0)) for item in summaries) / max(weighted_count, 1.0),
        "std": sum(float(item.get("std", 0.0)) for item in summaries) / max(len(summaries), 1),
    }
    reference = None
    if reference_path and Path(reference_path).is_file():
        reference = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    return {
        "events": len(events),
        "status": "ok" if reference is None else drift_from_summaries(reference, current)["status"],
        "current": current,
        "reference": reference,
        "drift": drift_from_summaries(reference, current) if reference else None,
    }
