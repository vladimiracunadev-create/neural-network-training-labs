from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from neural_labs.distributed_checkpoint import load_distributed_checkpoint, save_distributed_checkpoint
from neural_labs.monitoring import PredictionLogger, drift_from_summaries, monitoring_report
from neural_labs.model_registry import LocalModelRegistry


def test_prediction_monitoring(tmp_path: Path) -> None:
    log = tmp_path / "predictions.jsonl"
    logger = PredictionLogger(log, model="iris", reference="champion")
    event = logger.log(np.asarray([[1.0, 2.0], [3.0, 4.0]]), [0, 1])
    assert event.features_summary["mean"] == pytest.approx(2.5)
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps({"mean": 2.0, "std": 1.0}))
    report = monitoring_report(log, reference)
    assert report["events"] == 1
    assert report["drift"]["status"] in {"ok", "alert"}
    assert drift_from_summaries({"mean": 0, "std": 1}, {"mean": 3, "std": 1})["status"] == "alert"
    assert monitoring_report(tmp_path / "empty.jsonl")["status"] == "no_data"


def test_registry_promotion_gates(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "best_model.pt").write_bytes(b"model")
    (run / "metrics.json").write_text(json.dumps({"accuracy": 0.91, "latency_ms_p95": 4.0}))
    (run / "experiment.lock.json").write_text(json.dumps({
        "schema_version": "1.0", "lab_id": "demo", "split_seed": 42, "training_seed": 42,
        "config_name": "baseline", "selection_metric": "accuracy", "selected_checkpoint": "best_model.pt",
        "dataset_hash": "hash", "frozen_at": "2026-07-24T00:00:00+00:00", "status": "frozen_before_test"
    }))
    registry = LocalModelRegistry(tmp_path / "registry.json")
    entry = registry.register("demo", run, alias="challenger")
    report = registry.promotion_report("demo", entry.version, metric="accuracy", minimum=0.9, max_latency_ms=5)
    assert report["passed"]
    champion = registry.promote("demo", entry.version, metric="accuracy", minimum=0.9, max_latency_ms=5)
    assert champion.alias == "champion"
    with pytest.raises(ValueError):
        registry.promote("demo", entry.version, metric="accuracy", minimum=0.99)


def test_portable_distributed_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    target = save_distributed_checkpoint(tmp_path / "checkpoint", {"tensor": torch.tensor([1, 2])})
    restored = load_distributed_checkpoint(target)
    assert restored["tensor"].tolist() == [1, 2]
