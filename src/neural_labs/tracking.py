from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


class Tracker(Protocol):
    def start(self, *, run_name: str, experiment_name: str, tags: dict[str, str] | None = None) -> None: ...
    def log_params(self, params: dict[str, Any]) -> None: ...
    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None: ...
    def log_artifact(self, path: Path) -> None: ...
    def finish(self, status: str = "FINISHED") -> None: ...


@dataclass
class JsonlTracker:
    path: Path
    _started: bool = field(default=False, init=False)

    def _write(self, event: str, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"event": event, **payload}, ensure_ascii=False, default=str) + "\n")

    def start(self, *, run_name: str, experiment_name: str, tags: dict[str, str] | None = None) -> None:
        self._started = True
        self._write("start", {"run_name": run_name, "experiment_name": experiment_name, "tags": tags or {}})

    def log_params(self, params: dict[str, Any]) -> None:
        self._write("params", {"params": params})

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        numeric = {key: value for key, value in metrics.items() if isinstance(value, (int, float)) and not isinstance(value, bool)}
        self._write("metrics", {"metrics": numeric, "step": step})

    def log_artifact(self, path: Path) -> None:
        self._write("artifact", {"path": str(path)})

    def finish(self, status: str = "FINISHED") -> None:
        if self._started:
            self._write("finish", {"status": status})


@dataclass
class MLflowTracker:
    tracking_uri: str | None = None
    _mlflow: Any = field(default=None, init=False)
    _active: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        try:
            import mlflow
        except ImportError as exc:
            raise RuntimeError('MLflow no está instalado. Use: pip install -e ".[mlops]"') from exc
        self._mlflow = mlflow
        uri = self.tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
        if uri:
            mlflow.set_tracking_uri(uri)

    def start(self, *, run_name: str, experiment_name: str, tags: dict[str, str] | None = None) -> None:
        self._mlflow.set_experiment(experiment_name)
        self._mlflow.start_run(run_name=run_name, tags=tags or {})
        self._active = True

    def log_params(self, params: dict[str, Any]) -> None:
        flattened: dict[str, Any] = {}
        for key, value in params.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                flattened[key] = value
            else:
                flattened[key] = json.dumps(value, ensure_ascii=False, default=str)[:500]
        self._mlflow.log_params(flattened)

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        numeric = {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float)) and not isinstance(value, bool)}
        if numeric:
            self._mlflow.log_metrics(numeric, step=step)

    def log_artifact(self, path: Path) -> None:
        if path.is_file():
            self._mlflow.log_artifact(str(path))

    def finish(self, status: str = "FINISHED") -> None:
        if self._active:
            self._mlflow.end_run(status=status)
            self._active = False


@dataclass
class CompositeTracker:
    trackers: list[Tracker]

    def start(self, **kwargs: Any) -> None:
        for tracker in self.trackers:
            tracker.start(**kwargs)

    def log_params(self, params: dict[str, Any]) -> None:
        for tracker in self.trackers:
            tracker.log_params(params)

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        for tracker in self.trackers:
            tracker.log_metrics(metrics, step)

    def log_artifact(self, path: Path) -> None:
        for tracker in self.trackers:
            tracker.log_artifact(path)

    def finish(self, status: str = "FINISHED") -> None:
        for tracker in self.trackers:
            tracker.finish(status)


def create_tracker(kind: str, run_dir: Path) -> Tracker:
    normalized = kind.lower()
    json_tracker = JsonlTracker(run_dir / "tracking.jsonl")
    if normalized in {"json", "jsonl"}:
        return json_tracker
    if normalized == "mlflow":
        return CompositeTracker([json_tracker, MLflowTracker()])
    if normalized == "none":
        return CompositeTracker([])
    raise ValueError(f"Tracker no soportado: {kind}")
