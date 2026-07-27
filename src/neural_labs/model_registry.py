from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .core.protocol import assert_lock_matches
from .runtime import save_json


@dataclass
class RegistryEntry:
    name: str
    version: int
    run_dir: str
    alias: str | None
    created_at: str
    metrics: dict[str, Any]


class LocalModelRegistry:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            save_json(self.path, {"models": {}})

    def _read(self) -> dict[str, Any]:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _write(self, payload: dict[str, Any]) -> None:
        save_json(self.path, payload)

    def register(self, name: str, run_dir: Path, *, alias: str | None = None) -> RegistryEntry:
        run_dir = Path(run_dir).resolve()
        assert_lock_matches(run_dir)
        metrics_path = run_dir / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
        payload = self._read()
        versions = payload["models"].setdefault(name, [])
        entry = RegistryEntry(
            name=name,
            version=len(versions) + 1,
            run_dir=str(run_dir),
            alias=alias,
            created_at=datetime.now(timezone.utc).isoformat(),
            metrics=metrics,
        )
        if alias:
            for existing in versions:
                if existing.get("alias") == alias:
                    existing["alias"] = None
        versions.append(entry.__dict__)
        self._write(payload)
        return entry

    def set_alias(self, name: str, version: int, alias: str) -> RegistryEntry:
        payload = self._read()
        versions = payload["models"].get(name, [])
        target = None
        for entry in versions:
            if entry.get("alias") == alias:
                entry["alias"] = None
            if int(entry["version"]) == int(version):
                target = entry
        if target is None:
            raise KeyError(f"No existe {name} versión {version}.")
        target["alias"] = alias
        self._write(payload)
        return RegistryEntry(**target)

    def resolve(self, name: str, reference: str = "champion") -> RegistryEntry:
        versions = self._read()["models"].get(name, [])
        if not versions:
            raise KeyError(f"No hay modelos registrados con nombre {name!r}.")
        if reference.isdigit():
            for entry in versions:
                if int(entry["version"]) == int(reference):
                    return RegistryEntry(**entry)
        else:
            for entry in versions:
                if entry.get("alias") == reference:
                    return RegistryEntry(**entry)
        raise KeyError(f"No existe la referencia {name}@{reference}.")


    def promotion_report(
        self,
        name: str,
        version: int,
        *,
        metric: str,
        minimum: float | None = None,
        maximum: float | None = None,
        max_latency_ms: float | None = None,
    ) -> dict[str, Any]:
        entry = self.resolve(name, str(version))
        value = entry.metrics.get(metric)
        checks: dict[str, bool] = {"metric_present": isinstance(value, (int, float))}
        if minimum is not None:
            checks["minimum"] = checks["metric_present"] and float(value) >= minimum
        if maximum is not None:
            checks["maximum"] = checks["metric_present"] and float(value) <= maximum
        if max_latency_ms is not None:
            latency = entry.metrics.get("latency_ms_p95", entry.metrics.get("latency_ms"))
            checks["latency"] = isinstance(latency, (int, float)) and float(latency) <= max_latency_ms
        return {"entry": entry.__dict__, "metric": metric, "value": value, "checks": checks, "passed": all(checks.values())}

    def promote(
        self,
        name: str,
        version: int,
        *,
        alias: str = "champion",
        metric: str,
        minimum: float | None = None,
        maximum: float | None = None,
        max_latency_ms: float | None = None,
    ) -> RegistryEntry:
        report = self.promotion_report(
            name, version, metric=metric, minimum=minimum, maximum=maximum, max_latency_ms=max_latency_ms
        )
        if not report["passed"]:
            raise ValueError(f"El candidato no cumple las puertas de promoción: {report['checks']}")
        return self.set_alias(name, version, alias)

    def list(self) -> dict[str, Any]:
        return self._read()
