from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SeedPlan:
    """Separates data partitioning randomness from training randomness."""

    split_seed: int = 42
    training_seed: int = 42

    @classmethod
    def resolve(
        cls,
        *,
        split_seed: int | None = None,
        training_seed: int | None = None,
        legacy_seed: int | None = None,
    ) -> "SeedPlan":
        fallback = 42 if legacy_seed is None else int(legacy_seed)
        return cls(
            split_seed=fallback if split_seed is None else int(split_seed),
            training_seed=fallback if training_seed is None else int(training_seed),
        )


@dataclass
class ExperimentLock:
    lab_id: str
    split_seed: int
    training_seed: int
    config_name: str
    selection_metric: str
    selected_checkpoint: str
    dataset_hash: str
    frozen_at: str
    status: str = "frozen_before_test"

    @classmethod
    def create(
        cls,
        *,
        lab_id: str,
        seeds: SeedPlan,
        config_name: str,
        selection_metric: str,
        selected_checkpoint: Path,
        dataset_hash: str,
    ) -> "ExperimentLock":
        return cls(
            lab_id=lab_id,
            split_seed=seeds.split_seed,
            training_seed=seeds.training_seed,
            config_name=config_name,
            selection_metric=selection_metric,
            selected_checkpoint=selected_checkpoint.name,
            dataset_hash=dataset_hash,
            frozen_at=datetime.now(timezone.utc).isoformat(),
        )

    def write(self, run_dir: Path) -> Path:
        path = run_dir / "experiment.lock.json"
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")
        return path


def stable_payload_hash(payload: Any) -> str:
    normalized = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def assert_lock_matches(run_dir: Path, *, lab_id: str | None = None) -> dict[str, Any]:
    path = run_dir / "experiment.lock.json"
    if not path.is_file():
        raise FileNotFoundError(f"La ejecución no está sellada: falta {path.name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "frozen_before_test":
        raise ValueError("El experimento no fue congelado antes de evaluar test.")
    if lab_id is not None and payload.get("lab_id") != lab_id:
        raise ValueError(f"El lock pertenece a {payload.get('lab_id')!r}, no a {lab_id!r}.")
    return payload
