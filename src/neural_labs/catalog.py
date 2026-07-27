from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def _labs() -> dict[str, dict[str, Any]]:
    payload = yaml.safe_load((ROOT / "configs" / "labs.yaml").read_text(encoding="utf-8"))
    return {item["id"]: item for item in payload["labs"]}


@lru_cache(maxsize=1)
def _datasets() -> dict[str, dict[str, Any]]:
    payload = yaml.safe_load((ROOT / "configs" / "datasets.yaml").read_text(encoding="utf-8"))
    return payload["datasets"]


def list_labs() -> list[str]:
    return sorted(_labs())


def get_lab(lab_id: str) -> dict[str, Any]:
    try:
        return dict(_labs()[lab_id])
    except KeyError as exc:
        raise KeyError(f"Laboratorio desconocido: {lab_id}") from exc


def get_dataset(name_or_lab: str) -> dict[str, Any]:
    dataset_name = get_lab(name_or_lab)["dataset"] if name_or_lab in _labs() else name_or_lab
    try:
        return dict(_datasets()[dataset_name])
    except KeyError as exc:
        raise KeyError(f"Dataset desconocido: {dataset_name}") from exc
