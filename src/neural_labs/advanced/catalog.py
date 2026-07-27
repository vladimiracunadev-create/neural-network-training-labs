from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from ..catalog import ROOT


@lru_cache(maxsize=1)
def tracks() -> list[dict[str, Any]]:
    payload = yaml.safe_load((ROOT / "configs" / "advanced_tracks.yaml").read_text(encoding="utf-8"))
    return list(payload["tracks"])


def list_tracks() -> list[str]:
    return [str(item["id"]) for item in tracks()]


def get_track(track_id: str) -> dict[str, Any]:
    for item in tracks():
        if item["id"] == track_id:
            return item
    raise KeyError(f"Laboratorio avanzado desconocido: {track_id}")
