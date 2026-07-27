from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .catalog import ROOT


def load_config(lab_id: str, name: str = "baseline") -> dict[str, Any]:
    path = ROOT / "labs" / lab_id / "configs" / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"No existe la configuración {name!r} para {lab_id}: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def merged_config(lab_id: str, name: str = "baseline", **overrides: Any) -> dict[str, Any]:
    config = load_config(lab_id, name)
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    return config
