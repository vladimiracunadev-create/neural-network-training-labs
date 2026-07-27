from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_distributed_checkpoint(path: Path, state: dict[str, Any]) -> Path:
    """Use torch.distributed.checkpoint when initialized; otherwise create a portable local checkpoint."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        try:
            import torch.distributed.checkpoint as dcp

            dcp.save(state, checkpoint_id=str(path))
            return path
        except Exception:
            pass
    torch.save(state, path / "checkpoint.pt")
    return path


def load_distributed_checkpoint(path: Path, state: dict[str, Any] | None = None) -> dict[str, Any]:
    path = Path(path)
    local = path / "checkpoint.pt"
    if local.is_file():
        return torch.load(local, map_location="cpu", weights_only=False)
    if state is None:
        raise ValueError("Se requiere una plantilla de estado para cargar un checkpoint distribuido.")
    import torch.distributed.checkpoint as dcp

    dcp.load(state, checkpoint_id=str(path))
    return state
