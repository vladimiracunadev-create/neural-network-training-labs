from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from .catalog import ROOT


def seed_everything(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def get_device(preferred: str = "auto") -> torch.device:
    value = preferred.lower()
    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA fue solicitado, pero no está disponible.")
    if value == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("MPS fue solicitado, pero no está disponible.")
    if value not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Dispositivo no soportado: {preferred}")
    return torch.device(value)


def create_run_dir(lab_id: str, output_dir: str | Path = "runs") -> Path:
    root = Path(output_dir)
    if not root.is_absolute():
        root = ROOT / root
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    candidate = root / lab_id / stamp
    suffix = 1
    while candidate.exists():
        candidate = root / lab_id / f"{stamp}-{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def resolve_run(lab_id: str, run: str | Path = "latest") -> Path:
    base = ROOT / "runs" / lab_id
    if str(run) != "latest":
        path = Path(run)
        return path if path.is_absolute() else ROOT / path
    choices = sorted((p for p in base.glob("*") if p.is_dir()), reverse=True)
    if not choices:
        raise FileNotFoundError(f"No existen ejecuciones para {lab_id}")
    return choices[0]


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_strings(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def environment_info(device: torch.device | None = None) -> dict[str, Any]:
    package_names = [
        "numpy",
        "pandas",
        "torch",
        "torchvision",
        "scikit-learn",
        "matplotlib",
        "PyYAML",
        "ucimlrepo",
        "datasets",
        "torch-geometric",
        "kagglehub",
        "optuna",
        "onnx",
        "onnxruntime",
        "mlflow",
        "dvc",
        "tabulate",
    ]
    dependencies: dict[str, str] = {}
    for package_name in package_names:
        try:
            dependencies[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            continue
    info: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "dependencies": dependencies,
        "git_commit": git_commit(),
    }
    if device is not None:
        info["device"] = str(device)
    if torch.cuda.is_available():
        info["cuda"] = torch.version.cuda
        info["gpu"] = torch.cuda.get_device_name(0)
    if hasattr(torch.backends, "mps"):
        info["mps_available"] = bool(torch.backends.mps.is_available())
    return info


def parameter_count(model: torch.nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}
