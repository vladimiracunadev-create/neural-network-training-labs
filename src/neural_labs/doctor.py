from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import torch

from .catalog import ROOT


OPTIONAL_PACKAGES = {
    "data": "ucimlrepo",
    "vision": "torchvision",
    "text": "datasets",
    "transformers": "transformers",
    "audio": "torchaudio",
    "graph": "torch_geometric",
    "kaggle": "kagglehub",
    "search": "optuna",
    "export": "onnxruntime",
    "notebooks": "jupyterlab",
    "dashboard": "streamlit",
    "quantization": "torchao",
    "mlops": "mlflow",
    "data_versioning": "dvc",
}


def environment_doctor() -> dict[str, Any]:
    disk = shutil.disk_usage(ROOT)
    packages = {name: importlib.util.find_spec(module) is not None for name, module in OPTIONAL_PACKAGES.items()}
    kaggle_credentials = bool(os.getenv("KAGGLE_API_TOKEN") or os.getenv("KAGGLE_USERNAME") or (Path.home() / ".kaggle" / "kaggle.json").exists())
    checks = {
        "python_supported": (3, 11) <= sys.version_info[:2] < (3, 14),
        "project_root_writable": os.access(ROOT, os.W_OK),
        "free_disk_gb": round(disk.free / 1024**3, 2),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        "kaggle_credentials_detected": kaggle_credentials,
        "optional_packages": packages,
    }
    warnings: list[str] = []
    if checks["free_disk_gb"] < 10:
        warnings.append("Quedan menos de 10 GB libres; visión, audio y modelos pueden requerir más espacio.")
    if not kaggle_credentials:
        warnings.append("Los laboratorios Kaggle requerirán autenticación antes de descargar.")
    if not packages["mlops"]:
        warnings.append('MLflow es opcional. Instale el extra con: pip install -e ".[mlops]"')
    if not packages["data_versioning"]:
        warnings.append('DVC es opcional. Instale el extra con: pip install -e ".[data-versioning]"')
    return {"checks": checks, "warnings": warnings, "ready_for_core_labs": bool(checks["python_supported"] and checks["project_root_writable"])}
