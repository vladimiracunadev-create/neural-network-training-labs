#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import nbformat
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_labs.schema import validate_repository  # noqa: E402


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def main() -> None:
    catalog = yaml.safe_load((ROOT / "configs/labs.yaml").read_text(encoding="utf-8"))["labs"]
    if len(catalog) != 25:
        fail(f"se esperaban 25 laboratorios y existen {len(catalog)}")
    source_types = {"uci", "uci_retail", "uci_har", "torchvision", "huggingface", "pyg", "sklearn", "kaggle"}
    unique_datasets: set[str] = set()
    for lab in catalog:
        lab_id = lab["id"]
        unique_datasets.add(lab["dataset"])
        if lab["source_type"] not in source_types:
            fail(f"fuente no autorizada en {lab_id}: {lab['source_type']}")
        folder = ROOT / "labs" / lab_id
        for notebook_name in ("notebook.ipynb", "notebook_student.ipynb", "notebook_solution.ipynb"):
            notebook = nbformat.read(folder / notebook_name, as_version=4)
            if len(notebook.cells) < 20:
                fail(f"notebook superficial en {lab_id}/{notebook_name}: {len(notebook.cells)} celdas")
            if sum(cell.cell_type == "code" for cell in notebook.cells) < 8:
                fail(f"notebook sin suficiente código ejecutable en {lab_id}/{notebook_name}")
    if len(unique_datasets) != 19:
        fail(f"se esperaban 19 datasets únicos y existen {len(unique_datasets)}")
    advanced = yaml.safe_load((ROOT / "configs/advanced_tracks.yaml").read_text(encoding="utf-8"))["tracks"]
    if len(advanced) != 6:
        fail(f"se esperaban 6 especializaciones avanzadas y existen {len(advanced)}")
    for track in advanced:
        track_id = track["id"]
        folder = ROOT / "advanced_labs" / track_id
        if not folder.is_dir():
            fail(f"falta advanced_labs/{track_id}")
        manifest = yaml.safe_load((folder / "data" / "dataset.yaml").read_text(encoding="utf-8"))
        if manifest.get("synthetic") is not False:
            fail(f"{track_id} no confirma dataset real")
        for notebook_name in ("notebook.ipynb", "notebook_student.ipynb", "notebook_solution.ipynb"):
            notebook = nbformat.read(folder / notebook_name, as_version=4)
            if len(notebook.cells) < 18:
                fail(f"notebook avanzado superficial en {track_id}/{notebook_name}")
            if sum(cell.cell_type == "code" for cell in notebook.cells) < 8:
                fail(f"notebook avanzado sin suficiente código en {track_id}/{notebook_name}")
    if (ROOT / "src/neural_labs/synthetic.py").exists():
        fail("quedó un generador de datos sintéticos")
    for pattern in ("*.csv", "*.parquet", "*.npz", "*.pt", "*.pth", "*.onnx"):
        committed = [path for path in (ROOT / "labs").rglob(pattern) if path.is_file()]
        if committed:
            fail(f"archivos de datos/modelos versionados dentro de labs: {committed[:3]}")
    issues = validate_repository(ROOT)
    errors = [issue for issue in issues if issue.severity == "error"]
    if errors:
        fail("; ".join(f"{issue.path}: {issue.message}" for issue in errors[:10]))
    manifest = json.loads((ROOT / "repository-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("synthetic_datasets") != 0:
        fail("repository-manifest no confirma cero datasets sintéticos")
    print(json.dumps({
        "ok": True, "version": "1.0.0", "core_labs": len(catalog), "advanced_tracks": len(advanced),
        "learning_paths": len(catalog) + len(advanced), "datasets": len(unique_datasets),
        "notebooks": (len(catalog) + len(advanced)) * 3,
        "schema_warnings": sum(issue.severity == "warning" for issue in issues),
    }, indent=2))


if __name__ == "__main__":
    main()
