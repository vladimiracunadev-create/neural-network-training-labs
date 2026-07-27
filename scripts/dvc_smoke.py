from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

from neural_labs.experiments import run_lab

ROOT = Path(__file__).resolve().parents[1]
params = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))
out = ROOT / "reports" / "dvc-smoke"
if out.exists():
    shutil.rmtree(out)
out.mkdir(parents=True)
result = run_lab(
    params["lab"],
    quick=bool(params["quick"]),
    config_name=params["config"],
    seed=int(params["seed"]),
    device=params["device"],
    output_dir=out,
    tracker="json",
)
(out / "summary.json").write_text(
    json.dumps({"lab": result.lab_id, "run_dir": str(result.run_dir), "metrics": result.metrics}, indent=2, ensure_ascii=False, default=str),
    encoding="utf-8",
)
