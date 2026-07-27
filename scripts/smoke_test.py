#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from neural_labs.catalog import list_labs
from neural_labs.experiments import run_lab


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenamiento real reducido")
    parser.add_argument("--lab", required=True, choices=list_labs())
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps", "auto"])
    parser.add_argument("--output-dir", default="runs/smoke")
    args = parser.parse_args()
    result = run_lab(args.lab, quick=True, output_dir=Path(args.output_dir), device=args.device)
    print(json.dumps({"ok": True, "lab": args.lab, "run_dir": str(result.run_dir), "metrics": result.metrics}, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
