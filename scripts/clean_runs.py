#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Elimina artefactos locales, nunca datasets fuente")
    parser.add_argument("--runs", action="store_true", help="Vacía runs/")
    parser.add_argument("--processed", action="store_true", help="Vacía data/processed/")
    args = parser.parse_args()
    for enabled, folder in ((args.runs, ROOT / "runs"), (args.processed, ROOT / "data/processed")):
        if enabled:
            for path in folder.iterdir():
                if path.name == ".gitkeep":
                    continue
                shutil.rmtree(path) if path.is_dir() else path.unlink()


if __name__ == "__main__":
    main()
