#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from neural_labs.catalog import list_labs
from neural_labs.datasets import describe_bundle, prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga y prepara datasets reales")
    parser.add_argument("--lab", action="append", choices=list_labs(), help="Puede repetirse. Sin --lab prepara todos.")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    selected = args.lab or list_labs()
    results = []
    for lab_id in selected:
        try:
            bundle = prepare_dataset(lab_id, quick=args.quick, seed=args.seed)
            results.append({"lab": lab_id, "ok": True, "description": describe_bundle(bundle)})
        except Exception as exc:
            results.append({"lab": lab_id, "ok": False, "error": str(exc)})
            if not args.continue_on_error:
                print(json.dumps(results, indent=2, ensure_ascii=False, default=str))
                raise
    print(json.dumps(results, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
