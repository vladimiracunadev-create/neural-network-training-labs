#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from neural_labs.catalog import list_labs
from neural_labs.datasets import audit_bundle, prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Audita train/validation/test")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--lab", choices=list_labs())
    group.add_argument("--all", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    selected = list_labs() if args.all else [args.lab]
    reports = {}
    for lab_id in selected:
        reports[lab_id] = audit_bundle(prepare_dataset(lab_id, quick=args.quick, seed=args.seed))
    print(json.dumps(reports, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
