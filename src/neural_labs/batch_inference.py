from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .inference import load_external_input, load_inference_package


def batch_predict(run_dir: Path, input_path: Path, *, output_path: Path, batch_size: int = 128, device: str = "cpu") -> Path:
    package = load_inference_package(run_dir, torch.device(device))
    tensor = load_external_input(input_path, package.contract, package.run_dir)
    rows: list[dict[str, Any]] = []
    for start in range(0, len(tensor), batch_size):
        result = package.predict_tensor(tensor[start : start + batch_size])
        for offset, prediction in enumerate(result["predictions"]):
            probability = result["probabilities"][offset] if result["probabilities"] is not None else None
            rows.append({"row": start + offset, "prediction": prediction, "probabilities": probability})
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".jsonl":
        output_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    else:
        import pandas as pd

        pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path
