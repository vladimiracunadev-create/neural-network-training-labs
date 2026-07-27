from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

from .core.protocol import assert_lock_matches
from .datasets import DataBundle
from .models import build_model
from .runtime import save_json


@dataclass
class InferencePackage:
    run_dir: Path
    model: torch.nn.Module
    contract: dict[str, Any]
    device: torch.device

    def predict_tensor(self, tensor: torch.Tensor) -> dict[str, Any]:
        self.model.eval()
        with torch.inference_mode():
            output = self.model(tensor.to(self.device)).detach().cpu()
        task = str(self.contract.get("task", ""))
        if task == "regression":
            predictions = output.reshape(-1)
            probabilities = None
        elif output.ndim == 1 or output.shape[-1] == 1:
            positive = torch.sigmoid(output.reshape(-1))
            probabilities = torch.stack([1 - positive, positive], dim=1)
            predictions = (positive >= float(self.contract.get("decision_threshold", 0.5))).long()
        else:
            probabilities = torch.softmax(output, dim=1)
            predictions = probabilities.argmax(dim=1)
        return {
            "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist() if probabilities is not None else None,
            "output": output.tolist(),
            "class_names": self.contract.get("class_names", []),
        }


def persist_inference_contract(bundle: DataBundle, run_dir: Path, *, architecture: str) -> Path:
    contract = {
        "schema_version": "1.0",
        "lab_id": bundle.lab_id,
        "dataset": bundle.dataset_name,
        "architecture": architecture,
        "task": bundle.task,
        "input_shape": list(bundle.input_shape or []),
        "num_classes": bundle.num_classes,
        "class_names": bundle.class_names,
        "feature_names": bundle.feature_names,
        "decision_threshold": 0.5,
        "accepted_inputs": ["json", "csv", "npy", "image"],
        "servable": architecture not in {
            "numpy_logistic",
            "numpy_mlp",
            "dcgan",
            "gcn",
            "autoencoder",
            "dqn",
            "dueling_dqn",
            "dqn_inventory",
        },
        "preprocessing_fitted_on": "train_only",
        "split_manifest": bundle.metadata.get("split_manifest"),
    }
    vocabulary = bundle.metadata.get("vocabulary")
    if vocabulary:
        vocabulary_path = run_dir / "vocabulary.json"
        vocabulary_path.write_text(json.dumps(vocabulary, ensure_ascii=False), encoding="utf-8")
        contract["vocabulary_file"] = vocabulary_path.name
    transformer = bundle.metadata.get("transformer")
    if transformer is not None:
        transformer_path = run_dir / "preprocessor.joblib"
        joblib.dump(transformer, transformer_path)
        contract["preprocessor_file"] = transformer_path.name
    path = run_dir / "inference_contract.json"
    save_json(path, contract)
    return path


def _load_model_spec(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "model_spec.json"
    if not path.is_file():
        raise FileNotFoundError(f"Falta {path.name} en la ejecución.")
    return json.loads(path.read_text(encoding="utf-8"))


def load_inference_package(run_dir: Path, device: torch.device | str = "cpu") -> InferencePackage:
    run_dir = Path(run_dir).resolve()
    lock = assert_lock_matches(run_dir)
    contract_path = run_dir / "inference_contract.json"
    if not contract_path.is_file():
        raise FileNotFoundError(f"Falta {contract_path.name}; vuelva a entrenar con la versión 3.")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("servable") is False:
        raise NotImplementedError(f"La arquitectura {contract.get('architecture')} requiere un servidor especializado.")
    if lock["lab_id"] != contract["lab_id"]:
        raise ValueError("El lock y el contrato de inferencia pertenecen a laboratorios diferentes.")
    spec = _load_model_spec(run_dir)
    resolved_device = torch.device(device)
    architecture = spec["architecture"]
    if architecture == "distillation_cnn":
        architecture = "distillation_student"
    model = build_model(
        architecture,
        tuple(spec["input_shape"]),
        spec.get("num_classes"),
        spec.get("metadata") or {},
    )
    checkpoint = torch.load(run_dir / "best_model.pt", map_location=resolved_device, weights_only=False)
    state_dict = checkpoint.get("student") or checkpoint.get("state_dict") or checkpoint
    model.load_state_dict(state_dict)
    return InferencePackage(run_dir, model.to(resolved_device).eval(), contract, resolved_device)


def load_external_input(path: Path, contract: dict[str, Any], artifacts_dir: Path | None = None) -> torch.Tensor:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = np.load(path)
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        array = np.asarray(payload["features"] if isinstance(payload, dict) and "features" in payload else payload)
    elif suffix == ".csv":
        import pandas as pd

        frame = pd.read_csv(path)
        preprocessor_name = contract.get("preprocessor_file")
        if preprocessor_name:
            base_dir = Path(artifacts_dir) if artifacts_dir is not None else path.parent
            preprocessor = joblib.load(base_dir / preprocessor_name)
            array = preprocessor.transform(frame)
        else:
            array = frame.to_numpy()
    elif suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        from PIL import Image

        image = Image.open(path).convert("RGB")
        expected = contract.get("input_shape", [3, 32, 32])
        height, width = int(expected[-2]), int(expected[-1])
        image = image.resize((width, height))
        array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    else:
        raise ValueError(f"Formato de entrada no soportado: {suffix}")
    architecture = str(contract.get("architecture", ""))
    dtype = torch.long if architecture in {"rnn", "rnn_text", "lstm_text", "transformer", "transformer_text"} else torch.float32
    tensor = torch.as_tensor(array, dtype=dtype)
    expected_rank = len(contract.get("input_shape", []))
    if tensor.ndim == expected_rank:
        tensor = tensor.unsqueeze(0)
    return tensor
