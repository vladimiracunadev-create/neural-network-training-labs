import json
from pathlib import Path

import numpy as np
import torch

from neural_labs.core.protocol import ExperimentLock, SeedPlan
from neural_labs.inference import load_external_input, load_inference_package
from neural_labs.model_registry import LocalModelRegistry
from neural_labs.models import build_model


def make_run(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    model = build_model("mlp", (4,), 3, {})
    torch.save({"state_dict": model.state_dict()}, run / "best_model.pt")
    (run / "model_spec.json").write_text(json.dumps({"architecture": "mlp", "input_shape": [4], "num_classes": 3, "metadata": {}}))
    (run / "inference_contract.json").write_text(json.dumps({"lab_id": "demo", "architecture": "mlp", "input_shape": [4], "num_classes": 3, "class_names": ["a", "b", "c"]}))
    (run / "metrics.json").write_text(json.dumps({"accuracy": 0.8}))
    ExperimentLock.create(
        lab_id="demo",
        seeds=SeedPlan(42, 43),
        config_name="baseline",
        selection_metric="accuracy",
        selected_checkpoint=run / "best_model.pt",
        dataset_hash="hash",
    ).write(run)
    return run


def test_inference_package_and_external_inputs(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    package = load_inference_package(run)
    result = package.predict_tensor(torch.zeros(2, 4))
    assert len(result["predictions"]) == 2
    json_path = tmp_path / "x.json"
    json_path.write_text(json.dumps({"features": [1, 2, 3, 4]}))
    assert load_external_input(json_path, package.contract).shape == (1, 4)
    npy_path = tmp_path / "x.npy"
    np.save(npy_path, np.ones(4, dtype=np.float32))
    assert load_external_input(npy_path, package.contract).shape == (1, 4)


def test_local_model_registry_aliases(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    registry = LocalModelRegistry(tmp_path / "registry.json")
    first = registry.register("classifier", run, alias="champion")
    second = registry.register("classifier", run, alias="challenger")
    assert first.version == 1
    assert second.version == 2
    registry.set_alias("classifier", 2, "champion")
    assert registry.resolve("classifier", "champion").version == 2
    assert len(registry.list()["models"]["classifier"]) == 2
