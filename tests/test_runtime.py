import json

import torch

from neural_labs.runtime import environment_info, parameter_count, save_json, seed_everything, sha256_strings


def test_seed_and_hash_are_reproducible() -> None:
    seed_everything(123)
    first = torch.rand(4)
    seed_everything(123)
    second = torch.rand(4)
    assert torch.equal(first, second)
    assert sha256_strings(["a", "b"]) == sha256_strings(["a", "b"])
    assert sha256_strings(["a", "b"]) != sha256_strings(["b", "a"])


def test_artifact_helpers(tmp_path) -> None:
    path = tmp_path / "value.json"
    save_json(path, {"ok": True})
    assert json.loads(path.read_text())["ok"] is True
    model = torch.nn.Linear(3, 2)
    counts = parameter_count(model)
    assert counts["total"] == 8
    assert "python" in environment_info(torch.device("cpu"))
