from pathlib import Path

import numpy as np

from neural_labs.cross_validation import cross_validate_lab
from neural_labs.datasets import ArrayDataset, DataBundle


def test_cross_validate_without_touching_test(monkeypatch, tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(60, 4)).astype(np.float32)
    y = np.tile(np.array([0, 1, 2]), 20)
    ids = [str(i) for i in range(60)]
    bundle = DataBundle(
        "16_backpropagation_manual", "iris", "multiclass_classification",
        ArrayDataset(x[:40], y[:40], ids[:40]),
        ArrayDataset(x[40:50], y[40:50], ids[40:50]),
        ArrayDataset(x[50:], y[50:], ids[50:]),
        (4,), 3, train_ids=ids[:40], validation_ids=ids[40:50], test_ids=ids[50:],
    )
    monkeypatch.setattr("neural_labs.cross_validation.prepare_dataset", lambda *a, **k: bundle)
    monkeypatch.setattr("neural_labs.cross_validation.ROOT", tmp_path)
    monkeypatch.setattr("neural_labs.cross_validation.create_run_dir", lambda name, output: Path(output) / name)

    def fake_train(model, fold_bundle, config, device, fold_dir, evaluate_test=True):
        fold_dir.mkdir(parents=True, exist_ok=True)
        metrics = {"macro_f1": 0.5 + config["seed"] / 1000}
        return model, {}, metrics, np.array([]), np.array([]), None

    monkeypatch.setattr("neural_labs.cross_validation._train_torch_model", fake_train)
    result = cross_validate_lab(
        "16_backpropagation_manual",
        folds=2,
        split_seed=5,
        training_seeds=[1, 2],
        quick=True,
        output_dir="cv",
    )
    assert len(result["results"]) == 4
    assert result["test_set_used"] is False
    assert (tmp_path / "cv" / "16_backpropagation_manual" / "summary.json").exists()
