import numpy as np

from neural_labs.baselines import run_baseline
from neural_labs.datasets import ArrayDataset, DataBundle


def bundle(task="regression", lab_id="18_optimizers_and_schedulers"):
    rng = np.random.default_rng(7)
    x = rng.normal(size=(80, 4)).astype(np.float32)
    y = (x[:, 0] * 2 + 0.1).astype(np.float32) if task == "regression" else (x[:, 0] > 0).astype(np.int64)
    ids = [f"id-{i}" for i in range(80)]
    return DataBundle(
        lab_id, "demo", task,
        ArrayDataset(x[:50], y[:50], ids[:50]),
        ArrayDataset(x[50:65], y[50:65], ids[50:65]),
        ArrayDataset(x[65:], y[65:], ids[65:]),
        (4,), None if task == "regression" else 2,
        train_ids=ids[:50], validation_ids=ids[50:65], test_ids=ids[65:],
    )


def test_regression_baselines() -> None:
    result = run_baseline("18_optimizers_and_schedulers", bundle(), quick=True, evaluation_split="validation")
    assert "ridge" in result
    assert "rmse" in result["ridge"]


def test_generation_reference() -> None:
    data = bundle("multiclass_classification", "08_gan_generation")
    data.task = "generation"
    result = run_baseline("08_gan_generation", data, quick=True)
    assert result["real_data_reference"]["samples"] == 50


def test_inventory_baseline() -> None:
    data = bundle("reinforcement_learning", "10_dqn_reinforcement")
    data.raw = {
        "train_demand": np.array([1, 2, 3, 2, 4], dtype=np.float32),
        "validation_demand": np.array([2, 2, 5], dtype=np.float32),
        "test_demand": np.array([1, 4, 2], dtype=np.float32),
    }
    result = run_baseline("10_dqn_reinforcement", data, evaluation_split="validation")
    assert result["periodic_reorder_policy"]["service_level"] <= 1.0
