import numpy as np
import pytest

from neural_labs.datasets import ArrayDataset, DataBundle, audit_bundle


def make_bundle(test_ids: list[str]) -> DataBundle:
    train_ids = ["train-1", "train-2"]
    validation_ids = ["validation-1"]
    train = ArrayDataset(np.zeros((2, 2)), np.array([0, 1]), train_ids)
    validation = ArrayDataset(np.zeros((1, 2)), np.array([0]), validation_ids)
    test = ArrayDataset(np.zeros((len(test_ids), 2)), np.zeros(len(test_ids), dtype=int), test_ids)
    return DataBundle("fixture", "fixture", "binary_classification", train, validation, test, (2,), 2, train_ids=train_ids, validation_ids=validation_ids, test_ids=test_ids)


def test_split_audit_accepts_disjoint_ids() -> None:
    assert audit_bundle(make_bundle(["test-1"]))["ok"] is True


def test_split_audit_rejects_leakage() -> None:
    with pytest.raises(ValueError, match="Fuga de datos"):
        audit_bundle(make_bundle(["train-1"]))
