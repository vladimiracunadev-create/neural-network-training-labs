import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from neural_labs.baselines import run_baseline
from neural_labs.datasets import ArrayDataset, DataBundle


def test_baseline_respects_requested_split() -> None:
    iris = load_iris()
    x_train, x_temp, y_train, y_temp = train_test_split(iris.data, iris.target, test_size=0.4, random_state=7, stratify=iris.target)
    x_validation, x_test, y_validation, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=7, stratify=y_temp)
    train_ids = [f"t{i}" for i in range(len(x_train))]
    val_ids = [f"v{i}" for i in range(len(x_validation))]
    test_ids = [f"e{i}" for i in range(len(x_test))]
    bundle = DataBundle(
        "16_backpropagation_manual", "iris", "multiclass_classification",
        ArrayDataset(x_train, y_train, train_ids),
        ArrayDataset(x_validation, y_validation, val_ids),
        ArrayDataset(x_test, y_test, test_ids),
        (4,), 3, class_names=iris.target_names.tolist(),
        train_ids=train_ids, validation_ids=val_ids, test_ids=test_ids,
    )
    validation = run_baseline("16_backpropagation_manual", bundle, quick=True, evaluation_split="validation")
    test = run_baseline("16_backpropagation_manual", bundle, quick=True, evaluation_split="test")
    assert validation["evaluation_split"] == "validation"
    assert test["evaluation_split"] == "test"
