import numpy as np

from neural_labs.statistics import bootstrap_confidence_intervals


def test_bootstrap_classification_intervals_are_bounded() -> None:
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0])
    report = bootstrap_confidence_intervals(y_true, y_pred, task="binary_classification", samples=60, seed=7)
    assert 0 <= report["accuracy"]["lower"] <= report["accuracy"]["upper"] <= 1
    assert 0 <= report["macro_f1"]["lower"] <= report["macro_f1"]["upper"] <= 1
