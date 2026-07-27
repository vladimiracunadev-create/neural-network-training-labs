import numpy as np

from neural_labs.metrics import classification_metrics, expected_calibration_error, regression_metrics


def test_binary_classification_metrics() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    probabilities = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
    metrics = classification_metrics(y_true, y_pred, probabilities)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_regression_metrics() -> None:
    metrics = regression_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["r2"] == 1.0


def test_ece_is_bounded() -> None:
    value = expected_calibration_error(np.array([0, 1]), np.array([0.1, 0.9]))
    assert 0.0 <= value <= 1.0
