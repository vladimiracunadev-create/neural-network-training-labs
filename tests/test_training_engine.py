import numpy as np
import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from neural_labs.datasets import ArrayDataset, DataBundle
from neural_labs.experiments import _train_torch_model
from neural_labs.models import build_model


def test_training_engine_with_bundled_real_iris_data(tmp_path) -> None:
    iris = load_iris()
    x_train, x_temp, y_train, y_temp = train_test_split(iris.data, iris.target, test_size=0.4, random_state=7, stratify=iris.target)
    x_validation, x_test, y_validation, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=7, stratify=y_temp)
    scaler = StandardScaler().fit(x_train)
    x_train, x_validation, x_test = scaler.transform(x_train), scaler.transform(x_validation), scaler.transform(x_test)
    train_ids = [f"iris-train-{i}" for i in range(len(x_train))]
    validation_ids = [f"iris-validation-{i}" for i in range(len(x_validation))]
    test_ids = [f"iris-test-{i}" for i in range(len(x_test))]
    bundle = DataBundle(
        "iris-engine-test", "iris", "multiclass_classification",
        ArrayDataset(x_train, y_train, train_ids),
        ArrayDataset(x_validation, y_validation, validation_ids),
        ArrayDataset(x_test, y_test, test_ids),
        (4,), 3, class_names=iris.target_names.tolist(),
        train_ids=train_ids, validation_ids=validation_ids, test_ids=test_ids,
    )
    config = {
        "learning_rate": 0.01, "batch_size": 16, "epochs": 2, "patience": 2,
        "selection_metric": "macro_f1", "quick": {"epochs": 2}, "_quick": True,
    }
    model = build_model("mlp", (4,), 3, {"hidden": (16,)})
    trained, history, metrics, y_true, y_pred, probabilities = _train_torch_model(model, bundle, config, torch.device("cpu"), tmp_path)
    assert trained is not None
    assert len(history["train_loss"]) >= 1
    assert np.isfinite(history["train_loss"]).all()
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert len(y_true) == len(y_pred) == len(x_test)
    assert probabilities.shape == (len(x_test), 3)
    assert (tmp_path / "best_model.pt").exists()
