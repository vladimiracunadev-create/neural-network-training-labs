import matplotlib.pyplot as plt
import numpy as np

from neural_labs.datasets import ArrayDataset, DataBundle
from neural_labs.visualization import dataset_overview, prediction_errors, run_summary


def test_dataset_overview_and_run_helpers(tmp_path) -> None:
    ids = [f"iris-{index}" for index in range(12)]
    dataset = ArrayDataset(np.arange(48, dtype=np.float32).reshape(12, 4), np.array([0, 1, 2] * 4), ids)
    bundle = DataBundle(
        "iris", "iris", "multiclass_classification", dataset, dataset, dataset, (4,), 3,
        class_names=["setosa", "versicolor", "virginica"],
        train_ids=[f"train-{value}" for value in ids],
        validation_ids=[f"validation-{value}" for value in ids],
        test_ids=[f"test-{value}" for value in ids],
    )
    table, figure = dataset_overview(bundle)
    assert not table.empty
    assert figure is not None
    plt.close(figure)

    (tmp_path / "metrics.json").write_text('{"accuracy": 1.0}', encoding="utf-8")
    (tmp_path / "predictions.csv").write_text("sample_id,y_true,y_pred\na,0,1\nb,1,1\n", encoding="utf-8")
    assert len(prediction_errors(tmp_path)) == 1
    assert run_summary(tmp_path)["metrics"]["accuracy"] == 1.0
