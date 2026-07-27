from sklearn.datasets import load_iris

from neural_labs.data_quality import quality_report
from neural_labs.datasets import ArrayDataset, DataBundle


def test_quality_report_uses_real_iris_and_detects_no_overlap() -> None:
    iris = load_iris()
    train_ids = [f"iris-{index}" for index in range(90)]
    validation_ids = [f"iris-{index}" for index in range(90, 120)]
    test_ids = [f"iris-{index}" for index in range(120, 150)]
    bundle = DataBundle(
        "iris-quality",
        "iris",
        "multiclass_classification",
        ArrayDataset(iris.data[:90], iris.target[:90], train_ids),
        ArrayDataset(iris.data[90:120], iris.target[90:120], validation_ids),
        ArrayDataset(iris.data[120:], iris.target[120:], test_ids),
        (4,),
        3,
        train_ids=train_ids,
        validation_ids=validation_ids,
        test_ids=test_ids,
    )
    report = quality_report(bundle)
    assert report["passed"] is True
    assert report["splits"]["train"]["samples"] == 90
    assert report["splits"]["train"]["feature_nan_count"] == 0
