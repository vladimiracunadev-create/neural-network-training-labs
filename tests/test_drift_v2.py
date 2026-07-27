from sklearn.datasets import load_iris

from neural_labs.datasets import ArrayDataset, DataBundle
from neural_labs.drift import drift_report


def test_drift_report_for_real_iris_data() -> None:
    iris = load_iris()
    ids = [f"iris-{index}" for index in range(len(iris.data))]
    bundle = DataBundle(
        "iris-drift",
        "iris",
        "multiclass_classification",
        ArrayDataset(iris.data[:90], iris.target[:90], ids[:90]),
        ArrayDataset(iris.data[90:120], iris.target[90:120], ids[90:120]),
        ArrayDataset(iris.data[120:], iris.target[120:], ids[120:]),
        (4,),
        3,
        feature_names=iris.feature_names,
        train_ids=ids[:90],
        validation_ids=ids[90:120],
        test_ids=ids[120:],
    )
    report = drift_report(bundle)
    assert report["available"] is True
    assert "test" in report["comparisons"]
    assert report["comparisons"]["test"]["max_absolute_smd"] >= 0
