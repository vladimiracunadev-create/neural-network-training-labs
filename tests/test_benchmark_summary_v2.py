from neural_labs.benchmarking import summarize_benchmark


def test_benchmark_summary_aggregates_numeric_metrics() -> None:
    records = [
        {"seed": 1, "metrics": {"accuracy": 0.8, "label": "a"}},
        {"seed": 2, "metrics": {"accuracy": 0.9, "label": "b"}},
        {"seed": 3, "metrics": {"accuracy": 1.0, "label": "c"}},
    ]
    summary = summarize_benchmark(records)
    assert summary["runs"] == 3
    assert round(summary["metrics"]["accuracy"]["mean"], 6) == 0.9
    assert summary["metrics"]["accuracy"]["std"] > 0
