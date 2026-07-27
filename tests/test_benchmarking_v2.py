import json

from neural_labs.benchmarking import collect_runs


def test_collect_runs_reads_metrics(tmp_path) -> None:
    run = tmp_path / "runs" / "00_numpy_neuron" / "20260101-000000"
    run.mkdir(parents=True)
    (run / "metrics.json").write_text(json.dumps({"accuracy": 0.91, "wall_time_seconds": 1.2}), encoding="utf-8")
    (run / "config.yaml").write_text("seed: 42\ndevice: cpu\n", encoding="utf-8")
    frame = collect_runs(tmp_path, "00_numpy_neuron")
    assert len(frame) == 1
    assert frame.iloc[0]["accuracy"] == 0.91
