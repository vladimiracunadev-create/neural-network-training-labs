import json

from neural_labs.tracking import JsonlTracker


def test_jsonl_tracker_records_lifecycle(tmp_path) -> None:
    tracker = JsonlTracker(tmp_path / "tracking.jsonl")
    tracker.start(run_name="run-1", experiment_name="lab", tags={"dataset": "iris"})
    tracker.log_params({"seed": 42})
    tracker.log_metrics({"accuracy": 0.9, "label": "ignored"})
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("ok", encoding="utf-8")
    tracker.log_artifact(artifact)
    tracker.finish()
    events = [json.loads(line)["event"] for line in (tmp_path / "tracking.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events == ["start", "params", "metrics", "artifact", "finish"]
