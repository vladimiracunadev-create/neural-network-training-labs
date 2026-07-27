from pathlib import Path

import torch
from fastapi.testclient import TestClient

from neural_labs.deployment.api import ModelServer, create_app
from neural_labs.inference import InferencePackage
from neural_labs.telemetry import MetricsCollector


class FakeServer(ModelServer):
    def __init__(self):
        pass

    def get(self):
        model = torch.nn.Linear(4, 3)
        return InferencePackage(Path("."), model, {"lab_id": "demo", "architecture": "mlp", "input_shape": [4], "class_names": ["a", "b", "c"]}, torch.device("cpu"))


def test_api_endpoints() -> None:
    client = TestClient(create_app(FakeServer()))
    assert client.get("/health").status_code == 200
    assert client.get("/model").json()["lab_id"] == "demo"
    response = client.post("/predict", json={"features": [0, 0, 0, 0]})
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 1
    batch = client.post("/predict-batch", json={"instances": [[0, 0, 0, 0], [1, 1, 1, 1]]})
    assert len(batch.json()["predictions"]) == 2
    assert "neural_labs_requests_total" in client.get("/metrics").text


def test_metrics_collector_tracks_errors() -> None:
    collector = MetricsCollector()
    try:
        with collector.request(prediction=True):
            raise ValueError("boom")
    except ValueError:
        pass
    assert collector.metrics.requests_total == 1
    assert collector.metrics.predictions_total == 1
    assert collector.metrics.errors_total == 1
    assert "neural_labs_errors_total" in collector.metrics.as_prometheus()
