from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from ..inference import InferencePackage, load_inference_package
from ..model_registry import LocalModelRegistry
from ..monitoring import PredictionLogger, monitoring_report
from ..telemetry import MetricsCollector, configure_opentelemetry


class PredictionRequest(BaseModel):
    features: list[Any] = Field(..., description="Una muestra o un lote de muestras preprocesadas.")


class BatchPredictionRequest(BaseModel):
    instances: list[list[Any]]


class ModelServer:
    def __init__(self, registry_path: Path, model_name: str, reference: str = "champion", device: str = "cpu"):
        self.registry = LocalModelRegistry(registry_path)
        self.model_name = model_name
        self.reference = reference
        self.device = device
        self.package: InferencePackage | None = None

    def load(self) -> InferencePackage:
        entry = self.registry.resolve(self.model_name, self.reference)
        self.package = load_inference_package(Path(entry.run_dir), torch.device(self.device))
        return self.package

    def get(self) -> InferencePackage:
        return self.package or self.load()


@lru_cache(maxsize=1)
def settings() -> tuple[Path, str, str, str]:
    return (
        Path(os.environ.get("NEURAL_LABS_REGISTRY", "model-registry.json")),
        os.environ.get("NEURAL_LABS_MODEL", "default"),
        os.environ.get("NEURAL_LABS_MODEL_REFERENCE", "champion"),
        os.environ.get("NEURAL_LABS_DEVICE", "cpu"),
    )


def create_app(server: ModelServer | None = None) -> FastAPI:
    registry_path, model_name, reference, device = settings()
    model_server = server or ModelServer(registry_path, model_name, reference, device)
    metrics = MetricsCollector()
    configure_opentelemetry()
    prediction_log = Path(os.environ.get("NEURAL_LABS_PREDICTION_LOG", "monitoring/predictions.jsonl"))
    reference_stats = Path(os.environ.get("NEURAL_LABS_REFERENCE_STATS", "monitoring/reference_stats.json"))
    prediction_logger = PredictionLogger(prediction_log, model=model_name, reference=reference)
    app = FastAPI(title="Neural Network Training Labs API", version="1.0.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        with metrics.request():
            try:
                package = model_server.get()
                return {"status": "ok", "lab_id": package.contract["lab_id"], "device": str(package.device)}
            except Exception as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/model")
    def model_info() -> dict[str, Any]:
        with metrics.request():
            package = model_server.get()
            return package.contract

    @app.post("/predict")
    def predict(request: PredictionRequest) -> dict[str, Any]:
        with metrics.request(prediction=True):
            package = model_server.get()
            tensor = torch.as_tensor(request.features)
            if tensor.ndim == len(package.contract.get("input_shape", [])):
                tensor = tensor.unsqueeze(0)
            expected_dtype = torch.long if package.contract.get("architecture") in {"rnn", "rnn_text", "lstm_text", "transformer", "transformer_text"} else torch.float32
            result = package.predict_tensor(tensor.to(expected_dtype))
            prediction_logger.log(request.features, result["predictions"])
            return result

    @app.post("/predict-batch")
    def predict_batch(request: BatchPredictionRequest) -> dict[str, Any]:
        with metrics.request(prediction=True):
            package = model_server.get()
            expected_dtype = torch.long if package.contract.get("architecture") in {"rnn", "rnn_text", "lstm_text", "transformer", "transformer_text"} else torch.float32
            result = package.predict_tensor(torch.as_tensor(request.instances, dtype=expected_dtype))
            prediction_logger.log(request.instances, result["predictions"])
            return result

    @app.get("/drift")
    def drift() -> dict[str, Any]:
        return monitoring_report(prediction_log, reference_stats)

    @app.get("/metrics")
    def prometheus_metrics() -> Response:
        return Response(metrics.metrics.as_prometheus(), media_type="text/plain; version=0.0.4")

    return app


app = create_app()
