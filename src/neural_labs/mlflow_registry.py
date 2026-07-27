from __future__ import annotations

from pathlib import Path
from typing import Any

from .inference import load_inference_package


def register_run_with_mlflow(
    run_dir: Path,
    *,
    model_name: str,
    alias: str = "challenger",
    tracking_uri: str | None = None,
) -> dict[str, Any]:
    try:
        import mlflow
        import mlflow.pytorch
        from mlflow import MlflowClient
    except ImportError as exc:
        raise RuntimeError('Instale el extra MLOps: pip install -e ".[mlops]"') from exc

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    package = load_inference_package(Path(run_dir), "cpu")
    with mlflow.start_run(run_name=Path(run_dir).name) as active_run:
        model_info = mlflow.pytorch.log_model(
            package.model,
            name="model",
            registered_model_name=model_name,
        )
        mlflow.log_artifacts(str(run_dir), artifact_path="run_artifacts")
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    matching = [version for version in versions if version.run_id == active_run.info.run_id]
    if not matching:
        raise RuntimeError("MLflow no devolvió una versión registrada para la ejecución.")
    version = max(matching, key=lambda item: int(item.version))
    client.set_registered_model_alias(model_name, alias, version.version)
    return {
        "model_name": model_name,
        "version": int(version.version),
        "alias": alias,
        "run_id": active_run.info.run_id,
        "model_uri": model_info.model_uri,
    }
