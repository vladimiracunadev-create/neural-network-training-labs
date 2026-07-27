from __future__ import annotations

import sys
import types
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import torch

from neural_labs.datasets import ArrayDataset, DataBundle


def _small_bundle() -> DataBundle:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(36, 4)).astype(np.float32)
    y = np.asarray([0, 1, 2] * 12, dtype=np.int64)
    train_ids = [f"tr-{i}" for i in range(24)]
    validation_ids = [f"va-{i}" for i in range(6)]
    test_ids = [f"te-{i}" for i in range(6)]
    return DataBundle(
        lab_id="02_mlp_nonlinear",
        dataset_name="real-fixture",
        task="multiclass_classification",
        train=ArrayDataset(x[:24], y[:24], train_ids),
        validation=ArrayDataset(x[24:30], y[24:30], validation_ids),
        test=ArrayDataset(x[30:], y[30:], test_ids),
        input_shape=(4,),
        num_classes=3,
        class_names=["a", "b", "c"],
        feature_names=["f1", "f2", "f3", "f4"],
        train_ids=train_ids,
        validation_ids=validation_ids,
        test_ids=test_ids,
        metadata={"dataset_hash": "fixture-hash"},
    )


def test_distributed_training_single_process(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import neural_labs.distributed_training as module
    from neural_labs.distributed import DistributedContext

    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "initialize_distributed", lambda: DistributedContext(0, 1, 0, "gloo", "cpu"))
    monkeypatch.setattr(module, "cleanup_distributed", lambda: None)
    monkeypatch.setattr(module, "get_lab", lambda _lab: {"task": "multiclass_classification", "architecture": "linear"})
    monkeypatch.setattr(module, "prepare_dataset", lambda *_args, **_kwargs: _small_bundle())
    monkeypatch.setattr(
        module,
        "merged_config",
        lambda *_args, **_kwargs: {
            "batch_size": 8,
            "learning_rate": 1e-3,
            "epochs": 2,
            "quick": {"epochs": 1},
            "selection_metric": "macro_f1",
        },
    )
    result = module.train_distributed(
        "02_mlp_nonlinear",
        quick=True,
        output_dir="distributed-runs",
        split_seed=11,
        training_seed=12,
    )
    run = Path(result["run_dir"])
    assert result["rank"] == 0
    assert (run / "best_model.pt").is_file()
    assert (run / "checkpoint-rank0.pt").is_file()
    assert (run / "experiment.lock.json").is_file()
    assert (run / "distributed_manifest.json").is_file()


def test_mlflow_registry_backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import neural_labs.mlflow_registry as module

    events: dict[str, object] = {}
    fake_mlflow = types.ModuleType("mlflow")
    fake_pytorch = types.ModuleType("mlflow.pytorch")

    class ActiveRun:
        info = types.SimpleNamespace(run_id="run-123")

    @contextmanager
    def start_run(**kwargs):
        events["run_name"] = kwargs["run_name"]
        yield ActiveRun()

    def log_model(model, **kwargs):
        events["model"] = model
        events["registered_name"] = kwargs["registered_model_name"]
        return types.SimpleNamespace(model_uri="models:/demo/3")

    class Client:
        def search_model_versions(self, _query):
            return [types.SimpleNamespace(run_id="other", version="2"), types.SimpleNamespace(run_id="run-123", version="3")]

        def set_registered_model_alias(self, name, alias, version):
            events["alias"] = (name, alias, version)

    fake_mlflow.start_run = start_run
    fake_mlflow.log_artifacts = lambda *args, **kwargs: events.setdefault("artifacts", (args, kwargs))
    fake_mlflow.set_tracking_uri = lambda uri: events.setdefault("tracking_uri", uri)
    fake_mlflow.MlflowClient = Client
    fake_mlflow.pytorch = fake_pytorch
    fake_pytorch.log_model = log_model
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    monkeypatch.setitem(sys.modules, "mlflow.pytorch", fake_pytorch)
    monkeypatch.setattr(module, "load_inference_package", lambda *_args, **_kwargs: types.SimpleNamespace(model=torch.nn.Linear(2, 1)))

    result = module.register_run_with_mlflow(tmp_path / "run", model_name="demo", alias="champion", tracking_uri="http://mlflow")
    assert result["version"] == 3
    assert result["alias"] == "champion"
    assert events["alias"] == ("demo", "champion", "3")


def test_graph_models_with_and_without_optional_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    from neural_labs.domains.graphs.models import GraphModelUnavailable, build_graph_model

    unavailable = build_graph_model("gcn", 4, 8, 3)
    if isinstance(unavailable, GraphModelUnavailable):
        with pytest.raises(RuntimeError):
            unavailable(torch.ones(2, 4), torch.zeros(2, 1, dtype=torch.long))

    fake_root = types.ModuleType("torch_geometric")
    fake_nn = types.ModuleType("torch_geometric.nn")

    class FakeConv(torch.nn.Module):
        def __init__(self, in_channels, out_channels, **_kwargs):
            super().__init__()
            self.linear = torch.nn.Linear(in_channels, out_channels)

        def forward(self, x, edge_index):
            del edge_index
            return self.linear(x)

    fake_nn.GATConv = fake_nn.GCNConv = fake_nn.SAGEConv = FakeConv
    fake_root.nn = fake_nn
    monkeypatch.setitem(sys.modules, "torch_geometric", fake_root)
    monkeypatch.setitem(sys.modules, "torch_geometric.nn", fake_nn)
    for kind in ("gcn", "graphsage", "gat"):
        model = build_graph_model(kind, 4, 8, 3)
        assert model(torch.ones(5, 4), torch.zeros(2, 1, dtype=torch.long)).shape == (5, 3)
    with pytest.raises(ValueError):
        build_graph_model("unknown", 4, 8, 3)


def test_generative_facade_exports() -> None:
    from neural_labs.domains.generative.models import Discriminator, Generator

    generator = Generator(8)
    samples = generator(torch.randn(2, 8))
    discriminator = Discriminator()
    assert samples.ndim == 4
    assert discriminator(samples).shape[0] == 2
