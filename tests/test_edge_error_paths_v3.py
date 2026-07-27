from pathlib import Path

import pytest
import torch

from neural_labs.core.registry import ensure_module, first_parameter_device, module_parameter_count
from neural_labs.distributed import DistributedContext, wrap_distributed
from neural_labs.exporting import export_executorch
from neural_labs.telemetry import configure_opentelemetry
from test_inference_registry_v3 import make_run


def test_registry_helpers() -> None:
    model = torch.nn.Linear(2, 1)
    assert ensure_module(model) is model
    assert module_parameter_count(model) == 3
    assert first_parameter_device(model).type == "cpu"
    with pytest.raises(TypeError):
        ensure_module("bad")


def test_distributed_unknown_strategy() -> None:
    context = DistributedContext(0, 2, 0, "gloo", "cpu")
    with pytest.raises(ValueError):
        wrap_distributed(torch.nn.Linear(2, 1), context, "unknown")


def test_executorch_missing_extra(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    with pytest.raises(RuntimeError):
        export_executorch(run)


def test_otel_configuration_returns_boolean() -> None:
    assert isinstance(configure_opentelemetry("test-service"), bool)
