from __future__ import annotations

import copy
import json
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .inference import load_inference_package
from .runtime import save_json


@dataclass
class ExportReport:
    format: str
    path: str
    verified: bool
    max_absolute_error: float | None
    size_bytes: int
    latency_ms_mean: float | None = None
    notes: str = ""


def _example_input(contract: dict[str, Any], batch_size: int = 1) -> torch.Tensor:
    shape = tuple(int(value) for value in contract["input_shape"])
    dtype = torch.long if contract.get("architecture") in {"rnn", "rnn_text", "lstm_text", "transformer", "transformer_text"} else torch.float32
    if dtype == torch.long:
        return torch.ones((batch_size, *shape), dtype=dtype)
    return torch.zeros((batch_size, *shape), dtype=dtype)


def export_onnx(run_dir: Path, *, verify: bool = True, dynamic_batch: bool = True) -> ExportReport:
    package = load_inference_package(run_dir, "cpu")
    sample = _example_input(package.contract)
    path = Path(run_dir) / "model.onnx"
    dynamic_shapes = {"input": {0: "batch"}} if dynamic_batch else None
    verified = False
    max_error: float | None = None
    notes = ""
    try:
        program = torch.onnx.export(
            package.model,
            (sample,),
            f=str(path),
            input_names=["input"],
            output_names=["output"],
            dynamo=True,
            dynamic_shapes=dynamic_shapes,
            verify=verify,
            report=True,
            artifacts_dir=str(Path(run_dir) / "onnx_artifacts"),
        )
        verified = bool(verify)
        if verify:
            try:
                import onnxruntime as ort

                session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
                with torch.inference_mode():
                    expected = package.model(sample).detach().cpu().numpy()
                actual = session.run(None, {session.get_inputs()[0].name: sample.numpy()})[0]
                max_error = float(np.max(np.abs(expected - actual)))
                verified = bool(np.allclose(expected, actual, rtol=1e-4, atol=1e-5))
            except ImportError:
                notes = "ONNX Runtime no está instalado; se verificó únicamente mediante el exportador de PyTorch."
        del program
    except Exception as exc:
        raise RuntimeError(f"No fue posible exportar ONNX: {exc}") from exc
    report = ExportReport("onnx", str(path), verified, max_error, path.stat().st_size, notes=notes)
    save_json(Path(run_dir) / "onnx_export_report.json", asdict(report))
    return report


def quantize_dynamic(run_dir: Path) -> ExportReport:
    package = load_inference_package(run_dir, "cpu")
    quantized = copy.deepcopy(package.model).eval()
    backend = "torchao"
    notes = ""
    try:
        from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_

        quantize_(quantized, Int8DynamicActivationInt8WeightConfig())
    except ImportError:
        # Compatibility fallback for environments that have not installed the
        # dedicated TorchAO package yet. PyTorch has announced the migration of
        # eager quantization to TorchAO, so this path is deliberately labeled.
        backend = "torch.ao-compatibility-fallback"
        notes = 'Instale el extra export con TorchAO para usar la API quantize_ moderna.'
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            quantized = torch.ao.quantization.quantize_dynamic(quantized, {torch.nn.Linear}, dtype=torch.qint8)
    path = Path(run_dir) / "model-int8.pt"
    torch.save({"state_dict": quantized.state_dict(), "format": "dynamic-int8", "backend": backend}, path)
    sample = _example_input(package.contract, batch_size=16)
    started = time.perf_counter()
    with torch.inference_mode():
        for _ in range(10):
            quantized(sample)
    latency = (time.perf_counter() - started) * 1000 / 10
    report = ExportReport(
        "pytorch-dynamic-int8",
        str(path),
        True,
        None,
        path.stat().st_size,
        latency_ms_mean=latency,
        notes=f"backend={backend}. {notes}".strip(),
    )
    save_json(Path(run_dir) / "quantization_report.json", asdict(report))
    return report


def export_executorch(run_dir: Path) -> ExportReport:
    package = load_inference_package(run_dir, "cpu")
    sample = _example_input(package.contract)
    path = Path(run_dir) / "model.pte"
    try:
        from executorch.exir import to_edge_transform_and_lower
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

        exported_program = torch.export.export(package.model, (sample,), strict=True)
        edge = to_edge_transform_and_lower(exported_program, partitioner=[XnnpackPartitioner()])
        program = edge.to_executorch()
        path.write_bytes(program.buffer)
    except ImportError as exc:
        raise RuntimeError('Instale el extra edge: pip install -e ".[edge]"') from exc
    report = ExportReport("executorch", str(path), True, None, path.stat().st_size)
    save_json(Path(run_dir) / "executorch_export_report.json", asdict(report))
    return report


def benchmark_formats(run_dir: Path) -> dict[str, Any]:
    package = load_inference_package(run_dir, "cpu")
    sample = _example_input(package.contract, batch_size=16)
    results: dict[str, Any] = {}
    for name, model in [("pytorch_fp32", package.model)]:
        times = []
        with torch.inference_mode():
            for _ in range(3):
                model(sample)
            for _ in range(20):
                started = time.perf_counter()
                model(sample)
                times.append((time.perf_counter() - started) * 1000)
        results[name] = {
            "latency_ms_mean": float(np.mean(times)),
            "latency_ms_p95": float(np.percentile(times, 95)),
            "throughput_per_second": float(len(sample) / (np.mean(times) / 1000)),
        }
    save_json(Path(run_dir) / "format_benchmark.json", results)
    return results
