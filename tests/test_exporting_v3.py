from pathlib import Path

import torch

from neural_labs.exporting import benchmark_formats, export_onnx, quantize_dynamic
from test_inference_registry_v3 import make_run


def test_quantize_and_benchmark(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    report = quantize_dynamic(run)
    assert report.format == "pytorch-dynamic-int8"
    assert Path(report.path).exists()
    benchmark = benchmark_formats(run)
    assert benchmark["pytorch_fp32"]["throughput_per_second"] > 0


def test_onnx_export_contract_with_mock(tmp_path: Path, monkeypatch) -> None:
    run = make_run(tmp_path)

    def fake_export(model, args, f, **kwargs):
        Path(f).write_bytes(b"onnx")
        return object()

    monkeypatch.setattr(torch.onnx, "export", fake_export)
    report = export_onnx(run, verify=False)
    assert report.size_bytes == 4
    assert Path(report.path).exists()
