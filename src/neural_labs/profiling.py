from __future__ import annotations

import statistics
import time
from typing import Any

import torch
from torch import nn


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def profile_inference(
    model: nn.Module,
    sample: torch.Tensor,
    device: torch.device,
    *,
    warmup: int = 3,
    iterations: int = 10,
) -> dict[str, Any]:
    model.eval()
    sample = sample.to(device)
    with torch.inference_mode():
        for _ in range(max(0, warmup)):
            model(sample)
        _synchronize(device)
        measurements: list[float] = []
        for _ in range(max(1, iterations)):
            started = time.perf_counter()
            model(sample)
            _synchronize(device)
            measurements.append((time.perf_counter() - started) * 1000.0)
    mean_ms = statistics.fmean(measurements)
    sorted_values = sorted(measurements)
    p95_index = min(len(sorted_values) - 1, max(0, int(round(0.95 * (len(sorted_values) - 1)))))
    batch = int(sample.shape[0]) if sample.ndim else 1
    result: dict[str, Any] = {
        "latency_mean_ms": float(mean_ms),
        "latency_p95_ms": float(sorted_values[p95_index]),
        "throughput_samples_per_second": float(batch / max(mean_ms / 1000.0, 1e-12)),
        "profile_batch_size": batch,
        "profile_iterations": len(measurements),
    }
    if device.type == "cuda":
        result["cuda_peak_memory_mb"] = float(torch.cuda.max_memory_allocated(device) / 1024**2)
    return result
