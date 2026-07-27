from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from threading import Lock
from typing import Iterator


@dataclass
class ServiceMetrics:
    requests_total: int = 0
    predictions_total: int = 0
    errors_total: int = 0
    latency_seconds_sum: float = 0.0
    latency_seconds_max: float = 0.0

    def as_prometheus(self) -> str:
        values = asdict(self)
        lines = []
        for key, value in values.items():
            name = f"neural_labs_{key}"
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value}")
        return "\n".join(lines) + "\n"


class MetricsCollector:
    def __init__(self):
        self.metrics = ServiceMetrics()
        self._lock = Lock()

    @contextmanager
    def request(self, *, prediction: bool = False) -> Iterator[None]:
        started = time.perf_counter()
        with self._lock:
            self.metrics.requests_total += 1
            if prediction:
                self.metrics.predictions_total += 1
        try:
            yield
        except Exception:
            with self._lock:
                self.metrics.errors_total += 1
            raise
        finally:
            elapsed = time.perf_counter() - started
            with self._lock:
                self.metrics.latency_seconds_sum += elapsed
                self.metrics.latency_seconds_max = max(self.metrics.latency_seconds_max, elapsed)


def configure_opentelemetry(service_name: str = "neural-labs-api") -> bool:
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    except ImportError:
        return False
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    return True
