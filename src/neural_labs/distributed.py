from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from .runtime import save_json


@dataclass
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    backend: str
    device: str


def initialize_distributed(backend: str | None = None) -> DistributedContext:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    resolved_backend = backend or ("nccl" if torch.cuda.is_available() else "gloo")
    if world_size > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend=resolved_backend, rank=rank, world_size=world_size)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"
    return DistributedContext(rank, world_size, local_rank, resolved_backend, device)


def wrap_distributed(model: nn.Module, context: DistributedContext, strategy: str = "ddp") -> nn.Module:
    model = model.to(context.device)
    if context.world_size == 1:
        return model
    if strategy == "ddp":
        return DistributedDataParallel(model, device_ids=[context.local_rank] if context.device.startswith("cuda") else None)
    if strategy == "fsdp2":
        try:
            from torch.distributed.fsdp import fully_shard
        except ImportError as exc:
            raise RuntimeError("La instalación de PyTorch no expone FSDP2.") from exc
        fully_shard(model)
        return model
    raise ValueError("strategy debe ser 'ddp' o 'fsdp2'.")


def distributed_diagnostics(output: Path | None = None) -> dict[str, Any]:
    context = initialize_distributed()
    payload = asdict(context)
    payload["initialized"] = torch.distributed.is_initialized()
    if output:
        save_json(output, payload)
    return payload


def cleanup_distributed() -> None:
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
