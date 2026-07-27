from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler

from .catalog import ROOT, get_lab
from .config import merged_config
from .core.protocol import ExperimentLock, SeedPlan
from .datasets import prepare_dataset
from .distributed import cleanup_distributed, initialize_distributed, wrap_distributed
from .models import build_model
from .runtime import save_json, seed_everything


def _loss(task: str, num_classes: int | None, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if task == "regression":
        return nn.functional.mse_loss(logits.reshape(-1), targets.float().reshape(-1))
    if num_classes == 2:
        return nn.functional.binary_cross_entropy_with_logits(logits.reshape(-1), targets.float().reshape(-1))
    return nn.functional.cross_entropy(logits, targets.long())


def train_distributed(
    lab_id: str,
    *,
    strategy: str = "ddp",
    split_seed: int = 42,
    training_seed: int = 42,
    quick: bool = False,
    config_name: str = "baseline",
    output_dir: str | Path = "runs-distributed",
) -> dict[str, Any]:
    context = initialize_distributed()
    try:
        lab = get_lab(lab_id)
        if lab["task"] in {"node_classification", "reinforcement_learning", "generation", "anomaly_detection"}:
            raise ValueError("El entrenador distribuido genérico admite por ahora tareas supervisadas estándar.")
        seed_everything(training_seed + context.rank)
        bundle = prepare_dataset(lab_id, quick=quick, seed=split_seed)
        config = merged_config(lab_id, config_name, seed=training_seed, device=context.device)
        architecture = {"numpy_mlp": "mlp", "numpy_logistic": "linear", "distillation_cnn": "distillation_student"}.get(
            lab["architecture"], lab["architecture"]
        )
        model = build_model(architecture, bundle.input_shape or (1,), bundle.num_classes, bundle.metadata)
        model = wrap_distributed(model, context, strategy)
        sampler = DistributedSampler(
            bundle.train,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=True,
            seed=training_seed,
        )
        loader = DataLoader(bundle.train, batch_size=int(config["batch_size"]), sampler=sampler)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))
        epochs = int(config["quick"]["epochs"] if quick else config["epochs"])
        history: list[float] = []
        for epoch in range(epochs):
            sampler.set_epoch(epoch)
            model.train()
            losses = []
            for features, targets in loader:
                features = features.to(context.device)
                targets = targets.to(context.device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(features)
                loss = _loss(bundle.task, bundle.num_classes, logits, targets)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            history.append(sum(losses) / max(1, len(losses)))

        shared_run_id = (
            os.environ.get("NEURAL_LABS_RUN_ID")
            or os.environ.get("TORCHELASTIC_RUN_ID")
            or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        )
        run_dir = ROOT / Path(output_dir) / lab_id / shared_run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        seeds = SeedPlan(split_seed, training_seed)
        ExperimentLock.create(
            lab_id=lab_id,
            seeds=seeds,
            config_name=config_name,
            selection_metric=str(config.get("selection_metric", "loss")),
            selected_checkpoint=run_dir / "best_model.pt",
            dataset_hash=str(bundle.metadata.get("dataset_hash", "unknown")),
        ).write(run_dir)
        state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        shard_path = run_dir / f"checkpoint-rank{context.rank}.pt"
        torch.save({"state_dict": state, "rank": context.rank, "world_size": context.world_size, "strategy": strategy}, shard_path)
        if context.rank == 0:
            torch.save({"state_dict": state}, run_dir / "best_model.pt")
            save_json(
                run_dir / "distributed_manifest.json",
                {
                    "lab_id": lab_id,
                    "strategy": strategy,
                    "world_size": context.world_size,
                    "backend": context.backend,
                    "split_seed": split_seed,
                    "training_seed": training_seed,
                    "history": history,
                },
            )
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        return {
            "run_dir": str(run_dir),
            "rank": context.rank,
            "world_size": context.world_size,
            "strategy": strategy,
            "checkpoint": str(shard_path),
        }
    finally:
        cleanup_distributed()
