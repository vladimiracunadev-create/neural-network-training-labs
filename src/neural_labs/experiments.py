from __future__ import annotations

import copy
from contextlib import nullcontext
import json
import math
import random
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .artifacts import ExperimentResult, initialize_run, save_predictions, write_model_card, write_report
from .baselines import run_baseline
from .catalog import get_dataset, get_lab
from .config import merged_config
from .core.protocol import ExperimentLock, SeedPlan
from .datasets import ArrayDataset, DataBundle, prepare_dataset
from .data_quality import quality_report
from .drift import drift_report
from .metrics import classification_metrics, expected_calibration_error, regression_metrics, save_confusion_matrix, save_history
from .models import (
    Autoencoder,
    DQN,
    DuelingDQN,
    Discriminator,
    Generator,
    SmallCNN,
    TabularMLP,
    build_model,
)
from .runtime import create_run_dir, get_device, parameter_count, save_json, seed_everything
from .profiling import profile_inference
from .tracking import create_tracker
from .statistics import bootstrap_confidence_intervals
from .slicing import subgroup_report
from .inference import persist_inference_contract


def _loader(dataset: Any, batch_size: int, shuffle: bool = False) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=torch.cuda.is_available())


def _criterion(task: str, num_classes: int | None) -> nn.Module:
    if task == "regression":
        return nn.MSELoss()
    if num_classes == 2:
        return nn.BCEWithLogitsLoss()
    return nn.CrossEntropyLoss()


def _loss(logits: torch.Tensor, targets: torch.Tensor, task: str, num_classes: int | None, criterion: nn.Module) -> torch.Tensor:
    if task == "regression":
        return criterion(logits.float().view(-1), targets.float().view(-1))
    if num_classes == 2:
        return criterion(logits.float().view(-1), targets.float().view(-1))
    return criterion(logits, targets.long())


def _predict_from_logits(logits: torch.Tensor, num_classes: int | None, task: str) -> tuple[np.ndarray, np.ndarray | None]:
    if task == "regression":
        return logits.detach().cpu().numpy().reshape(-1), None
    if num_classes == 2:
        positive = torch.sigmoid(logits.view(-1))
        probabilities = torch.stack([1 - positive, positive], dim=1)
        predictions = (positive >= 0.5).long()
    else:
        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)
    return predictions.detach().cpu().numpy(), probabilities.detach().cpu().numpy()


def evaluate(model: nn.Module, dataset: Any, device: torch.device, task: str, num_classes: int | None, batch_size: int = 256) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray | None, float]:
    model.eval()
    criterion = _criterion(task, num_classes)
    targets_all: list[np.ndarray] = []
    predictions_all: list[np.ndarray] = []
    probabilities_all: list[np.ndarray] = []
    losses: list[float] = []
    with torch.no_grad():
        for features, targets in _loader(dataset, batch_size, False):
            features, targets = features.to(device), targets.to(device)
            logits = model(features)
            batch_loss = _loss(logits, targets, task, num_classes, criterion)
            predictions, probabilities = _predict_from_logits(logits, num_classes, task)
            targets_all.append(targets.detach().cpu().numpy().reshape(-1))
            predictions_all.append(predictions)
            if probabilities is not None:
                probabilities_all.append(probabilities)
            losses.append(float(batch_loss.item()))
    y_true = np.concatenate(targets_all)
    y_pred = np.concatenate(predictions_all)
    probabilities = np.concatenate(probabilities_all) if probabilities_all else None
    metrics = regression_metrics(y_true, y_pred) if task == "regression" else classification_metrics(y_true, y_pred, probabilities)
    metrics["loss"] = float(np.mean(losses))
    return metrics, y_true, y_pred, probabilities, metrics["loss"]


def _selection_value(metrics: dict[str, float], selection_metric: str) -> tuple[float, bool]:
    minimize = selection_metric in {"loss", "rmse", "mae", "mape", "brier", "ece"}
    value = float(metrics.get(selection_metric, metrics.get("loss", math.inf if minimize else -math.inf)))
    return value, minimize


def _train_torch_model(
    model: nn.Module,
    bundle: DataBundle,
    config: dict[str, Any],
    device: torch.device,
    run_dir: Path,
    *,
    optimizer_name: str = "adamw",
    activation_label: str | None = None,
    evaluate_test: bool = True,
) -> tuple[nn.Module, dict[str, list[float]], dict[str, float], np.ndarray, np.ndarray, np.ndarray | None]:
    model = model.to(device)
    criterion = _criterion(bundle.task, bundle.num_classes)
    lr = float(config["learning_rate"])
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)
    train_loader = DataLoader(
        bundle.train,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
        persistent_workers=bool(config.get("num_workers", 0)),
    )
    amp_enabled = bool(config.get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    autocast_context = (
        lambda: torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled)
        if device.type == "cuda"
        else nullcontext()
    )
    epochs = int(config["quick"]["epochs"] if config.get("_quick") else config["epochs"])
    patience = min(int(config["patience"]), max(1, epochs))
    selection_metric = str(config["selection_metric"])
    best_state = copy.deepcopy(model.state_dict())
    best_value: float | None = None
    stale = 0
    history: dict[str, list[float]] = {"train_loss": [], "validation_loss": [], "validation_metric": [], "learning_rate": []}
    for _epoch in range(epochs):
        model.train()
        train_losses: list[float] = []
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context():
                logits = model(features)
                loss = _loss(logits, targets, bundle.task, bundle.num_classes, criterion)
            if not torch.isfinite(loss):
                raise FloatingPointError("La pérdida dejó de ser finita.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(config.get("gradient_clip_norm", 5.0)))
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.item()))
        validation_metrics, _, _, _, validation_loss = evaluate(model, bundle.validation, device, bundle.task, bundle.num_classes, int(config["batch_size"]) * 2)
        value, minimize = _selection_value(validation_metrics, selection_metric)
        improved = best_value is None or (value < best_value if minimize else value > best_value)
        if improved:
            best_value = value
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        scheduler.step(validation_loss)
        history["train_loss"].append(float(np.mean(train_losses)))
        history["validation_loss"].append(float(validation_loss))
        history["validation_metric"].append(float(value))
        history["learning_rate"].append(float(optimizer.param_groups[0]["lr"]))
        if stale >= patience:
            break
    torch.save({"state_dict": model.state_dict()}, run_dir / "last_model.pt")
    model.load_state_dict(best_state)
    torch.save({"state_dict": model.state_dict()}, run_dir / "best_model.pt")
    if not evaluate_test:
        provisional = {"selection_metric": selection_metric, "selection_value": best_value, "parameters": parameter_count(model)}
        if activation_label:
            provisional["variant"] = activation_label
        return model, history, provisional, np.array([]), np.array([]), None
    test_metrics, y_true, y_pred, probabilities, _ = evaluate(model, bundle.test, device, bundle.task, bundle.num_classes, int(config["batch_size"]) * 2)
    test_metrics["parameters"] = parameter_count(model)
    if activation_label:
        test_metrics["variant"] = activation_label
    return model, history, test_metrics, y_true, y_pred, probabilities


def _metadata_for_model(bundle: DataBundle) -> dict[str, Any]:
    vocabulary = bundle.metadata.get("vocabulary")
    return {"vocab_size": len(vocabulary) if vocabulary else None}


def _standard_model(lab: dict[str, Any], bundle: DataBundle, config: dict[str, Any]) -> nn.Module:
    kwargs: dict[str, Any] = {}
    if bundle.task == "regression":
        kwargs["regression"] = True
    if lab["architecture"] in {"regularization_comparison"}:
        kwargs.update(dropout=0.35, batch_norm=True)
    return build_model(lab["architecture"], bundle.input_shape or (1,), bundle.num_classes, _metadata_for_model(bundle), **kwargs)


def _run_numpy_logistic(bundle: DataBundle, config: dict[str, Any], run_dir: Path, freeze: Callable[[], Path]):
    x_train = bundle.train.features.numpy(); y_train = bundle.train.targets.numpy().astype(np.float64)
    x_val = bundle.validation.features.numpy(); y_val = bundle.validation.targets.numpy().astype(np.float64)
    weights = np.zeros(x_train.shape[1], dtype=np.float64); bias = 0.0
    lr = float(config["learning_rate"] if config["learning_rate"] > 0.01 else 0.05)
    epochs = int(config["quick"]["epochs"] if config.get("_quick") else max(config["epochs"], 100))
    history = {"train_loss": [], "validation_loss": [], "validation_metric": [], "learning_rate": []}
    best = (float("inf"), weights.copy(), bias)
    for _ in range(epochs):
        logits = x_train @ weights + bias
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
        error = probs - y_train
        weights -= lr * (x_train.T @ error / len(x_train))
        bias -= lr * float(error.mean())
        train_loss = float(-np.mean(y_train * np.log(probs + 1e-8) + (1-y_train) * np.log(1-probs + 1e-8)))
        val_probs = 1.0 / (1.0 + np.exp(-np.clip(x_val @ weights + bias, -40, 40)))
        val_loss = float(-np.mean(y_val * np.log(val_probs + 1e-8) + (1-y_val) * np.log(1-val_probs + 1e-8)))
        val_pred = (val_probs >= 0.5).astype(int)
        val_f1 = classification_metrics(y_val.astype(int), val_pred, np.stack([1-val_probs, val_probs], axis=1))["f1"]
        history["train_loss"].append(train_loss); history["validation_loss"].append(val_loss); history["validation_metric"].append(val_f1); history["learning_rate"].append(lr)
        if val_loss < best[0]: best = (val_loss, weights.copy(), bias)
    _, weights, bias = best
    np.savez(run_dir / "best_model.npz", weights=weights, bias=bias)
    np.savez(run_dir / "last_model.npz", weights=weights, bias=bias)
    freeze()
    x_test = bundle.test.features.numpy(); y_test = bundle.test.targets.numpy().astype(np.int64)
    test_probs = 1.0 / (1.0 + np.exp(-np.clip(x_test @ weights + bias, -40, 40)))
    test_pred = (test_probs >= 0.5).astype(int)
    probabilities = np.stack([1-test_probs, test_probs], axis=1)
    metrics = classification_metrics(y_test, test_pred, probabilities)
    metrics["parameters"] = {"total": int(weights.size + 1), "trainable": int(weights.size + 1)}
    return None, history, metrics, y_test, test_pred, probabilities


def _run_numpy_mlp(bundle: DataBundle, config: dict[str, Any], run_dir: Path, freeze: Callable[[], Path]):
    x_train = bundle.train.features.numpy().astype(np.float64); y_train = bundle.train.targets.numpy().astype(int)
    x_val = bundle.validation.features.numpy().astype(np.float64); y_val = bundle.validation.targets.numpy().astype(int)
    classes = int(bundle.num_classes or 3); hidden = 16; rng = np.random.default_rng(int(config["seed"]))
    w1 = rng.normal(0, 0.1, (x_train.shape[1], hidden)); b1 = np.zeros(hidden); w2 = rng.normal(0, 0.1, (hidden, classes)); b2 = np.zeros(classes)
    lr = 0.03; epochs = int(config["quick"]["epochs"] if config.get("_quick") else 250)
    history = {"train_loss": [], "validation_loss": [], "validation_metric": [], "learning_rate": []}
    best = (float("inf"), None)
    def forward(x):
        h = np.tanh(x @ w1 + b1); logits = h @ w2 + b2; logits -= logits.max(axis=1, keepdims=True); p = np.exp(logits); p /= p.sum(axis=1, keepdims=True); return h, p
    for _ in range(epochs):
        h, p = forward(x_train); onehot = np.eye(classes)[y_train]; loss = -np.mean(np.log(p[np.arange(len(y_train)), y_train] + 1e-8))
        dlogits = (p - onehot) / len(x_train); dw2 = h.T @ dlogits; db2 = dlogits.sum(0); dh = dlogits @ w2.T * (1-h*h); dw1 = x_train.T @ dh; db1 = dh.sum(0)
        w1 -= lr*dw1; b1 -= lr*db1; w2 -= lr*dw2; b2 -= lr*db2
        _, vp = forward(x_val); vl = -np.mean(np.log(vp[np.arange(len(y_val)), y_val] + 1e-8)); vpred = vp.argmax(1); vf1 = classification_metrics(y_val, vpred)["macro_f1"]
        history["train_loss"].append(float(loss)); history["validation_loss"].append(float(vl)); history["validation_metric"].append(float(vf1)); history["learning_rate"].append(lr)
        if vl < best[0]: best = (float(vl), (w1.copy(), b1.copy(), w2.copy(), b2.copy()))
    w1,b1,w2,b2 = best[1]
    np.savez(run_dir/"best_model.npz",w1=w1,b1=b1,w2=w2,b2=b2)
    freeze()
    x_test = bundle.test.features.numpy().astype(np.float64); y_test = bundle.test.targets.numpy().astype(int)
    _, probs = forward(x_test); pred = probs.argmax(1); metrics = classification_metrics(y_test,pred,probs); metrics["parameters"]={"total":int(w1.size+b1.size+w2.size+b2.size),"trainable":int(w1.size+b1.size+w2.size+b2.size)}
    return None, history, metrics, y_test, pred, probs


def _run_autoencoder(
    bundle: DataBundle,
    config: dict[str, Any],
    device: torch.device,
    run_dir: Path,
    freeze: Callable[[], Path],
):
    train_x = bundle.train.features
    train_y = bundle.train.targets
    normal = train_x[train_y == 0]
    model = Autoencoder(normal.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))
    loader = _loader(TensorDataset(normal), int(config["batch_size"]), True)
    epochs = int(config["quick"]["epochs"] if config.get("_quick") else config["epochs"])
    history = {"train_loss": [], "validation_loss": [], "validation_metric": [], "learning_rate": []}
    best_score = -math.inf
    best_state = copy.deepcopy(model.state_dict())
    best_threshold = 0.0
    best_center = 0.0
    best_scale = 1.0

    for _ in range(epochs):
        model.train()
        losses: list[float] = []
        for (features,) in loader:
            features = features.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(features)
            loss = nn.functional.mse_loss(reconstruction, features)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            validation_x = bundle.validation.features.to(device)
            errors = ((model(validation_x) - validation_x) ** 2).mean(1).cpu().numpy()
        validation_y = bundle.validation.targets.numpy()
        normal_errors = errors[validation_y == 0]
        threshold = float(np.quantile(normal_errors, 0.99)) if len(normal_errors) else float(np.quantile(errors, 0.95))
        scale = float(max(np.std(normal_errors), 1e-8)) if len(normal_errors) else float(max(np.std(errors), 1e-8))
        scores = 1.0 / (1.0 + np.exp(-np.clip((errors - threshold) / scale, -40, 40)))
        predictions = (errors >= threshold).astype(int)
        probabilities = np.stack([1.0 - scores, scores], axis=1)
        validation_metrics = classification_metrics(validation_y, predictions, probabilities)
        selection_score = float(validation_metrics.get("pr_auc", validation_metrics["f1"]))
        history["train_loss"].append(float(np.mean(losses)))
        history["validation_loss"].append(float(errors.mean()))
        history["validation_metric"].append(selection_score)
        history["learning_rate"].append(float(optimizer.param_groups[0]["lr"]))
        if selection_score > best_score:
            best_score = selection_score
            best_state = copy.deepcopy(model.state_dict())
            best_threshold = threshold
            best_center = threshold
            best_scale = scale

    torch.save({"state_dict": model.state_dict()}, run_dir / "last_model.pt")
    model.load_state_dict(best_state)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "threshold": best_threshold,
            "score_center": best_center,
            "score_scale": best_scale,
        },
        run_dir / "best_model.pt",
    )
    freeze()
    model.eval()
    with torch.no_grad():
        test_x = bundle.test.features.to(device)
        test_errors = ((model(test_x) - test_x) ** 2).mean(1).cpu().numpy()
    y_true = bundle.test.targets.numpy()
    y_pred = (test_errors >= best_threshold).astype(int)
    scores = 1.0 / (1.0 + np.exp(-np.clip((test_errors - best_center) / best_scale, -40, 40)))
    probabilities = np.stack([1.0 - scores, scores], axis=1)
    metrics = classification_metrics(y_true, y_pred, probabilities)
    metrics.update(
        threshold=best_threshold,
        best_validation_pr_auc=best_score,
        parameters=parameter_count(model),
    )
    return model, history, metrics, y_true, y_pred, probabilities

def _run_gan(
    bundle: DataBundle,
    config: dict[str, Any],
    device: torch.device,
    run_dir: Path,
    freeze: Callable[[], Path],
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    latent = 64
    generator = Generator(latent).to(device)
    discriminator = Discriminator().to(device)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()
    epochs = int(config["quick"]["epochs"] if config.get("_quick") else config["epochs"])
    history = {"generator_loss": [], "discriminator_loss": []}

    for _ in range(epochs):
        generator_losses: list[float] = []
        discriminator_losses: list[float] = []
        for real, _targets in _loader(bundle.train, int(config["batch_size"]), True):
            real = real.to(device)
            batch = len(real)
            noise = torch.randn(batch, latent, device=device)
            fake = generator(noise)

            optimizer_d.zero_grad(set_to_none=True)
            discriminator_loss = criterion(discriminator(real), torch.ones(batch, device=device))
            discriminator_loss += criterion(discriminator(fake.detach()), torch.zeros(batch, device=device))
            discriminator_loss.backward()
            optimizer_d.step()

            optimizer_g.zero_grad(set_to_none=True)
            generator_loss = criterion(discriminator(fake), torch.ones(batch, device=device))
            generator_loss.backward()
            optimizer_g.step()
            generator_losses.append(float(generator_loss.item()))
            discriminator_losses.append(float(discriminator_loss.item()))

        history["generator_loss"].append(float(np.mean(generator_losses)))
        history["discriminator_loss"].append(float(np.mean(discriminator_losses)))

    torch.save(
        {"generator": generator.state_dict(), "discriminator": discriminator.state_dict()},
        run_dir / "best_model.pt",
    )
    with torch.no_grad():
        generated = generator(torch.randn(256, latent, device=device)).cpu()
    freeze()
    real_batch = next(iter(_loader(bundle.test, 256, False)))[0].cpu()
    generated_flat = generated.flatten(1)
    real_flat = real_batch[: len(generated_flat)].flatten(1)

    # Kernel MMD and nearest-neighbour coverage are explicit proxies suitable for
    # Fashion-MNIST. They are not presented as Inception FID.
    combined = torch.cat([generated_flat, real_flat], dim=0)
    distances = torch.cdist(combined, combined).pow(2)
    positive = distances[distances > 0]
    bandwidth = positive.median().clamp_min(1e-6)
    kernel = torch.exp(-distances / (2.0 * bandwidth))
    n_generated = len(generated_flat)
    k_gg = kernel[:n_generated, :n_generated]
    k_rr = kernel[n_generated:, n_generated:]
    k_gr = kernel[:n_generated, n_generated:]
    mmd_rbf = float((k_gg.mean() + k_rr.mean() - 2.0 * k_gr.mean()).item())
    diversity = float(torch.pdist(generated_flat).mean().item())
    nearest_real_distance = float(torch.cdist(real_flat, generated_flat).min(dim=1).values.mean().item())
    moment_distance = float(
        abs(generated_flat.mean().item() - real_flat.mean().item())
        + abs(generated_flat.std().item() - real_flat.std().item())
    )

    figure, axes = plt.subplots(4, 4, figsize=(6, 6))
    for axis, image in zip(axes.ravel(), generated[:16]):
        axis.imshow(image.squeeze().numpy(), cmap="gray")
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(run_dir / "generated_samples.png", dpi=160)
    plt.close(figure)

    metrics = {
        "generator_loss": history["generator_loss"][-1],
        "discriminator_loss": history["discriminator_loss"][-1],
        "mmd_rbf": mmd_rbf,
        "diversity": diversity,
        "nearest_real_distance": nearest_real_distance,
        "moment_distance": moment_distance,
        "generator_parameters": parameter_count(generator),
        "discriminator_parameters": parameter_count(discriminator),
    }
    return generator, history, metrics, np.array([]), np.array([]), None

def _run_gnn(bundle: DataBundle, config: dict[str, Any], device: torch.device, run_dir: Path, freeze: Callable[[], Path]):
    from .domains.graphs.models import build_graph_model

    graph = bundle.train.to(device)
    requested = config.get("graph_models", ["gcn", "graphsage", "gat"])
    if config.get("_quick"):
        requested = requested[:1]
    epochs = int(config["quick"]["epochs"] if config.get("_quick") else max(config["epochs"], 100))
    candidates: list[dict[str, Any]] = []
    best_model = None
    best_kind = None
    best_score = -math.inf
    best_history: dict[str, list[float]] = {}
    for kind in requested:
        model = build_graph_model(kind, graph.num_node_features, 64, int(bundle.num_classes)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        history = {"train_loss": [], "validation_loss": [], "validation_metric": [], "learning_rate": []}
        local_best = (-math.inf, None)
        for _ in range(epochs):
            model.train()
            optimizer.zero_grad()
            logits = model(graph.x, graph.edge_index)
            loss = nn.functional.cross_entropy(logits[graph.train_mask], graph.y[graph.train_mask])
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                probabilities = torch.softmax(model(graph.x, graph.edge_index), dim=1)
                predictions = probabilities.argmax(dim=1)
                validation = classification_metrics(
                    graph.y[graph.val_mask].cpu().numpy(),
                    predictions[graph.val_mask].cpu().numpy(),
                    probabilities[graph.val_mask].cpu().numpy(),
                )
            history["train_loss"].append(float(loss.item()))
            history["validation_loss"].append(float(1 - validation["accuracy"]))
            history["validation_metric"].append(float(validation["macro_f1"]))
            history["learning_rate"].append(float(optimizer.param_groups[0]["lr"]))
            if validation["macro_f1"] > local_best[0]:
                local_best = (validation["macro_f1"], copy.deepcopy(model.state_dict()))
        model.load_state_dict(local_best[1])
        candidates.append({"architecture": kind, "validation_macro_f1": float(local_best[0]), "parameters": parameter_count(model)})
        if local_best[0] > best_score:
            best_score = float(local_best[0])
            best_model = model
            best_kind = kind
            best_history = history
    if best_model is None:
        raise RuntimeError("No se entrenó ningún modelo de grafo.")
    torch.save({"state_dict": best_model.state_dict(), "graph_architecture": best_kind}, run_dir / "best_model.pt")
    freeze()
    best_model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(best_model(graph.x, graph.edge_index), dim=1)
        predictions = probabilities.argmax(dim=1)
    y_true = graph.y[graph.test_mask].cpu().numpy()
    y_pred = predictions[graph.test_mask].cpu().numpy()
    test_probabilities = probabilities[graph.test_mask].cpu().numpy()
    metrics = classification_metrics(y_true, y_pred, test_probabilities)
    metrics.update({"parameters": parameter_count(best_model), "selected_graph_architecture": best_kind, "validation_macro_f1": best_score})
    save_json(run_dir / "graph_model_comparison.json", {"candidates": candidates, "selected": best_kind})
    return best_model, best_history, metrics, y_true, y_pred, test_probabilities

def _inventory_episode(
    model: nn.Module,
    demand: np.ndarray,
    device: torch.device,
    *,
    epsilon: float = 0.0,
    start: int = 0,
    horizon: int | None = None,
    replay: deque | None = None,
) -> dict[str, float]:
    demand = np.asarray(demand, dtype=np.float32)
    scale = max(float(np.quantile(demand, 0.75)), 1.0)
    order_levels = np.asarray([0.0, 0.5 * scale, 1.0 * scale, 2.0 * scale], dtype=np.float32)
    inventory = float(scale)
    end = min(len(demand), start + (horizon or len(demand)))
    total_reward = 0.0
    stockouts = 0
    holding_total = 0.0
    served = 0.0
    total_demand = 0.0
    transitions = 0

    def make_state(index: int, current_inventory: float) -> np.ndarray:
        recent_start = max(0, index - 7)
        recent = demand[recent_start:index]
        recent_mean = float(recent.mean()) if len(recent) else float(demand[:1].mean())
        return np.asarray([
            current_inventory / (4.0 * scale),
            recent_mean / scale,
            math.sin(2.0 * math.pi * index / 7.0),
            math.cos(2.0 * math.pi * index / 7.0),
        ], dtype=np.float32)

    state = make_state(start, inventory)
    for index in range(start, end):
        if random.random() < epsilon:
            action = random.randrange(len(order_levels))
        else:
            with torch.no_grad():
                tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                action = int(model(tensor).argmax(dim=1).item())
        order = float(order_levels[action])
        available = inventory + order
        actual_demand = float(demand[index])
        sales = min(available, actual_demand)
        shortage = max(0.0, actual_demand - available)
        inventory = max(0.0, available - actual_demand)
        reward = sales - 0.20 * order - 0.05 * inventory - 1.50 * shortage
        next_state = make_state(index + 1, inventory)
        done = index == end - 1
        if replay is not None:
            replay.append((state, action, reward, next_state, float(done)))
        state = next_state
        total_reward += reward
        stockouts += int(shortage > 0)
        holding_total += inventory
        served += sales
        total_demand += actual_demand
        transitions += 1
    return {
        "mean_return": float(total_reward),
        "stockout_rate": float(stockouts / max(1, transitions)),
        "holding_cost": float(0.05 * holding_total),
        "service_level": float(served / max(total_demand, 1e-8)),
        "episode_length": float(transitions),
    }


def _run_dqn(bundle: DataBundle, config: dict[str, Any], device: torch.device, run_dir: Path, freeze: Callable[[], Path]):
    train_demand = np.asarray(bundle.raw["train_demand"], dtype=np.float32)
    validation_demand = np.asarray(bundle.raw["validation_demand"], dtype=np.float32)
    test_demand = np.asarray(bundle.raw["test_demand"], dtype=np.float32)
    policy = DuelingDQN(state_dim=4, actions=4).to(device)
    target = DuelingDQN(state_dim=4, actions=4).to(device)
    target.load_state_dict(policy.state_dict())
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    replay: deque = deque(maxlen=50000)
    batch_size = 64
    gamma = 0.98
    training_steps = 300 if config.get("_quick") else 5000
    horizon = min(30, len(train_demand))
    best_state = copy.deepcopy(policy.state_dict())
    best_validation = -math.inf
    returns: list[float] = []
    losses: list[float] = []

    for step in range(training_steps):
        maximum_start = max(0, len(train_demand) - horizon)
        start = random.randint(0, maximum_start) if maximum_start else 0
        epsilon = max(0.05, 1.0 - step / max(1, training_steps * 0.8))
        episode = _inventory_episode(policy, train_demand, device, epsilon=epsilon, start=start, horizon=horizon, replay=replay)
        returns.append(episode["mean_return"])
        if len(replay) >= batch_size:
            sample = random.sample(replay, batch_size)
            states = torch.tensor(np.asarray([item[0] for item in sample]), dtype=torch.float32, device=device)
            actions = torch.tensor([item[1] for item in sample], dtype=torch.long, device=device)
            rewards = torch.tensor([item[2] for item in sample], dtype=torch.float32, device=device)
            next_states = torch.tensor(np.asarray([item[3] for item in sample]), dtype=torch.float32, device=device)
            dones = torch.tensor([item[4] for item in sample], dtype=torch.float32, device=device)
            q_values = policy(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_actions = policy(next_states).argmax(dim=1, keepdim=True)
                next_values = target(next_states).gather(1, next_actions).squeeze(1)
                targets = rewards + gamma * (1.0 - dones) * next_values
            loss = nn.functional.smooth_l1_loss(q_values, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.item()))
        if step % 50 == 0:
            target.load_state_dict(policy.state_dict())
            validation = _inventory_episode(policy, validation_demand, device)
            if validation["mean_return"] > best_validation:
                best_validation = validation["mean_return"]
                best_state = copy.deepcopy(policy.state_dict())

    policy.load_state_dict(best_state)
    torch.save({"state_dict": policy.state_dict(), "source": "UCI Online Retail demand trace"}, run_dir / "best_model.pt")
    freeze()
    test_metrics = _inventory_episode(policy, test_demand, device)
    test_metrics["validation_return"] = float(best_validation)
    test_metrics["parameters"] = parameter_count(policy)
    test_metrics["algorithm"] = "double_dueling_dqn"
    history = {
        "return": returns,
        "loss": losses + [np.nan] * max(0, len(returns) - len(losses)),
    }
    return policy, history, test_metrics, np.array([]), np.array([]), None

def _run_transfer(bundle: DataBundle, config: dict[str, Any], device: torch.device, run_dir: Path, freeze: Callable[[], Path]):
    try:
        from torchvision.models import ResNet18_Weights, resnet18
    except ImportError as exc:
        raise RuntimeError('Instale el extra de visión: pip install -e ".[vision]"') from exc
    baseline_dir = run_dir / "from_scratch_baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    scratch = SmallCNN(bundle.input_shape[0], int(bundle.num_classes), dropout=0.3)
    scratch, scratch_history, scratch_validation, _, _, _ = _train_torch_model(
        scratch, bundle, config, device, baseline_dir, evaluate_test=False
    )
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, int(bundle.num_classes))
    model, history, transfer_validation, _, _, _ = _train_torch_model(
        model, bundle, config, device, run_dir, evaluate_test=False
    )
    freeze()
    metrics, y_true, y_pred, probs, _ = evaluate(model, bundle.test, device, bundle.task, bundle.num_classes)
    scratch_metrics, _, _, _, _ = evaluate(scratch, bundle.test, device, bundle.task, bundle.num_classes)
    metrics["from_scratch_accuracy"] = scratch_metrics.get("accuracy")
    metrics["transfer_gain"] = float(metrics.get("accuracy", 0.0) - scratch_metrics.get("accuracy", 0.0))
    metrics["validation_selection"] = transfer_validation.get("selection_value")
    metrics["parameters"] = parameter_count(model)
    save_json(run_dir / "transfer_comparison.json", {
        "selection_policy": "pretrained model selected before test",
        "from_scratch_validation": scratch_validation,
        "transfer_validation": transfer_validation,
        "from_scratch_test": scratch_metrics,
        "pretrained_resnet18_test": metrics,
    })
    return model, history, metrics, y_true, y_pred, probs


def _run_distillation(bundle: DataBundle, config: dict[str, Any], device: torch.device, run_dir: Path, freeze: Callable[[], Path]):
    teacher=build_model("distillation_teacher", bundle.input_shape, bundle.num_classes, {}).to(device); teacher_cfg=dict(config); teacher_cfg["epochs"]=max(4,int(config["epochs"])); teacher,teacher_history,teacher_metrics,_,_,_=_train_torch_model(teacher,bundle,teacher_cfg,device,run_dir/"teacher") if False else (teacher,None,None,None,None,None)
    # Dedicated loop avoids nested artifact paths and trains teacher then student.
    def train_supervised(model,epochs):
        opt=torch.optim.AdamW(model.parameters(),lr=1e-3); loader=_loader(bundle.train,int(config["batch_size"]),True)
        for _ in range(epochs):
            model.train()
            for x,y in loader:
                x,y=x.to(device),y.to(device); opt.zero_grad(); loss=nn.functional.cross_entropy(model(x),y.long()); loss.backward(); opt.step()
        return model
    epochs=2 if config.get("_quick") else int(config["epochs"])
    teacher=train_supervised(teacher,max(epochs,4)); student=build_model("distillation_student", bundle.input_shape, bundle.num_classes, {}).to(device); opt=torch.optim.AdamW(student.parameters(),lr=1e-3); loader=_loader(bundle.train,int(config["batch_size"]),True); temperature=3.0; alpha=0.5; history={"train_loss":[],"validation_loss":[],"validation_metric":[],"learning_rate":[]}; best_student=copy.deepcopy(student.state_dict()); best_validation=-math.inf
    for _ in range(epochs):
        student.train(); losses=[]; teacher.eval()
        for x,y in loader:
            x,y=x.to(device),y.to(device); opt.zero_grad(); s=student(x)
            with torch.no_grad(): t=teacher(x)
            hard=nn.functional.cross_entropy(s,y.long()); soft=nn.functional.kl_div(nn.functional.log_softmax(s/temperature,1),nn.functional.softmax(t/temperature,1),reduction="batchmean")*(temperature**2); loss=alpha*hard+(1-alpha)*soft; loss.backward(); opt.step(); losses.append(loss.item())
        val,_,_,_,vl=evaluate(student,bundle.validation,device,bundle.task,bundle.num_classes)
        history["train_loss"].append(float(np.mean(losses))); history["validation_loss"].append(vl); history["validation_metric"].append(val["macro_f1"]); history["learning_rate"].append(float(opt.param_groups[0]["lr"]))
        if val["macro_f1"] > best_validation:
            best_validation=float(val["macro_f1"]); best_student=copy.deepcopy(student.state_dict())
    student.load_state_dict(best_student)
    torch.save({"teacher":teacher.state_dict(),"student":student.state_dict()},run_dir/"best_model.pt"); freeze(); metrics,y_true,y_pred,probs,_=evaluate(student,bundle.test,device,bundle.task,bundle.num_classes); teacher_test,_,_,_,_=evaluate(teacher,bundle.test,device,bundle.task,bundle.num_classes); metrics["teacher_accuracy"]=teacher_test["accuracy"]; metrics["student_parameters"]=parameter_count(student); metrics["teacher_parameters"]=parameter_count(teacher); metrics["best_validation_macro_f1"]=best_validation
    start=time.perf_counter();
    with torch.no_grad(): student(next(iter(_loader(bundle.test,64,False)))[0].to(device))
    metrics["latency_ms"]=float((time.perf_counter()-start)*1000)
    return student,history,metrics,y_true,y_pred,probs


def _run_federated(bundle: DataBundle, config: dict[str, Any], device: torch.device, run_dir: Path, freeze: Callable[[], Path]):
    global_model=TabularMLP(bundle.input_shape[0],bundle.num_classes,hidden=(128,64),dropout=0.1).to(device); subjects=np.asarray(bundle.train.extra.get("subjects")); unique=np.unique(subjects); rounds=2 if config.get("_quick") else 10; history={"validation_metric":[],"validation_loss":[]}
    for _round in range(rounds):
        states=[]; weights=[]
        for subject in unique:
            indices=np.where(subjects==subject)[0]
            if len(indices)<4: continue
            local=copy.deepcopy(global_model); optimizer=torch.optim.SGD(local.parameters(),lr=0.02,momentum=0.9); data=TensorDataset(bundle.train.features[indices],bundle.train.targets[indices]); local.train()
            for x,y in _loader(data,64,True):
                x,y=x.to(device),y.to(device); optimizer.zero_grad(); loss=nn.functional.cross_entropy(local(x),y.long()); loss.backward(); optimizer.step()
            states.append({k:v.detach().cpu() for k,v in local.state_dict().items()}); weights.append(len(indices))
        total=sum(weights); averaged={key:sum(state[key]*weight/total for state,weight in zip(states,weights)) for key in states[0]}; global_model.load_state_dict(averaged); val,_,_,_,vl=evaluate(global_model,bundle.validation,device,bundle.task,bundle.num_classes); history["validation_metric"].append(val["macro_f1"]); history["validation_loss"].append(vl)
    torch.save({"state_dict":global_model.state_dict()},run_dir/"best_model.pt"); freeze(); metrics,y_true,y_pred,probs,_=evaluate(global_model,bundle.test,device,bundle.task,bundle.num_classes); client_scores=[]; test_subjects=np.asarray(bundle.test.extra.get("subjects"))
    for subject in np.unique(test_subjects):
        idx=np.where(test_subjects==subject)[0]; subset=TensorDataset(bundle.test.features[idx],bundle.test.targets[idx]); score,_,_,_,_=evaluate(global_model,subset,device,bundle.task,bundle.num_classes); client_scores.append(score["accuracy"])
    metrics["client_accuracy_std"]=float(np.std(client_scores)); metrics["parameters"]=parameter_count(global_model)
    return global_model,history,metrics,y_true,y_pred,probs


def _run_variant_comparison(lab: dict[str,Any], bundle: DataBundle, config: dict[str,Any], device: torch.device, run_dir: Path, freeze: Callable[[], Path]):
    if lab["id"]=="17_activations_and_losses":
        variants=[("relu",{"activation":"relu"}),("gelu",{"activation":"gelu"}),("tanh",{"activation":"tanh"})]
    elif lab["id"]=="18_optimizers_and_schedulers":
        variants=[("sgd",{}),("adam",{}),("adamw",{})]
    elif lab["id"]=="19_regularization_dropout_batchnorm":
        variants=[("none",{"dropout":0.0,"batch_norm":False}),("dropout",{"dropout":0.4,"batch_norm":False}),("dropout_batchnorm",{"dropout":0.3,"batch_norm":True})]
    else:
        variants=[]
    results=[]
    best=None
    selection_metric=str(config["selection_metric"])
    for name,kwargs in variants:
        model=TabularMLP(bundle.input_shape[0],bundle.num_classes,hidden=(128,64),regression=bundle.task=="regression",**kwargs)
        optimizer=name if lab["id"]=="18_optimizers_and_schedulers" else "adamw"
        variant_dir=run_dir/f"variant-{name}"
        variant_dir.mkdir(parents=True,exist_ok=True)
        trained,history,provisional,_,_,_=_train_torch_model(model,bundle,config,device,variant_dir,optimizer_name=optimizer,activation_label=name,evaluate_test=False)
        validation_values=history["validation_metric"]
        if selection_metric in {"loss","rmse","mae","mape","brier","ece"}:
            score=min(validation_values)
            better=best is None or score<best[0]
        else:
            score=max(validation_values)
            better=best is None or score>best[0]
        results.append({"variant":name,"validation_selection":float(score),"epochs":len(history["train_loss"]),"parameters":provisional["parameters"]})
        if better:
            best=(float(score),trained,history,name)
    save_json(run_dir/"variant_comparison.json",results)
    _,model,history,name=best
    torch.save({"state_dict": model.state_dict()}, run_dir / "best_model.pt")
    freeze()
    metrics,y_true,y_pred,probs,_=evaluate(model,bundle.test,device,bundle.task,bundle.num_classes,int(config["batch_size"])*2)
    metrics["best_variant"] = name
    metrics["parameters"] = parameter_count(model)
    if lab["id"] == "19_regularization_dropout_batchnorm":
        train_metrics, _, _, _, _ = evaluate(
            model,
            bundle.train,
            device,
            bundle.task,
            bundle.num_classes,
            int(config["batch_size"]) * 2,
        )
        metrics["generalization_gap"] = float(train_metrics["accuracy"] - metrics["accuracy"])
    return model, history, metrics, y_true, y_pred, probs


def _run_hyperparameter_search(bundle: DataBundle, config: dict[str, Any], device: torch.device, run_dir: Path, freeze: Callable[[], Path]):
    trials_count = 3 if config.get("_quick") else 20
    epochs = 2 if config.get("_quick") else 8
    selection_metric = str(config["selection_metric"])

    def train_candidate(params: dict[str, Any]) -> tuple[float, nn.Module]:
        model = TabularMLP(
            bundle.input_shape[0],
            bundle.num_classes,
            hidden=tuple(params["hidden"]),
            dropout=float(params["dropout"]),
            activation=str(params["activation"]),
            batch_norm=bool(params["batch_norm"]),
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(params["learning_rate"]), weight_decay=float(params["weight_decay"]))
        criterion = _criterion(bundle.task, bundle.num_classes)
        for _ in range(epochs):
            model.train()
            for x, y in _loader(bundle.train, int(config["batch_size"]), True):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = _loss(model(x), y, bundle.task, bundle.num_classes, criterion)
                loss.backward()
                optimizer.step()
        validation, _, _, _, _ = evaluate(model, bundle.validation, device, bundle.task, bundle.num_classes)
        return float(validation[selection_metric]), model

    candidates: list[dict[str, Any]] = []
    best: tuple[float, dict[str, Any]] | None = None
    minimize = selection_metric in {"loss","rmse","mae","mape","brier","ece"}
    try:
        import optuna
        def objective(trial):
            widths = [trial.suggest_categorical("width", [64, 128, 256]), trial.suggest_categorical("second_width", [32, 64, 128])]
            params = {
                "hidden": widths,
                "dropout": trial.suggest_float("dropout", 0.0, 0.5),
                "activation": trial.suggest_categorical("activation", ["relu", "gelu", "silu"]),
                "batch_norm": trial.suggest_categorical("batch_norm", [False, True]),
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            }
            score, _ = train_candidate(params)
            candidates.append({"trial": trial.number, "score": score, **params})
            return score
        study = optuna.create_study(direction="minimize" if minimize else "maximize")
        study.optimize(objective, n_trials=trials_count, show_progress_bar=False)
        best_params = dict(study.best_params)
        best_params["hidden"] = [best_params.pop("width"), best_params.pop("second_width")]
        best_score = float(study.best_value)
    except ImportError:
        grid = [
            {"hidden":[64,32],"dropout":0.1,"activation":"relu","batch_norm":False,"learning_rate":1e-3,"weight_decay":1e-4},
            {"hidden":[128,64],"dropout":0.25,"activation":"gelu","batch_norm":True,"learning_rate":5e-4,"weight_decay":1e-4},
            {"hidden":[256,128],"dropout":0.35,"activation":"silu","batch_norm":True,"learning_rate":3e-4,"weight_decay":1e-3},
        ][:trials_count]
        best_params = grid[0]
        best_score = math.inf if minimize else -math.inf
        for index, params in enumerate(grid):
            score, _ = train_candidate(params)
            candidates.append({"trial": index, "score": score, **params})
            if (score < best_score if minimize else score > best_score):
                best_score, best_params = score, params
    save_json(run_dir / "hyperparameter_trials.json", {"selection_metric": selection_metric, "best_score": best_score, "best_params": best_params, "trials": candidates})
    final_model = TabularMLP(
        bundle.input_shape[0], bundle.num_classes,
        hidden=tuple(best_params["hidden"]), dropout=float(best_params["dropout"]),
        activation=str(best_params["activation"]), batch_norm=bool(best_params["batch_norm"]),
    )
    final_config = dict(config)
    final_config["learning_rate"] = float(best_params["learning_rate"])
    model, history, provisional, _, _, _ = _train_torch_model(final_model, bundle, final_config, device, run_dir, evaluate_test=False)
    freeze()
    metrics, y_true, y_pred, probs, _ = evaluate(model, bundle.test, device, bundle.task, bundle.num_classes, int(config["batch_size"]) * 2)
    metrics["parameters"] = parameter_count(model)
    metrics["best_validation_score"] = best_score
    metrics["best_params"] = best_params
    metrics["selection_value"] = provisional.get("selection_value")
    return model, history, metrics, y_true, y_pred, probs


def _run_augmentation_comparison(
    bundle: DataBundle,
    config: dict[str, Any],
    device: torch.device,
    run_dir: Path,
    freeze: Callable[[], Path],
):
    # Both branches use the same sample IDs. The validation split selects the
    # branch; only the selected branch is evaluated on test.
    reference_bundle = prepare_dataset(
        "03_cnn_vision",
        quick=bool(config.get("_quick")),
        seed=int(config.get("split_seed", 42)),
    )
    reference_dir = run_dir / "without_augmentation"
    augmented_dir = run_dir / "with_augmentation"
    reference_dir.mkdir(parents=True, exist_ok=True)
    augmented_dir.mkdir(parents=True, exist_ok=True)

    reference_model = SmallCNN(reference_bundle.input_shape[0], int(reference_bundle.num_classes))
    reference_model, reference_history, _provisional, _, _, _ = _train_torch_model(
        reference_model,
        reference_bundle,
        config,
        device,
        reference_dir,
        activation_label="without_augmentation",
        evaluate_test=False,
    )
    augmented_model = SmallCNN(bundle.input_shape[0], int(bundle.num_classes))
    augmented_model, augmented_history, _provisional, _, _, _ = _train_torch_model(
        augmented_model,
        bundle,
        config,
        device,
        augmented_dir,
        activation_label="with_augmentation",
        evaluate_test=False,
    )

    selection_metric = str(config["selection_metric"])
    minimize = selection_metric in {"loss", "rmse", "mae", "mape", "brier", "ece"}
    reference_values = reference_history["validation_metric"]
    augmented_values = augmented_history["validation_metric"]
    reference_score = min(reference_values) if minimize else max(reference_values)
    augmented_score = min(augmented_values) if minimize else max(augmented_values)
    augmented_wins = augmented_score < reference_score if minimize else augmented_score > reference_score

    selected_name = "with_augmentation" if augmented_wins else "without_augmentation"
    selected_model = augmented_model if augmented_wins else reference_model
    selected_bundle = bundle if augmented_wins else reference_bundle
    selected_history = augmented_history if augmented_wins else reference_history
    torch.save({"state_dict": selected_model.state_dict()}, run_dir / "best_model.pt")
    freeze()
    metrics, y_true, y_pred, probabilities, _ = evaluate(
        selected_model,
        selected_bundle.test,
        device,
        selected_bundle.task,
        selected_bundle.num_classes,
        int(config["batch_size"]) * 2,
    )
    # Robustness is a final diagnostic on the selected model only. It does not
    # influence model selection.
    robust_targets: list[np.ndarray] = []
    robust_predictions: list[np.ndarray] = []
    selected_model.eval()
    generator = torch.Generator(device="cpu").manual_seed(int(config["seed"]) + 700)
    with torch.no_grad():
        for features, targets in _loader(selected_bundle.test, int(config["batch_size"]) * 2, False):
            noise = torch.randn(features.shape, generator=generator) * 0.08
            corrupted = features + noise
            logits = selected_model(corrupted.to(device))
            predictions, _ = _predict_from_logits(logits, selected_bundle.num_classes, selected_bundle.task)
            robust_targets.append(targets.numpy().reshape(-1))
            robust_predictions.append(predictions)
    robust_metrics = classification_metrics(
        np.concatenate(robust_targets),
        np.concatenate(robust_predictions),
    )
    metrics.update(
        {
            "selected_variant": selected_name,
            "validation_without_augmentation": float(reference_score),
            "validation_with_augmentation": float(augmented_score),
            "robust_accuracy": float(robust_metrics["accuracy"]),
            "parameters": parameter_count(selected_model),
        }
    )
    save_json(
        run_dir / "augmentation_comparison.json",
        {
            "selection_metric": selection_metric,
            "selected_variant": selected_name,
            "without_augmentation": {"validation_selection": float(reference_score)},
            "with_augmentation": {"validation_selection": float(augmented_score)},
            "test_metrics_selected_only": metrics,
        },
    )
    return selected_model, selected_history, metrics, y_true, y_pred, probabilities

def _apply_explainability(model: nn.Module, bundle: DataBundle, device: torch.device, run_dir: Path) -> None:
    if not isinstance(bundle.test, ArrayDataset): return
    model.eval(); x=bundle.test.features[:min(128,len(bundle.test))].to(device); baseline=torch.zeros_like(x); steps=24; accumulated=torch.zeros_like(x)
    for alpha in torch.linspace(0,1,steps,device=device):
        interpolated=(baseline+alpha*(x-baseline)).requires_grad_(True); logits=model(interpolated); target=logits if logits.ndim==1 else logits.max(1).values; gradients=torch.autograd.grad(target.sum(),interpolated)[0]; accumulated+=gradients
    attributions=((x-baseline)*accumulated/steps).abs().mean(0).detach().cpu().numpy(); names=bundle.feature_names or [f"feature_{i}" for i in range(len(attributions))]; frame=pd.DataFrame({"feature":names,"mean_absolute_attribution":attributions}).sort_values("mean_absolute_attribution",ascending=False); frame.to_csv(run_dir/"feature_attributions.csv",index=False)


def _apply_calibration(model: nn.Module, bundle: DataBundle, device: torch.device, metrics: dict[str,Any], run_dir: Path) -> None:
    model.eval()
    def logits_targets(dataset):
        logits=[]; targets=[]
        with torch.no_grad():
            for x,y in _loader(dataset,256,False): logits.append(model(x.to(device)).view(-1).cpu()); targets.append(y.view(-1).cpu())
        return torch.cat(logits),torch.cat(targets).float()
    val_logits,val_targets=logits_targets(bundle.validation); temperature=torch.tensor(1.0,requires_grad=True); optimizer=torch.optim.LBFGS([temperature],lr=0.1,max_iter=50)
    def closure(): optimizer.zero_grad(); loss=nn.functional.binary_cross_entropy_with_logits(val_logits/temperature.clamp_min(0.05),val_targets); loss.backward(); return loss
    optimizer.step(closure); test_logits,test_targets=logits_targets(bundle.test); probs=torch.sigmoid(test_logits/temperature.detach().clamp_min(0.05)).numpy(); metrics["temperature"]=float(temperature.detach().clamp_min(0.05)); metrics["ece"]=expected_calibration_error(test_targets.numpy().astype(int),probs); save_json(run_dir/"calibration.json",{"temperature":metrics["temperature"],"ece":metrics["ece"]})


def _export_onnx(model: nn.Module, bundle: DataBundle, run_dir: Path, device: torch.device) -> Path | None:
    if model is None or bundle.input_shape is None or bundle.task in {"node_classification","reinforcement_learning","generation"}: return None
    sample=next(iter(_loader(bundle.test,1,False)))[0].to(device); path=run_dir/"model.onnx"; model.eval()
    try:
        torch.onnx.export(model,sample,path,input_names=["input"],output_names=["output"],dynamic_axes={"input":{0:"batch"},"output":{0:"batch"}},opset_version=18)
        return path
    except Exception as exc:
        (run_dir/"onnx_export_error.txt").write_text(str(exc),encoding="utf-8"); return None


def run_lab(
    lab_id: str,
    *,
    quick: bool = False,
    config_name: str = "baseline",
    prepared_bundle: DataBundle | None = None,
    output_dir: str | Path = "runs",
    device: str | None = None,
    seed: int | None = None,
    split_seed: int | None = None,
    training_seed: int | None = None,
    tracker: str = "json",
    amp: bool | None = None,
    compile_model: bool | None = None,
    deterministic: bool = True,
    num_workers: int | None = None,
) -> ExperimentResult:
    lab = get_lab(lab_id)
    seeds = SeedPlan.resolve(split_seed=split_seed, training_seed=training_seed, legacy_seed=seed)
    config = merged_config(lab_id, config_name, device=device, seed=seeds.training_seed)
    config["split_seed"] = seeds.split_seed
    config["training_seed"] = seeds.training_seed
    config["_quick"] = quick
    config["_config_name"] = config_name
    if amp is not None:
        config["amp"] = amp
    if compile_model is not None:
        config["compile"] = compile_model
    if num_workers is not None:
        config["num_workers"] = num_workers
    config.setdefault("amp", False)
    config.setdefault("compile", False)
    config.setdefault("num_workers", 0)
    config.setdefault("gradient_clip_norm", 5.0)
    config["deterministic"] = deterministic
    seed_everything(seeds.training_seed, deterministic=deterministic)
    resolved_device = get_device(str(config["device"]))
    bundle = prepared_bundle or prepare_dataset(lab_id, quick=quick, seed=seeds.split_seed)
    run_dir = create_run_dir(lab_id, output_dir)
    dataset_manifest = {**get_dataset(lab_id), **bundle.metadata, "summary": bundle.summary}
    initialize_run(run_dir, lab_id, config, dataset_manifest, resolved_device)
    save_json(run_dir / "data_quality.json", quality_report(bundle))
    save_json(run_dir / "drift_report.json", drift_report(bundle))
    try:
        baseline_validation = run_baseline(lab_id, bundle, quick=quick, evaluation_split="validation")
    except Exception as exc:
        baseline_validation = {"evaluation_split": "validation", "error": str(exc)}
    save_json(run_dir / "baseline_validation_metrics.json", baseline_validation)
    lock_path = run_dir / "experiment.lock.json"
    def freeze_experiment() -> Path:
        if lock_path.is_file():
            return lock_path
        experiment_lock = ExperimentLock.create(
            lab_id=lab_id,
            seeds=seeds,
            config_name=config_name,
            selection_metric=str(config.get("selection_metric", lab.get("selection_metric", "loss"))),
            selected_checkpoint=run_dir / "best_model.pt",
            dataset_hash=str(bundle.metadata.get("dataset_hash", "unknown")),
        )
        return experiment_lock.write(run_dir)

    persist_inference_contract(bundle, run_dir, architecture=str(lab["architecture"]))
    experiment_tracker = create_tracker(tracker, run_dir)
    experiment_tracker.start(
        run_name=run_dir.name,
        experiment_name=f"neural-labs-{lab_id}",
        tags={"lab": lab_id, "dataset": str(lab["dataset"]), "architecture": str(lab["architecture"])},
    )
    experiment_tracker.log_params({key: value for key, value in config.items() if not str(key).startswith("_")})

    started = time.perf_counter()
    architecture = lab["architecture"]
    if architecture == "numpy_logistic":
        result = _run_numpy_logistic(bundle, config, run_dir, freeze_experiment)
    elif architecture == "numpy_mlp":
        result = _run_numpy_mlp(bundle, config, run_dir, freeze_experiment)
    elif architecture == "autoencoder":
        result = _run_autoencoder(bundle, config, resolved_device, run_dir, freeze_experiment)
    elif architecture == "dcgan":
        result = _run_gan(bundle, config, resolved_device, run_dir, freeze_experiment)
    elif architecture == "gcn":
        result = _run_gnn(bundle, config, resolved_device, run_dir, freeze_experiment)
    elif architecture == "dqn_inventory":
        result = _run_dqn(bundle, config, resolved_device, run_dir, freeze_experiment)
    elif architecture == "transfer_resnet18":
        result = _run_transfer(bundle, config, resolved_device, run_dir, freeze_experiment)
    elif architecture == "distillation_cnn":
        result = _run_distillation(bundle, config, resolved_device, run_dir, freeze_experiment)
    elif architecture == "fedavg_mlp":
        result = _run_federated(bundle, config, resolved_device, run_dir, freeze_experiment)
    elif architecture == "mlp_optuna":
        result = _run_hyperparameter_search(bundle, config, resolved_device, run_dir, freeze_experiment)
    elif architecture == "augmentation_comparison":
        result = _run_augmentation_comparison(bundle, config, resolved_device, run_dir, freeze_experiment)
    elif architecture in {"activation_comparison", "optimizer_comparison", "regularization_comparison"}:
        result = _run_variant_comparison(lab, bundle, config, resolved_device, run_dir, freeze_experiment)
    else:
        model = _standard_model(lab, bundle, config)
        if bool(config.get("compile", False)) and hasattr(torch, "compile"):
            try:
                if hasattr(model, "compile"):
                    model.compile()
                else:
                    model = torch.compile(model)
                config["compiled_model"] = True
            except Exception as exc:
                config["compiled_model"] = False
                config["compile_warning"] = str(exc)
        result = _train_torch_model(model, bundle, config, resolved_device, run_dir, evaluate_test=False)

    model, history, metrics, y_true, y_pred, probabilities = result
    if not lock_path.is_file():
        freeze_experiment()
    if model is not None and len(y_true) == 0 and bundle.task not in {"node_classification", "reinforcement_learning", "generation", "anomaly_detection"}:
        final_metrics, y_true, y_pred, probabilities, _ = evaluate(
            model, bundle.test, resolved_device, bundle.task, bundle.num_classes, int(config["batch_size"]) * 2
        )
        final_metrics["parameters"] = parameter_count(model)
        metrics = final_metrics
    metrics["wall_time_seconds"] = float(time.perf_counter() - started)
    metrics["device_type"] = resolved_device.type
    metrics["amp_enabled"] = bool(config.get("amp", False) and resolved_device.type == "cuda")
    metrics["deterministic"] = bool(deterministic)
    if model is not None and bundle.task not in {"node_classification", "reinforcement_learning", "generation"}:
        try:
            sample = next(iter(_loader(bundle.test, min(32, len(bundle.test)), False)))[0]
            metrics.update(profile_inference(model, sample, resolved_device, warmup=1 if quick else 3, iterations=3 if quick else 10))
        except Exception as exc:
            save_json(run_dir / "profiling_error.json", {"error": str(exc)})

    # Baseline test results are calculated only after the neural experiment has
    # completed model selection and its final test evaluation.
    try:
        baseline = run_baseline(lab_id, bundle, quick=quick, evaluation_split="test")
    except Exception as exc:
        baseline = {"error": str(exc)}
    save_json(run_dir / "baseline_metrics.json", baseline)

    if lab_id == "21_explainability" and model is not None:
        _apply_explainability(model, bundle, resolved_device, run_dir)
    if lab_id == "22_uncertainty_calibration" and model is not None:
        _apply_calibration(model, bundle, resolved_device, metrics, run_dir)
    if lab_id == "23_model_export_and_inference" and model is not None:
        start = time.perf_counter()
        batch = next(iter(_loader(bundle.test, 64, False)))[0].to(resolved_device)
        with torch.no_grad():
            model(batch)
        elapsed = time.perf_counter() - start
        metrics["latency_ms"] = float(elapsed * 1000)
        metrics["throughput"] = float(len(batch) / max(elapsed, 1e-9))
        _export_onnx(model, bundle, run_dir, resolved_device)
        metrics["model_size_mb"] = (
            float((run_dir / "best_model.pt").stat().st_size / 1024 / 1024)
            if (run_dir / "best_model.pt").exists()
            else 0.0
        )

    save_json(run_dir / "metrics.json", metrics)
    history_frame = None
    if history:
        maximum = max(len(values) for values in history.values())
        padded = {
            key: list(values) + [np.nan] * (maximum - len(values))
            for key, values in history.items()
        }
        save_history(padded, run_dir / "history.csv", run_dir / "history.png")
        history_frame = pd.DataFrame(padded)
    if len(y_true):
        save_predictions(run_dir, bundle.test_ids[: len(y_true)], y_true, y_pred, probabilities)
        confidence_intervals = bootstrap_confidence_intervals(
            y_true,
            y_pred,
            task=bundle.task,
            seed=seeds.training_seed,
            samples=100 if quick else 500,
        )
        save_json(run_dir / "confidence_intervals.json", confidence_intervals)
        save_json(run_dir / "subgroup_metrics.json", subgroup_report(bundle, y_true, y_pred))
        if bundle.task != "regression":
            save_confusion_matrix(
                y_true,
                y_pred,
                run_dir / "confusion_matrix.png",
                bundle.class_names,
            )
    model_spec = {
        "architecture": architecture,
        "input_shape": bundle.input_shape,
        "num_classes": bundle.num_classes,
        "metadata": _metadata_for_model(bundle),
    }
    save_json(run_dir / "model_spec.json", model_spec)
    write_model_card(run_dir, lab_id, metrics, baseline, config)
    write_report(run_dir, lab_id, metrics, baseline)
    artifacts = {path.name: str(path) for path in run_dir.iterdir() if path.is_file()}
    experiment_tracker.log_metrics(metrics)
    for artifact_path in run_dir.iterdir():
        if artifact_path.is_file() and artifact_path.name != "tracking.jsonl":
            experiment_tracker.log_artifact(artifact_path)
    experiment_tracker.finish("FINISHED")
    return ExperimentResult(lab_id, run_dir, metrics, history_frame, artifacts)

def load_trained_model(lab_id: str, run_dir: Path, device: torch.device) -> nn.Module:
    spec=json.loads((run_dir/"model_spec.json").read_text(encoding="utf-8")); architecture=spec["architecture"]
    if architecture in {"numpy_logistic","numpy_mlp","gcn","dqn_inventory","dcgan","transfer_resnet18","distillation_cnn"}:
        raise NotImplementedError(f"Carga genérica no disponible para {architecture}; use el notebook del laboratorio.")
    model=build_model(architecture,tuple(spec["input_shape"]),spec["num_classes"],spec.get("metadata") or {})
    checkpoint=torch.load(run_dir/"best_model.pt",map_location=device); model.load_state_dict(checkpoint["state_dict"]); return model.to(device).eval()
