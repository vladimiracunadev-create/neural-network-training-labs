from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader

from .catalog import get_lab
from .datasets import ArrayDataset, DataBundle
from .metrics import classification_metrics, regression_metrics

EvaluationSplit = Literal["validation", "test"]


def _arrays(dataset: Any, limit: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(dataset, ArrayDataset):
        x = dataset.features.detach().cpu().numpy()
        y = dataset.targets.detach().cpu().numpy()
        return (x[:limit], y[:limit]) if limit else (x, y)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    xs, ys = [], []
    for features, targets in loader:
        xs.append(features.flatten(1).numpy())
        ys.append(targets.numpy())
        if limit and sum(len(item) for item in ys) >= limit:
            break
    x = np.concatenate(xs)[:limit]
    y = np.concatenate(ys)[:limit]
    return x, y


def _evaluation_dataset(bundle: DataBundle, split: EvaluationSplit) -> Any:
    return bundle.validation if split == "validation" else bundle.test


def _tabular_or_image(bundle: DataBundle, quick: bool, split: EvaluationSplit) -> dict[str, Any]:
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.linear_model import LogisticRegression, Ridge, SGDClassifier

    first_features, _first_target = bundle.train[0]
    input_dimension = int(first_features.numel())
    if input_dimension > 10_000:
        train_limit = 512 if quick else 1024
        eval_limit = 512 if quick else 1024
    else:
        train_limit = 4000 if quick else 12000
        eval_limit = 2000 if quick else None
    x_train, y_train = _arrays(bundle.train, train_limit)
    x_eval, y_eval = _arrays(_evaluation_dataset(bundle, split), eval_limit)
    if bundle.task == "regression":
        dummy = DummyRegressor(strategy="mean").fit(x_train, y_train)
        ridge = Ridge(alpha=1.0).fit(x_train, y_train)
        return {
            "evaluation_split": split,
            "dummy": regression_metrics(y_eval, dummy.predict(x_eval)),
            "ridge": regression_metrics(y_eval, ridge.predict(x_eval)),
        }
    dummy = DummyClassifier(strategy="prior").fit(x_train, y_train)
    if input_dimension > 500:
        linear = SGDClassifier(
            loss="log_loss",
            max_iter=60 if quick else 150,
            class_weight="balanced" if len(np.unique(y_train)) == 2 else None,
            random_state=42,
            tol=1e-3,
        ).fit(x_train, y_train)
        name = "sgd_linear_classifier"
    else:
        linear = LogisticRegression(
            max_iter=300,
            class_weight="balanced" if len(np.unique(y_train)) == 2 else None,
        ).fit(x_train, y_train)
        name = "logistic_regression"
    probabilities = linear.predict_proba(x_eval) if hasattr(linear, "predict_proba") else None
    return {
        "evaluation_split": split,
        "dummy": classification_metrics(y_eval, dummy.predict(x_eval)),
        name: classification_metrics(y_eval, linear.predict(x_eval), probabilities),
    }


def _text(bundle: DataBundle, quick: bool, split: EvaluationSplit) -> dict[str, Any]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    train_texts = bundle.raw["train_texts"]
    eval_texts = bundle.raw[f"{split}_texts"]
    train_labels = np.asarray(bundle.raw["train_labels"])
    eval_labels = np.asarray(bundle.raw[f"{split}_labels"])
    if quick:
        train_texts, train_labels = train_texts[:4000], train_labels[:4000]
        eval_texts, eval_labels = eval_texts[:1000], eval_labels[:1000]
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)
    train_x = vectorizer.fit_transform(train_texts)
    eval_x = vectorizer.transform(eval_texts)
    model = LogisticRegression(max_iter=300).fit(train_x, train_labels)
    return {
        "evaluation_split": split,
        "tfidf_logistic_regression": classification_metrics(
            eval_labels,
            model.predict(eval_x),
            model.predict_proba(eval_x),
        ),
    }


def _inventory_baseline(bundle: DataBundle, split: EvaluationSplit) -> dict[str, Any]:
    train_demand = np.asarray(bundle.raw["train_demand"], dtype=np.float32)
    eval_demand = np.asarray(bundle.raw[f"{split}_demand"], dtype=np.float32)
    target_stock = float(np.quantile(train_demand, 0.80))
    inventory = target_stock
    total_reward = 0.0
    stockouts = 0
    holding = 0.0
    served = 0.0
    total_demand = float(eval_demand.sum())
    for demand in eval_demand:
        order = max(0.0, target_stock - inventory)
        available = inventory + order
        sales = min(available, float(demand))
        shortage = max(0.0, float(demand) - available)
        inventory = max(0.0, available - float(demand))
        holding += inventory
        served += sales
        stockouts += int(shortage > 0)
        total_reward += sales - 0.20 * order - 0.05 * inventory - 1.50 * shortage
    return {
        "evaluation_split": split,
        "periodic_reorder_policy": {
            "mean_return": float(total_reward),
            "stockout_rate": float(stockouts / max(1, len(eval_demand))),
            "holding_cost": float(0.05 * holding),
            "service_level": float(served / max(total_demand, 1e-8)),
            "target_stock": target_stock,
        },
    }


def _cora(bundle: DataBundle, split: EvaluationSplit) -> dict[str, Any]:
    try:
        from torch_geometric.nn import MLP
    except ImportError as exc:
        raise RuntimeError('Instale el extra de grafos: pip install -e ".[graph]"') from exc
    graph = bundle.train
    model = MLP([graph.num_node_features, 64, int(graph.y.max().item()) + 1], dropout=0.3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for _ in range(60):
        model.train()
        optimizer.zero_grad()
        logits = model(graph.x)
        loss = torch.nn.functional.cross_entropy(logits[graph.train_mask], graph.y[graph.train_mask])
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        predictions = model(graph.x).argmax(dim=1)
    mask = graph.val_mask if split == "validation" else graph.test_mask
    metrics = classification_metrics(graph.y[mask].cpu().numpy(), predictions[mask].cpu().numpy())
    return {"evaluation_split": split, "mlp_without_edges": metrics}


def run_baseline(
    lab_id: str,
    bundle: DataBundle,
    *,
    quick: bool = False,
    evaluation_split: EvaluationSplit = "validation",
) -> dict[str, Any]:
    lab = get_lab(lab_id)
    if evaluation_split not in {"validation", "test"}:
        raise ValueError("evaluation_split debe ser 'validation' o 'test'.")
    if bundle.task == "reinforcement_learning":
        return _inventory_baseline(bundle, evaluation_split)
    if bundle.task == "node_classification":
        return _cora(bundle, evaluation_split)
    if lab["source_type"] == "huggingface":
        return _text(bundle, quick, evaluation_split)
    if bundle.task == "generation":
        x, _ = _arrays(bundle.train, 2000 if quick else 10000)
        return {
            "evaluation_split": evaluation_split,
            "real_data_reference": {
                "pixel_mean": float(x.mean()),
                "pixel_std": float(x.std()),
                "samples": int(len(x)),
            },
        }
    return _tabular_or_image(bundle, quick, evaluation_split)
