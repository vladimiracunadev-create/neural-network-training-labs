from __future__ import annotations

import json
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset

from .catalog import ROOT, get_dataset, get_lab
from .runtime import save_json, sha256_strings

RAW_ROOT = ROOT / "data" / "raw"
PROCESSED_ROOT = ROOT / "data" / "processed"


class ArrayDataset(Dataset):
    def __init__(self, features: np.ndarray | torch.Tensor, targets: np.ndarray | torch.Tensor, ids: Sequence[str], extra: dict[str, Any] | None = None):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        targets_array = np.asarray(targets)
        if np.issubdtype(targets_array.dtype, np.integer):
            self.targets = torch.as_tensor(targets_array, dtype=torch.long)
        else:
            self.targets = torch.as_tensor(targets_array, dtype=torch.float32)
        self.ids = list(map(str, ids))
        self.extra = extra or {}

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        return self.features[index], self.targets[index]


@dataclass
class DataBundle:
    lab_id: str
    dataset_name: str
    task: str
    train: Any
    validation: Any
    test: Any
    input_shape: tuple[int, ...] | None
    num_classes: int | None
    class_names: list[str] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)
    train_ids: list[str] = field(default_factory=list)
    validation_ids: list[str] = field(default_factory=list)
    test_ids: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


def _quick_indices(length: int, maximum: int, seed: int) -> np.ndarray:
    if length <= maximum:
        return np.arange(length)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(length, size=maximum, replace=False))


def _subset(dataset: Any, ids: list[str], maximum: int, seed: int) -> tuple[Any, list[str]]:
    indices = _quick_indices(len(dataset), maximum, seed)
    if isinstance(dataset, ArrayDataset):
        features = dataset.features[indices]
        targets = dataset.targets[indices]
        selected_ids = [ids[i] for i in indices]
        extra = {}
        for key, value in dataset.extra.items():
            try:
                extra[key] = np.asarray(value)[indices]
            except Exception:
                extra[key] = value
        return ArrayDataset(features, targets, selected_ids, extra=extra), selected_ids
    return Subset(dataset, indices.tolist()), [ids[i] for i in indices]


def _persist_split_manifest(bundle: DataBundle, seed: int) -> dict[str, Any]:
    processed = PROCESSED_ROOT / bundle.dataset_name
    processed.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": bundle.dataset_name,
        "lab": bundle.lab_id,
        "seed": seed,
        "policy": "train and validation may guide learning; test is isolated until final evaluation",
        "counts": {"train": len(bundle.train_ids), "validation": len(bundle.validation_ids), "test": len(bundle.test_ids)},
        "hashes": {
            "train": sha256_strings(bundle.train_ids),
            "validation": sha256_strings(bundle.validation_ids),
            "test": sha256_strings(bundle.test_ids),
        },
    }
    save_json(processed / f"splits-{bundle.lab_id}-seed{seed}.json", payload)
    bundle.metadata["split_manifest"] = payload
    bundle.metadata["dataset_hash"] = sha256_strings(bundle.train_ids + bundle.validation_ids + bundle.test_ids)
    return payload


def audit_bundle(bundle: DataBundle) -> dict[str, Any]:
    train = set(bundle.train_ids)
    validation = set(bundle.validation_ids)
    test = set(bundle.test_ids)
    overlaps = {
        "train_validation": sorted(train & validation)[:10],
        "train_test": sorted(train & test)[:10],
        "validation_test": sorted(validation & test)[:10],
    }
    ok = all(not values for values in overlaps.values())
    if not ok:
        raise ValueError(f"Fuga de datos detectada en {bundle.lab_id}: {overlaps}")
    return {"ok": True, "overlaps": overlaps, "counts": {"train": len(train), "validation": len(validation), "test": len(test)}}


def _encode_tabular(
    x_train: pd.DataFrame,
    x_validation: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_validation: pd.Series,
    y_test: pd.Series,
    *,
    classification: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str], Any]:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

    numeric = x_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [column for column in x_train.columns if column not in numeric]
    transformer = ColumnTransformer(
        [
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
            ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    train_x = transformer.fit_transform(x_train)
    validation_x = transformer.transform(x_validation)
    test_x = transformer.transform(x_test)
    feature_names = transformer.get_feature_names_out().tolist()
    if classification:
        encoder = LabelEncoder()
        train_y = encoder.fit_transform(y_train.astype(str).str.strip())
        validation_y = encoder.transform(y_validation.astype(str).str.strip())
        test_y = encoder.transform(y_test.astype(str).str.strip())
        class_names = encoder.classes_.tolist()
    else:
        encoder = None
        train_y = pd.to_numeric(y_train, errors="coerce").to_numpy(dtype=np.float32)
        validation_y = pd.to_numeric(y_validation, errors="coerce").to_numpy(dtype=np.float32)
        test_y = pd.to_numeric(y_test, errors="coerce").to_numpy(dtype=np.float32)
        class_names = []
    return (
        np.asarray(train_x, dtype=np.float32),
        np.asarray(validation_x, dtype=np.float32),
        np.asarray(test_x, dtype=np.float32),
        np.asarray(train_y),
        np.asarray(validation_y),
        np.asarray(test_y),
        feature_names,
        class_names,
        transformer,
    )


def _frame_split(x: pd.DataFrame, y: pd.Series, seed: int, *, chronological: bool = False, stratify: bool = True):
    from sklearn.model_selection import train_test_split

    ids = x.index.astype(str)
    if chronological:
        train_end = int(len(x) * 0.70)
        validation_end = int(len(x) * 0.85)
        return (
            x.iloc[:train_end], x.iloc[train_end:validation_end], x.iloc[validation_end:],
            y.iloc[:train_end], y.iloc[train_end:validation_end], y.iloc[validation_end:],
            ids[:train_end].tolist(), ids[train_end:validation_end].tolist(), ids[validation_end:].tolist(),
        )
    stratify_values = y if stratify and y.nunique(dropna=False) > 1 else None
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.30, random_state=seed, stratify=stratify_values)
    stratify_temp = y_temp if stratify_values is not None else None
    x_validation, x_test, y_validation, y_test = train_test_split(x_temp, y_temp, test_size=0.50, random_state=seed, stratify=stratify_temp)
    return (
        x_train, x_validation, x_test, y_train, y_validation, y_test,
        x_train.index.astype(str).tolist(), x_validation.index.astype(str).tolist(), x_test.index.astype(str).tolist(),
    )


def _load_uci(lab_id: str, dataset_id: int, seed: int, quick: bool) -> DataBundle:
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise RuntimeError('Instale el extra de datos: pip install -e ".[data]"') from exc

    lab = get_lab(lab_id)
    fetched = fetch_ucirepo(id=dataset_id)
    x = fetched.data.features.copy()
    targets = fetched.data.targets.copy()
    if targets.shape[1] != 1:
        target_name = targets.columns[0]
    else:
        target_name = targets.columns[0]
    y = targets[target_name]
    x.index = [f"uci-{dataset_id}-{index}" for index in range(len(x))]
    y.index = x.index

    chronological = lab["dataset"] == "seoul_bike"
    if chronological:
        # UCI provides date/hour fields. Keep original order or sort by date/hour when available.
        date_columns = [column for column in x.columns if "date" in column.lower()]
        hour_columns = [column for column in x.columns if "hour" in column.lower()]
        if date_columns:
            dates = pd.to_datetime(x[date_columns[0]], errors="coerce", dayfirst=True)
            hours = pd.to_numeric(x[hour_columns[0]], errors="coerce") if hour_columns else 0
            order = np.argsort((dates.view("int64") + np.asarray(hours) * 3_600_000_000_000).to_numpy())
            x = x.iloc[order]
            y = y.iloc[order]
    split = _frame_split(x, y, seed, chronological=chronological, stratify=lab["task"] != "regression")
    x_train, x_val, x_test, y_train, y_val, y_test, train_ids, val_ids, test_ids = split
    encoded = _encode_tabular(x_train, x_val, x_test, y_train, y_val, y_test, classification=lab["task"] != "regression")
    train_x, val_x, test_x, train_y, val_y, test_y, feature_names, class_names, transformer = encoded
    if chronological:
        window = 24
        def make_windows(features, targets, ids):
            if len(features) < window:
                raise ValueError(f"La partición temporal tiene menos de {window} observaciones.")
            sequences = np.stack([features[i-window+1:i+1] for i in range(window-1, len(features))]).astype(np.float32)
            window_targets = np.asarray(targets[window-1:], dtype=np.float32)
            window_ids = list(ids[window-1:])
            return sequences, window_targets, window_ids
        train_x, train_y, train_ids = make_windows(train_x, train_y, train_ids)
        val_x, val_y, val_ids = make_windows(val_x, val_y, val_ids)
        test_x, test_y, test_ids = make_windows(test_x, test_y, test_ids)
    train = ArrayDataset(train_x, train_y, train_ids)
    validation = ArrayDataset(val_x, val_y, val_ids)
    test = ArrayDataset(test_x, test_y, test_ids)
    if quick:
        train, train_ids = _subset(train, train_ids, 1024, seed)
        validation, val_ids = _subset(validation, val_ids, 256, seed + 1)
        test, test_ids = _subset(test, test_ids, 256, seed + 2)
    num_classes = len(class_names) if class_names else None
    bundle = DataBundle(
        lab_id=lab_id, dataset_name=lab["dataset"], task=lab["task"], train=train, validation=validation, test=test,
        input_shape=tuple(train.features.shape[1:]), num_classes=num_classes, class_names=class_names, feature_names=feature_names,
        train_ids=train_ids, validation_ids=val_ids, test_ids=test_ids,
        summary={"source_instances": len(x), "features_after_preprocessing": len(feature_names), "target": target_name, "chronological_split": chronological},
        metadata={"source_metadata": fetched.metadata, "transformer": transformer},
        raw={"x_train": x_train, "x_validation": x_val, "x_test": x_test, "y_train": y_train, "y_validation": y_val, "y_test": y_test},
    )
    return bundle


def _load_california(lab_id: str, seed: int, quick: bool) -> DataBundle:
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing(as_frame=True)
    x = data.data.copy()
    y = data.target.copy()
    x.index = [f"california-{index}" for index in range(len(x))]
    y.index = x.index
    split = _frame_split(x, y, seed, chronological=False, stratify=False)
    x_train, x_val, x_test, y_train, y_val, y_test, train_ids, val_ids, test_ids = split
    encoded = _encode_tabular(x_train, x_val, x_test, y_train, y_val, y_test, classification=False)
    train_x, val_x, test_x, train_y, val_y, test_y, feature_names, class_names, transformer = encoded
    train = ArrayDataset(train_x, train_y, train_ids)
    validation = ArrayDataset(val_x, val_y, val_ids)
    test = ArrayDataset(test_x, test_y, test_ids)
    if quick:
        train, train_ids = _subset(train, train_ids, 1024, seed)
        validation, val_ids = _subset(validation, val_ids, 256, seed + 1)
        test, test_ids = _subset(test, test_ids, 256, seed + 2)
    return DataBundle(lab_id, "california_housing", "regression", train, validation, test, (train.features.shape[1],), None,
        feature_names=feature_names, train_ids=train_ids, validation_ids=val_ids, test_ids=test_ids,
        summary={"source_instances": len(x), "features_after_preprocessing": len(feature_names)}, metadata={"transformer": transformer},
        raw={"x_train": x_train, "x_validation": x_val, "x_test": x_test, "y_train": y_train, "y_validation": y_val, "y_test": y_test})


def _vision_transforms(dataset_name: str, lab_id: str):
    from torchvision import transforms
    if dataset_name == "oxford_iiit_pet":
        train_transform = transforms.Compose([transforms.Resize((128, 128)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
        eval_transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor(), transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
        return train_transform, eval_transform, (3,128,128)
    if dataset_name == "cifar10":
        if lab_id == "20_data_augmentation":
            train_transform = transforms.Compose([transforms.RandomCrop(32,padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914,0.4822,0.4465),(0.247,0.243,0.261))])
        else:
            train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914,0.4822,0.4465),(0.247,0.243,0.261))])
        eval_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914,0.4822,0.4465),(0.247,0.243,0.261))])
        return train_transform, eval_transform, (3,32,32)
    if lab_id == "08_gan_generation":
        train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        return train_transform, train_transform, (1,28,28)
    train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
    return train_transform, train_transform, (1,28,28)


def _load_vision(lab_id: str, dataset_name: str, seed: int, quick: bool) -> DataBundle:
    try:
        from torchvision import datasets
    except ImportError as exc:
        raise RuntimeError('Instale el extra de visión: pip install -e ".[vision]"') from exc
    train_transform, eval_transform, input_shape = _vision_transforms(dataset_name, lab_id)
    root = RAW_ROOT / dataset_name
    root.mkdir(parents=True, exist_ok=True)
    if dataset_name == "cifar10":
        full_train_aug = datasets.CIFAR10(root=root, train=True, transform=train_transform, download=True)
        full_train_eval = datasets.CIFAR10(root=root, train=True, transform=eval_transform, download=False)
        test = datasets.CIFAR10(root=root, train=False, transform=eval_transform, download=True)
        class_names = list(full_train_aug.classes)
    elif dataset_name == "fashion_mnist":
        full_train_aug = datasets.FashionMNIST(root=root, train=True, transform=train_transform, download=True)
        full_train_eval = datasets.FashionMNIST(root=root, train=True, transform=eval_transform, download=False)
        test = datasets.FashionMNIST(root=root, train=False, transform=eval_transform, download=True)
        class_names = list(full_train_aug.classes)
    elif dataset_name == "oxford_iiit_pet":
        full_train_aug = datasets.OxfordIIITPet(root=root, split="trainval", target_types="category", transform=train_transform, download=True)
        full_train_eval = datasets.OxfordIIITPet(root=root, split="trainval", target_types="category", transform=eval_transform, download=False)
        test = datasets.OxfordIIITPet(root=root, split="test", target_types="category", transform=eval_transform, download=True)
        class_names = list(full_train_aug.classes)
    else:
        raise KeyError(dataset_name)
    rng = np.random.default_rng(seed)
    indices = np.arange(len(full_train_aug))
    rng.shuffle(indices)
    validation_size = max(1, int(len(indices) * 0.15))
    validation_indices = np.sort(indices[:validation_size])
    train_indices = np.sort(indices[validation_size:])
    train = Subset(full_train_aug, train_indices.tolist())
    validation = Subset(full_train_eval, validation_indices.tolist())
    train_ids = [f"{dataset_name}-train-{int(i)}" for i in train_indices]
    val_ids = [f"{dataset_name}-train-{int(i)}" for i in validation_indices]
    test_ids = [f"{dataset_name}-test-{i}" for i in range(len(test))]
    if quick:
        train, train_ids = _subset(train, train_ids, 1024, seed)
        validation, val_ids = _subset(validation, val_ids, 256, seed + 1)
        test, test_ids = _subset(test, test_ids, 256, seed + 2)
    return DataBundle(lab_id, dataset_name, get_lab(lab_id)["task"], train, validation, test, input_shape, len(class_names),
        class_names=class_names, train_ids=train_ids, validation_ids=val_ids, test_ids=test_ids,
        summary={"train": len(train), "validation": len(validation), "test": len(test), "image_shape": input_shape},
        metadata={"official_test_split": True})


def _tokenize_texts(train_texts: list[str], validation_texts: list[str], test_texts: list[str], max_length: int = 256, vocab_size: int = 20000):
    import re
    from collections import Counter
    def tokens(text: str) -> list[str]:
        return re.findall(r"[A-Za-z0-9']+", text.lower())
    counter = Counter(token for text in train_texts for token in tokens(text))
    vocabulary = {"<pad>": 0, "<unk>": 1}
    for token, _ in counter.most_common(vocab_size - 2):
        vocabulary[token] = len(vocabulary)
    def encode(values: list[str]) -> np.ndarray:
        matrix = np.zeros((len(values), max_length), dtype=np.int64)
        for row, text in enumerate(values):
            encoded = [vocabulary.get(token, 1) for token in tokens(text)[:max_length]]
            matrix[row, :len(encoded)] = encoded
        return matrix
    return encode(train_texts), encode(validation_texts), encode(test_texts), vocabulary


def _load_text(lab_id: str, dataset_id: str, seed: int, quick: bool) -> DataBundle:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError('Instale el extra de texto: pip install -e ".[text]"') from exc
    source = load_dataset(dataset_id)
    train_split = source["train"]
    test_split = source["test"]
    text_field = "text"
    label_field = "label"
    train_texts_all = list(train_split[text_field])
    train_labels_all = np.asarray(train_split[label_field], dtype=np.int64)
    test_texts = list(test_split[text_field])
    test_labels = np.asarray(test_split[label_field], dtype=np.int64)
    rng = np.random.default_rng(seed)
    indices = np.arange(len(train_texts_all))
    rng.shuffle(indices)
    val_n = max(1, int(len(indices) * 0.15))
    val_idx, train_idx = np.sort(indices[:val_n]), np.sort(indices[val_n:])
    train_texts = [train_texts_all[i] for i in train_idx]
    val_texts = [train_texts_all[i] for i in val_idx]
    train_labels = train_labels_all[train_idx]
    val_labels = train_labels_all[val_idx]
    if quick:
        train_sel = _quick_indices(len(train_texts), 1024, seed)
        val_sel = _quick_indices(len(val_texts), 256, seed + 1)
        test_sel = _quick_indices(len(test_texts), 256, seed + 2)
        train_texts = [train_texts[i] for i in train_sel]; train_labels = train_labels[train_sel]; train_idx = train_idx[train_sel]
        val_texts = [val_texts[i] for i in val_sel]; val_labels = val_labels[val_sel]; val_idx = val_idx[val_sel]
        test_texts = [test_texts[i] for i in test_sel]; test_labels = test_labels[test_sel]; test_indices = test_sel
    else:
        test_indices = np.arange(len(test_texts))
    train_x, val_x, test_x, vocab = _tokenize_texts(train_texts, val_texts, test_texts)
    train_ids = [f"{dataset_id}-train-{int(i)}" for i in train_idx]
    val_ids = [f"{dataset_id}-train-{int(i)}" for i in val_idx]
    test_ids = [f"{dataset_id}-test-{int(i)}" for i in test_indices]
    train = ArrayDataset(train_x, train_labels, train_ids)
    validation = ArrayDataset(val_x, val_labels, val_ids)
    test = ArrayDataset(test_x, test_labels, test_ids)
    features = source["train"].features[label_field]
    class_names = list(getattr(features, "names", [])) or [str(i) for i in sorted(set(train_labels.tolist()))]
    return DataBundle(lab_id, get_lab(lab_id)["dataset"], get_lab(lab_id)["task"], train, validation, test, (train_x.shape[1],), len(class_names),
        class_names=class_names, train_ids=train_ids, validation_ids=val_ids, test_ids=test_ids,
        summary={"vocabulary_size": len(vocab), "sequence_length": train_x.shape[1], "train": len(train), "validation": len(validation), "test": len(test)},
        metadata={"vocabulary": vocab, "official_test_split": True},
        raw={"train_texts": train_texts, "validation_texts": val_texts, "test_texts": test_texts, "train_labels": train_labels, "validation_labels": val_labels, "test_labels": test_labels})


def _download_har() -> Path:
    root = RAW_ROOT / "uci_har"
    extracted = root / "UCI HAR Dataset"
    if extracted.exists():
        return extracted
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "uci_har.zip"
    if not archive.exists():
        url = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
        urllib.request.urlretrieve(url, archive)
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(root)
    candidates = list(root.rglob("UCI HAR Dataset"))
    if not candidates:
        raise FileNotFoundError("No se encontró la carpeta UCI HAR Dataset tras extraer el archivo.")
    return candidates[0]


def _load_har_split(root: Path, split: str, raw_signals: bool):
    y = np.loadtxt(root / split / f"y_{split}.txt", dtype=np.int64) - 1
    subjects = np.loadtxt(root / split / f"subject_{split}.txt", dtype=np.int64)
    if raw_signals:
        signal_dir = root / split / "Inertial Signals"
        names = [
            "body_acc_x", "body_acc_y", "body_acc_z",
            "body_gyro_x", "body_gyro_y", "body_gyro_z",
            "total_acc_x", "total_acc_y", "total_acc_z",
        ]
        channels = [np.loadtxt(signal_dir / f"{name}_{split}.txt", dtype=np.float32) for name in names]
        x = np.stack(channels, axis=1)
    else:
        x = np.loadtxt(root / split / f"X_{split}.txt", dtype=np.float32)
    return x, y, subjects


def _load_har(lab_id: str, seed: int, quick: bool) -> DataBundle:
    root = _download_har()
    raw_signals = lab_id == "12_multimodal_fusion"
    train_x_full, train_y_full, train_subjects = _load_har_split(root, "train", raw_signals)
    test_x, test_y, test_subjects = _load_har_split(root, "test", raw_signals)
    # Validation is subject-disjoint: reserve complete participants from official training set.
    unique_subjects = np.unique(train_subjects)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_subjects)
    validation_subjects = set(unique_subjects[:max(1, int(len(unique_subjects) * 0.2))].tolist())
    val_mask = np.asarray([subject in validation_subjects for subject in train_subjects])
    train_mask = ~val_mask
    train_x, train_y, subjects_train = train_x_full[train_mask], train_y_full[train_mask], train_subjects[train_mask]
    val_x, val_y, subjects_val = train_x_full[val_mask], train_y_full[val_mask], train_subjects[val_mask]
    train_ids = [f"har-train-subject{subject}-window{index}" for index, subject in zip(np.where(train_mask)[0], subjects_train)]
    val_ids = [f"har-train-subject{subject}-window{index}" for index, subject in zip(np.where(val_mask)[0], subjects_val)]
    test_ids = [f"har-test-subject{subject}-window{index}" for index, subject in enumerate(test_subjects)]
    train = ArrayDataset(train_x, train_y, train_ids, extra={"subjects": subjects_train})
    validation = ArrayDataset(val_x, val_y, val_ids, extra={"subjects": subjects_val})
    test = ArrayDataset(test_x, test_y, test_ids, extra={"subjects": test_subjects})
    if quick:
        train, train_ids = _subset(train, train_ids, 1024, seed)
        validation, val_ids = _subset(validation, val_ids, 256, seed + 1)
        test, test_ids = _subset(test, test_ids, 256, seed + 2)
    names = [line.strip().split(maxsplit=1)[-1] for line in (root / "activity_labels.txt").read_text().splitlines()]
    input_shape = tuple(train.features.shape[1:])
    return DataBundle(lab_id, get_lab(lab_id)["dataset"], get_lab(lab_id)["task"], train, validation, test, input_shape, len(names),
        class_names=names, train_ids=train_ids, validation_ids=val_ids, test_ids=test_ids,
        summary={"raw_inertial_signals": raw_signals, "train_subjects": sorted(set(map(int, np.asarray(train.extra.get('subjects', []))))), "validation_subjects": sorted(validation_subjects), "official_test_subjects": sorted(set(map(int, test_subjects)))},
        metadata={"official_test_split": True, "subject_disjoint_validation": True})


def _load_kaggle(lab_id: str, handle: str, seed: int, quick: bool) -> DataBundle:
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError('Instale el extra Kaggle: pip install -e ".[kaggle]"') from exc
    output = RAW_ROOT / get_lab(lab_id)["dataset"]
    output.mkdir(parents=True, exist_ok=True)
    downloaded = Path(kagglehub.dataset_download(handle, output_dir=str(output)))
    csv_files = sorted(downloaded.rglob("*.csv")) if downloaded.is_dir() else []
    if not csv_files:
        csv_files = sorted(output.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"KaggleHub no entregó archivos CSV para {handle}")
    preferred = "creditcard.csv" if "creditcard" in handle else "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    path = next((item for item in csv_files if item.name == preferred), csv_files[0])
    frame = pd.read_csv(path)
    if "creditcard" in handle:
        y = frame.pop("Class")
        x = frame
        x.index = [f"creditcard-{i}" for i in range(len(x))]
        y.index = x.index
        if "Time" in x.columns:
            order = np.argsort(x["Time"].to_numpy())
            x, y = x.iloc[order], y.iloc[order]
        split = _frame_split(x, y, seed, chronological=True, stratify=False)
    else:
        target = "Churn"
        y = frame.pop(target)
        x = frame.drop(columns=[column for column in ["customerID"] if column in frame], errors="ignore")
        x.index = [f"telco-{i}" for i in range(len(x))]
        y.index = x.index
        split = _frame_split(x, y, seed, chronological=False, stratify=True)
    x_train, x_val, x_test, y_train, y_val, y_test, train_ids, val_ids, test_ids = split
    encoded = _encode_tabular(x_train, x_val, x_test, y_train, y_val, y_test, classification=True)
    train_x, val_x, test_x, train_y, val_y, test_y, feature_names, class_names, transformer = encoded
    train = ArrayDataset(train_x, train_y, train_ids)
    validation = ArrayDataset(val_x, val_y, val_ids)
    test = ArrayDataset(test_x, test_y, test_ids)
    if quick:
        if "creditcard" in handle:
            # Keep enough observations for the rare fraud class to remain visible
            # without changing the chronological partitions or rebalancing test.
            train_limit, validation_limit, test_limit = 8192, 8192, 8192
        else:
            train_limit, validation_limit, test_limit = 2048, 512, 512
        train, train_ids = _subset(train, train_ids, train_limit, seed)
        validation, val_ids = _subset(validation, val_ids, validation_limit, seed + 1)
        test, test_ids = _subset(test, test_ids, test_limit, seed + 2)
    return DataBundle(lab_id, get_lab(lab_id)["dataset"], get_lab(lab_id)["task"], train, validation, test, (train.features.shape[1],), len(class_names),
        class_names=class_names, feature_names=feature_names, train_ids=train_ids, validation_ids=val_ids, test_ids=test_ids,
        summary={"source_file": str(path), "source_instances": len(frame), "features_after_preprocessing": len(feature_names)},
        metadata={"transformer": transformer, "kaggle_handle": handle},
        raw={"x_train": x_train, "x_validation": x_val, "x_test": x_test, "y_train": y_train, "y_validation": y_val, "y_test": y_test})



def _load_online_retail(lab_id: str, seed: int, quick: bool) -> DataBundle:
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise RuntimeError('Instale el extra de datos: pip install -e ".[data]"') from exc
    fetched = fetch_ucirepo(id=352)
    frame = fetched.data.features.copy()
    required = {"InvoiceDate", "StockCode", "Quantity"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Online Retail no contiene las columnas requeridas: {sorted(missing)}")
    frame["InvoiceDate"] = pd.to_datetime(frame["InvoiceDate"], errors="coerce")
    frame["Quantity"] = pd.to_numeric(frame["Quantity"], errors="coerce")
    frame = frame.dropna(subset=["InvoiceDate", "StockCode", "Quantity"])
    frame = frame[frame["Quantity"] > 0].copy()
    top_stock = str(frame.groupby("StockCode")["Quantity"].sum().idxmax())
    selected = frame[frame["StockCode"].astype(str) == top_stock].copy()
    daily = selected.set_index("InvoiceDate")["Quantity"].resample("D").sum().asfreq("D", fill_value=0.0)
    daily = daily.astype(np.float32)
    if len(daily) < 90:
        raise ValueError("La serie diaria de Online Retail es demasiado corta para train/validation/test.")
    n = len(daily)
    train_end = int(n * 0.70)
    validation_end = int(n * 0.85)
    values = daily.to_numpy(dtype=np.float32).reshape(-1, 1)
    ids = [f"online-retail-{date.date().isoformat()}" for date in daily.index]
    train_values, validation_values, test_values = values[:train_end], values[train_end:validation_end], values[validation_end:]
    train_ids, validation_ids, test_ids = ids[:train_end], ids[train_end:validation_end], ids[validation_end:]
    if quick:
        train_values, train_ids = train_values[-min(180, len(train_values)):], train_ids[-min(180, len(train_ids)):]
        validation_values, validation_ids = validation_values[-min(45, len(validation_values)):], validation_ids[-min(45, len(validation_ids)):]
        test_values, test_ids = test_values[-min(45, len(test_values)):], test_ids[-min(45, len(test_ids)):]
    train = ArrayDataset(train_values, train_values.reshape(-1), train_ids)
    validation = ArrayDataset(validation_values, validation_values.reshape(-1), validation_ids)
    test = ArrayDataset(test_values, test_values.reshape(-1), test_ids)
    return DataBundle(
        lab_id=lab_id, dataset_name="online_retail", task="reinforcement_learning",
        train=train, validation=validation, test=test, input_shape=(4,), num_classes=4,
        class_names=["no_order", "small_order", "medium_order", "large_order"],
        train_ids=train_ids, validation_ids=validation_ids, test_ids=test_ids,
        summary={
            "source_transactions": int(len(frame)),
            "selected_stock_code": top_stock,
            "daily_observations": int(n),
            "split": "chronological",
            "demand_mean": float(daily.mean()),
            "demand_std": float(daily.std()),
        },
        metadata={"source_metadata": fetched.metadata, "official_source": "UCI Online Retail"},
        raw={
            "train_demand": train_values.reshape(-1),
            "validation_demand": validation_values.reshape(-1),
            "test_demand": test_values.reshape(-1),
        },
    )

def _load_cora(lab_id: str) -> DataBundle:
    try:
        from torch_geometric.datasets import Planetoid
    except ImportError as exc:
        raise RuntimeError('Instale el extra de grafos: pip install -e ".[graph]"') from exc
    dataset = Planetoid(root=str(RAW_ROOT / "cora"), name="Cora", split="public")
    graph = dataset[0]
    train_ids = [f"cora-node-{i}" for i in graph.train_mask.nonzero(as_tuple=False).view(-1).tolist()]
    val_ids = [f"cora-node-{i}" for i in graph.val_mask.nonzero(as_tuple=False).view(-1).tolist()]
    test_ids = [f"cora-node-{i}" for i in graph.test_mask.nonzero(as_tuple=False).view(-1).tolist()]
    return DataBundle(lab_id, "cora", "node_classification", graph, graph, graph, (dataset.num_node_features,), dataset.num_classes,
        class_names=[str(i) for i in range(dataset.num_classes)], train_ids=train_ids, validation_ids=val_ids, test_ids=test_ids,
        summary={"nodes": int(graph.num_nodes), "edges": int(graph.num_edges), "features": int(dataset.num_node_features), "classes": int(dataset.num_classes)},
        metadata={"official_public_masks": True})


def prepare_dataset(lab_id: str, *, quick: bool = False, seed: int = 42, force: bool = False) -> DataBundle:
    del force  # download libraries manage their own caches
    lab = get_lab(lab_id)
    dataset = get_dataset(lab_id)
    source_type = dataset["source_type"]
    if source_type == "uci":
        bundle = _load_uci(lab_id, int(dataset["source_id"]), seed, quick)
    elif source_type == "sklearn":
        bundle = _load_california(lab_id, seed, quick)
    elif source_type == "torchvision":
        bundle = _load_vision(lab_id, lab["dataset"], seed, quick)
    elif source_type == "huggingface":
        bundle = _load_text(lab_id, str(dataset["source_id"]), seed, quick)
    elif source_type == "uci_har":
        bundle = _load_har(lab_id, seed, quick)
    elif source_type == "uci_retail":
        bundle = _load_online_retail(lab_id, seed, quick)
    elif source_type == "kaggle":
        bundle = _load_kaggle(lab_id, str(dataset["source_id"]), seed, quick)
    elif source_type == "pyg":
        bundle = _load_cora(lab_id)
    else:
        raise NotImplementedError(f"Fuente no implementada: {source_type}")
    if bundle.task != "reinforcement_learning":
        audit_bundle(bundle)
    _persist_split_manifest(bundle, seed)
    bundle.metadata.update({"source": dataset["source"], "source_url": dataset["source_ref"], "license": dataset["license"], "real_world_data": True, "generated_data": False})
    return bundle


def describe_bundle(bundle: DataBundle) -> dict[str, Any]:
    description = {
        "lab": bundle.lab_id,
        "dataset": bundle.dataset_name,
        "task": bundle.task,
        "input_shape": bundle.input_shape,
        "num_classes": bundle.num_classes,
        "classes": bundle.class_names,
        "counts": {
            "train": len(bundle.train_ids),
            "validation": len(bundle.validation_ids),
            "test": len(bundle.test_ids),
        },
        "summary": bundle.summary,
        "audit": audit_bundle(bundle) if bundle.task != "reinforcement_learning" else {"ok": True, "note": "El entorno recolecta transiciones en línea."},
        "dataset_hash": bundle.metadata.get("dataset_hash"),
    }
    print(json.dumps(description, indent=2, ensure_ascii=False, default=str))
    return description
