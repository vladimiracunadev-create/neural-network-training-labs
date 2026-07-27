from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from ..catalog import ROOT


@dataclass
class AdvancedData:
    train: Any
    validation: Any
    test: Any
    classes: int | None
    class_names: list[str]
    metadata: dict[str, Any]


def _limited(dataset: Any, maximum: int | None) -> Any:
    if maximum is None or len(dataset) <= maximum:
        return dataset
    return Subset(dataset, list(range(maximum)))


def _split_train_validation(dataset: Any, split_seed: int, validation_fraction: float = 0.15) -> tuple[Any, Any]:
    validation_size = max(1, int(len(dataset) * validation_fraction))
    train_size = len(dataset) - validation_size
    return random_split(dataset, [train_size, validation_size], generator=torch.Generator().manual_seed(split_seed))


class PetSegmentationDataset(Dataset):
    def __init__(self, base: Any, image_size: int = 128):
        self.base = base
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as F

        image, mask = self.base[index]
        image = F.resize(image, [self.image_size, self.image_size], antialias=True)
        image = F.to_tensor(image)
        mask = F.resize(mask, [self.image_size, self.image_size], interpolation=InterpolationMode.NEAREST)
        mask = torch.from_numpy(np.asarray(mask, dtype=np.int64).copy())
        mask = mask.reshape(self.image_size, self.image_size).long().clamp(1, 3) - 1
        return image, mask


class SpeechCommandsSubset(Dataset):
    def __init__(self, root: Path, subset: str):
        try:
            from torchaudio.datasets import SPEECHCOMMANDS
        except ImportError as exc:
            raise RuntimeError('Instale el extra audio: pip install -e ".[audio]"') from exc
        self.dataset = SPEECHCOMMANDS(str(root), download=True)
        base = Path(self.dataset._path)
        validation = {line.strip() for line in (base / "validation_list.txt").read_text().splitlines()}
        testing = {line.strip() for line in (base / "testing_list.txt").read_text().splitlines()}

        def relative_name(path: str) -> str:
            candidate = Path(path)
            try:
                return candidate.relative_to(base).as_posix()
            except ValueError:
                return candidate.as_posix()

        if subset == "validation":
            self.walker = [path for path in self.dataset._walker if relative_name(path) in validation]
        elif subset == "testing":
            self.walker = [path for path in self.dataset._walker if relative_name(path) in testing]
        else:
            self.walker = [
                path for path in self.dataset._walker
                if relative_name(path) not in validation and relative_name(path) not in testing
            ]
        self.dataset._walker = self.walker

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        waveform, sample_rate, label, *_rest = self.dataset[index]
        desired = sample_rate
        if waveform.shape[-1] < desired:
            waveform = torch.nn.functional.pad(waveform, (0, desired - waveform.shape[-1]))
        waveform = waveform[..., :desired]
        return waveform, label


class LabelMappedDataset(Dataset):
    def __init__(self, base: Dataset, labels: list[str]):
        self.base = base
        self.mapping = {label: index for index, label in enumerate(labels)}

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        features, label = self.base[index]
        return features, self.mapping[label]


class TwoViewDataset(Dataset):
    def __init__(self, base: Any, transform: Any):
        self.base = base
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image, label = self.base[index]
        return self.transform(image), self.transform(image), label


def load_advanced_data(track_id: str, *, quick: bool, split_seed: int) -> AdvancedData:
    raw = ROOT / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    if track_id == "26_segmentation_unet":
        try:
            from torchvision.datasets import OxfordIIITPet
        except ImportError as exc:
            raise RuntimeError('Instale el extra vision: pip install -e ".[vision]"') from exc
        trainval = PetSegmentationDataset(OxfordIIITPet(raw, split="trainval", target_types="segmentation", download=True))
        test = PetSegmentationDataset(OxfordIIITPet(raw, split="test", target_types="segmentation", download=True))
        train, validation = _split_train_validation(trainval, split_seed)
        if quick:
            train, validation, test = _limited(train, 128), _limited(validation, 48), _limited(test, 48)
        return AdvancedData(train, validation, test, 3, ["pet", "background", "border"], {"source": "Oxford-IIIT Pet", "official_test": True})
    if track_id == "27_audio_speechcommands":
        training = SpeechCommandsSubset(raw / "speechcommands", "training")
        validation = SpeechCommandsSubset(raw / "speechcommands", "validation")
        testing = SpeechCommandsSubset(raw / "speechcommands", "testing")
        labels = sorted({training[index][1] for index in range(len(training))})
        train = LabelMappedDataset(training, labels)
        validation = LabelMappedDataset(validation, labels)
        test = LabelMappedDataset(testing, labels)
        if quick:
            train, validation, test = _limited(train, 512), _limited(validation, 128), _limited(test, 128)
        return AdvancedData(train, validation, test, len(labels), labels, {"source": "SpeechCommands v0.02", "sample_rate": 16000})
    if track_id in {"28_wgan_gp", "29_diffusion_ddpm"}:
        from torchvision import transforms
        from torchvision.datasets import FashionMNIST

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        trainval = FashionMNIST(raw, train=True, download=True, transform=transform)
        test = FashionMNIST(raw, train=False, download=True, transform=transform)
        train, validation = _split_train_validation(trainval, split_seed)
        if quick:
            train, validation, test = _limited(train, 1024), _limited(validation, 256), _limited(test, 256)
        return AdvancedData(train, validation, test, 10, list(trainval.classes), {"source": "Fashion-MNIST", "official_test": True})
    if track_id == "30_self_supervised_simclr":
        from torchvision import transforms
        from torchvision.datasets import CIFAR10

        augmentation = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
        ])
        evaluation = transforms.ToTensor()
        base_train = CIFAR10(raw, train=True, download=True)
        ssl = TwoViewDataset(base_train, augmentation)
        train, validation = _split_train_validation(ssl, split_seed)
        test = CIFAR10(raw, train=False, download=True, transform=evaluation)
        if quick:
            train, validation, test = _limited(train, 1024), _limited(validation, 256), _limited(test, 512)
        return AdvancedData(train, validation, test, 10, list(base_train.classes), {"source": "CIFAR-10", "official_test": True})
    if track_id == "25_transformer_finetuning":
        try:
            from datasets import load_dataset
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError('Instale el extra text-modern: pip install -e ".[text-modern]"') from exc
        dataset = load_dataset("ag_news")
        split = dataset["train"].train_test_split(test_size=0.1, seed=split_seed, stratify_by_column="label")
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        def tokenize(batch):
            return tokenizer(batch["text"], truncation=True, max_length=192)
        train = split["train"].map(tokenize, batched=True, remove_columns=["text"])
        validation = split["test"].map(tokenize, batched=True, remove_columns=["text"])
        test = dataset["test"].map(tokenize, batched=True, remove_columns=["text"])
        train.set_format("torch"); validation.set_format("torch"); test.set_format("torch")
        if quick:
            train = train.select(range(min(1000, len(train))))
            validation = validation.select(range(min(256, len(validation))))
            test = test.select(range(min(256, len(test))))
        return AdvancedData(train, validation, test, 4, ["World", "Sports", "Business", "Sci/Tech"], {"source": "AG News", "tokenizer": tokenizer})
    raise KeyError(track_id)


def loader(dataset: Any, batch_size: int, shuffle: bool, *, collate_fn: Any = None) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, collate_fn=collate_fn)
