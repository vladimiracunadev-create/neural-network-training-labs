from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from neural_labs.advanced.catalog import get_track, list_tracks, tracks
from neural_labs.advanced.datasets import AdvancedData, PetSegmentationDataset, _limited, _split_train_validation
from neural_labs.domains.generative.advanced import TinyDenoiser, add_noise, cosine_beta_schedule, gradient_penalty
from neural_labs.domains.vision.segmentation import UNetSmall, mean_iou
from neural_labs.domains.vision.self_supervised import nt_xent_loss


def test_advanced_catalog_has_six_real_tracks() -> None:
    identifiers = list_tracks()
    assert len(identifiers) == 6
    assert identifiers[0] == "25_transformer_finetuning"
    assert all(item["dataset"] and item["source"] for item in tracks())
    assert get_track("30_self_supervised_simclr")["domain"] == "vision"
    with pytest.raises(KeyError):
        get_track("missing")


def test_advanced_dataset_helpers() -> None:
    dataset = TensorDataset(torch.arange(40).reshape(20, 2), torch.arange(20))
    train, validation = _split_train_validation(dataset, 42, validation_fraction=0.2)
    assert len(train) == 16 and len(validation) == 4
    assert len(_limited(dataset, 5)) == 5
    assert _limited(dataset, None) is dataset


def test_unet_and_iou() -> None:
    model = UNetSmall(base=4)
    images = torch.randn(2, 3, 31, 29)
    logits = model(images)
    assert logits.shape == (2, 3, 31, 29)
    targets = logits.argmax(1)
    assert mean_iou(logits, targets, 3) == pytest.approx(1.0)


def test_diffusion_and_wgan_primitives() -> None:
    betas = cosine_beta_schedule(20)
    assert betas.shape == (20,)
    clean = torch.randn(4, 1, 28, 28)
    noisy = add_noise(clean, torch.randn_like(clean), torch.tensor([0, 1, 2, 3]), betas)
    assert noisy.shape == clean.shape
    denoiser = TinyDenoiser(hidden=16, time_dim=16)
    assert denoiser(noisy, torch.tensor([0, 1, 2, 3])).shape == clean.shape

    critic = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(28 * 28, 1))
    penalty = gradient_penalty(critic, clean, torch.randn_like(clean))
    assert penalty.ndim == 0 and torch.isfinite(penalty)


def test_contrastive_loss() -> None:
    first = torch.nn.functional.normalize(torch.randn(8, 16), dim=1)
    second = torch.nn.functional.normalize(torch.randn(8, 16), dim=1)
    loss = nt_xent_loss(first, second)
    assert torch.isfinite(loss)
    with pytest.raises(ValueError):
        nt_xent_loss(first, second[:, :8])


def test_advanced_training_dispatch_writes_lock(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import neural_labs.advanced.training as module

    dataset = TensorDataset(torch.randn(12, 3), torch.randint(0, 2, (12,)))
    data = AdvancedData(dataset, dataset, dataset, 2, ["a", "b"], {"source": "real-fixture"})
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "load_advanced_data", lambda *_args, **_kwargs: data)
    monkeypatch.setattr(module, "get_device", lambda _device: torch.device("cpu"))

    def fake_train(_data, run_dir, _device, _quick, freeze):
        torch.save({"state_dict": {}}, run_dir / "best_model.pt")
        lock_path = freeze()
        assert lock_path.is_file()
        return {"validation_mean_iou": 0.5, "test_mean_iou": 0.4}

    monkeypatch.setattr(module, "_train_segmentation", fake_train)
    result = module.train_advanced("26_segmentation_unet", quick=True, output_dir="runs")
    run_dir = Path(result["run_dir"])
    assert (run_dir / "experiment.lock.json").is_file()
    assert json.loads((run_dir / "metrics.json").read_text())["track"] == "26_segmentation_unet"
