import json
from pathlib import Path

import pytest

from neural_labs.core.protocol import ExperimentLock, SeedPlan, assert_lock_matches, stable_payload_hash
from neural_labs.core.registry import FactoryRegistry, RegistryError


def test_seed_plan_and_lock_round_trip(tmp_path: Path) -> None:
    seeds = SeedPlan.resolve(split_seed=7, training_seed=9)
    assert seeds.split_seed == 7
    assert seeds.training_seed == 9
    lock = ExperimentLock.create(
        lab_id="lab",
        seeds=seeds,
        config_name="baseline",
        selection_metric="f1",
        selected_checkpoint=tmp_path / "best_model.pt",
        dataset_hash="abc",
    )
    path = lock.write(tmp_path)
    assert path.exists()
    payload = assert_lock_matches(tmp_path, lab_id="lab")
    assert payload["status"] == "frozen_before_test"
    assert stable_payload_hash({"b": 2, "a": 1}) == stable_payload_hash({"a": 1, "b": 2})


def test_lock_validation_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        assert_lock_matches(tmp_path)
    (tmp_path / "experiment.lock.json").write_text(json.dumps({"status": "draft", "lab_id": "x"}))
    with pytest.raises(ValueError):
        assert_lock_matches(tmp_path)


def test_factory_registry_create_describe_and_duplicates() -> None:
    registry = FactoryRegistry("demo")

    @registry.register("one", domain="test", description="factory")
    def make_one(value=1):
        return value

    assert registry.create("one", value=3) == 3
    assert registry.get("one").domain == "test"
    assert registry.names() == ["one"]
    assert registry.describe()[0]["description"] == "factory"
    with pytest.raises(RegistryError):
        registry.register("one", domain="test")(lambda: 2)
    with pytest.raises(RegistryError):
        registry.create("missing")
