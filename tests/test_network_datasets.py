import os

import pytest

from neural_labs.datasets import audit_bundle, prepare_dataset

pytestmark = pytest.mark.network


@pytest.mark.skipif(os.environ.get("RUN_NETWORK_TESTS") != "1", reason="requiere descarga externa explícita")
@pytest.mark.parametrize("lab_id", ["00_numpy_neuron", "03_cnn_vision"])
def test_real_dataset_download_and_split(lab_id: str) -> None:
    bundle = prepare_dataset(lab_id, quick=True, seed=42)
    assert bundle.metadata["real_world_data"] is True
    assert audit_bundle(bundle)["ok"] is True
    assert len(bundle.train_ids) > 0
    assert len(bundle.validation_ids) > 0
    assert len(bundle.test_ids) > 0
