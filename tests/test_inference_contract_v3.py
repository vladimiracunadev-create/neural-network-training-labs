import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from sklearn.preprocessing import StandardScaler

from neural_labs.datasets import ArrayDataset, DataBundle
from neural_labs.inference import load_external_input, persist_inference_contract


def test_persist_contract_and_image_csv_inputs(tmp_path: Path) -> None:
    x = np.arange(24, dtype=np.float32).reshape(6, 4)
    y = np.array([0, 1, 0, 1, 0, 1])
    ids = [str(i) for i in range(6)]
    scaler = StandardScaler().fit(pd.DataFrame(x[:3], columns=list("abcd")))
    bundle = DataBundle(
        "demo", "real", "binary_classification",
        ArrayDataset(x[:3], y[:3], ids[:3]),
        ArrayDataset(x[3:5], y[3:5], ids[3:5]),
        ArrayDataset(x[5:], y[5:], ids[5:]),
        (4,), 2, class_names=["no", "yes"], feature_names=list("abcd"),
        train_ids=ids[:3], validation_ids=ids[3:5], test_ids=ids[5:],
        metadata={"transformer": scaler, "vocabulary": {"hello": 1}},
    )
    path = persist_inference_contract(bundle, tmp_path, architecture="mlp")
    contract = json.loads(path.read_text())
    assert (tmp_path / contract["preprocessor_file"]).exists()
    assert (tmp_path / contract["vocabulary_file"]).exists()

    csv_path = tmp_path / "sample.csv"
    pd.DataFrame([[1, 2, 3, 4]], columns=list("abcd")).to_csv(csv_path, index=False)
    # contract path resolution for preprocessors is intentionally caller-controlled;
    # use a contract without preprocessor to validate raw CSV loading.
    raw_contract = {"input_shape": [4]}
    assert load_external_input(csv_path, raw_contract).shape == (1, 4)

    image_path = tmp_path / "image.png"
    Image.new("RGB", (8, 8), color="white").save(image_path)
    assert load_external_input(image_path, {"input_shape": [3, 4, 4]}).shape == (1, 3, 4, 4)
    with pytest.raises(ValueError):
        load_external_input(tmp_path / "unknown.xyz", raw_contract)
