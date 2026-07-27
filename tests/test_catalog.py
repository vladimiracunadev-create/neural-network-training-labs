from pathlib import Path

from neural_labs.catalog import ROOT, get_dataset, list_labs


def test_catalog_has_25_labs_and_19_real_datasets() -> None:
    labs = list_labs()
    assert len(labs) == 25
    records = [get_dataset(lab_id) for lab_id in labs]
    assert len({record["dataset"] for record in records}) == 19
    assert all(record["source_type"] != "synthetic" for record in records)
    assert all(record["source_type"] != "gymnasium" for record in records)


def test_no_legacy_synthetic_module() -> None:
    assert not (Path(ROOT) / "src/neural_labs/synthetic.py").exists()
