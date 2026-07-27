import nbformat
import yaml

from neural_labs.catalog import ROOT, get_dataset, list_labs


def test_every_lab_has_documentation_config_notebook_and_manifest() -> None:
    for lab_id in list_labs():
        folder = ROOT / "labs" / lab_id
        for relative in ["README.md", "train.py", "notebook.ipynb", "configs/baseline.yaml", "configs/improved.yaml", "data/dataset.yaml"]:
            assert (folder / relative).exists(), (lab_id, relative)
        notebook = nbformat.read(folder / "notebook.ipynb", as_version=4)
        assert len(notebook.cells) >= 15
        assert sum(cell.cell_type == "code" for cell in notebook.cells) >= 7
        config = yaml.safe_load((folder / "configs/baseline.yaml").read_text(encoding="utf-8"))
        assert config["test_policy"] == "evaluate_once_after_model_selection"
        manifest = yaml.safe_load((folder / "data/dataset.yaml").read_text(encoding="utf-8"))
        assert manifest["real_world_data"] is True
        assert manifest["generated_data"] is False
        assert manifest["source_url"] == get_dataset(lab_id)["source_ref"]


def test_no_large_dataset_files_are_committed_inside_labs() -> None:
    forbidden_suffixes = {".csv", ".parquet", ".npz", ".pt", ".pth", ".onnx"}
    found = [path for path in (ROOT / "labs").rglob("*") if path.is_file() and path.suffix.lower() in forbidden_suffixes]
    assert found == []
