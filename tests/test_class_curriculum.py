from __future__ import annotations

import nbformat
import yaml

from neural_labs.catalog import ROOT


def _profiles() -> list[dict]:
    payload = yaml.safe_load((ROOT / "configs" / "classes.yaml").read_text(encoding="utf-8"))
    return payload["classes"]


def _folder(class_id: str):
    base = "advanced_labs" if int(class_id.split("_", 1)[0]) >= 25 else "labs"
    return ROOT / base / class_id


def test_the_course_has_31_numbered_and_distinct_classes() -> None:
    profiles = _profiles()
    assert [item["class_number"] for item in profiles] == list(range(1, 32))
    assert len({item["id"] for item in profiles}) == 31
    assert len({item["essential_question"] for item in profiles}) == 31
    assert len({item["practice"] for item in profiles}) == 31
    assert len({item["misconception"] for item in profiles}) == 31


def test_every_class_is_self_contained_and_uses_the_unified_schema() -> None:
    for profile in _profiles():
        folder = _folder(profile["id"])
        lesson = yaml.safe_load((folder / "lesson.yaml").read_text(encoding="utf-8"))
        assert lesson["schema_version"] == "3.0"
        assert lesson["id"] == profile["id"]
        assert lesson["class_number"] == profile["class_number"]
        assert lesson["essential_question"] == profile["essential_question"]
        assert (folder / "assets" / "class-map.svg").exists()
        assert (folder / "instructor-guide.md").exists()


def test_every_notebook_names_the_class_and_embeds_its_visual_map() -> None:
    for profile in _profiles():
        notebook = nbformat.read(_folder(profile["id"]) / "notebook.ipynb", as_version=4)
        source = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "markdown")
        assert f"Clase {profile['class_number']:02d}" in source
        assert "assets/class-map.svg" in source
        assert profile["essential_question"] in source
