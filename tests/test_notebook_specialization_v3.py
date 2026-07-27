from __future__ import annotations

import hashlib
from pathlib import Path

import nbformat

from neural_labs.catalog import ROOT, list_labs
from neural_labs.advanced.catalog import list_tracks


def test_all_notebooks_are_distinct_and_evaluable() -> None:
    folders = [ROOT / "labs" / lab for lab in list_labs()] + [ROOT / "advanced_labs" / track for track in list_tracks()]
    hashes: set[str] = set()
    for folder in folders:
        main = nbformat.read(folder / "notebook.ipynb", as_version=4)
        student = nbformat.read(folder / "notebook_student.ipynb", as_version=4)
        solution = nbformat.read(folder / "notebook_solution.ipynb", as_version=4)
        assert len(main.cells) >= 18
        assert len(student.cells) == len(solution.cells)
        assert sum(cell.cell_type == "code" for cell in main.cells) >= 8
        assert any("nbgrader" in cell.metadata for cell in student.cells)
        hashes.add(hashlib.sha256((folder / "notebook_solution.ipynb").read_bytes()).hexdigest())
    assert len(folders) == 31
    assert len(hashes) == len(folders)
