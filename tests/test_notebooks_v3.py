import hashlib

import nbformat

from neural_labs.catalog import ROOT, list_labs


def test_student_and_solution_notebooks_are_specialized() -> None:
    solution_hashes = set()
    domains = set()
    for lab_id in list_labs():
        folder = ROOT / "labs" / lab_id
        for filename in ["notebook.ipynb", "notebook_student.ipynb", "notebook_solution.ipynb"]:
            assert (folder / filename).is_file()
        student = nbformat.read(folder / "notebook_student.ipynb", as_version=4)
        solution = nbformat.read(folder / "notebook_solution.ipynb", as_version=4)
        assert len(student.cells) >= 20
        assert len(solution.cells) >= 20
        assert student.metadata["neural_labs"]["variant"] == "student"
        assert solution.metadata["neural_labs"]["variant"] == "solution"
        assert any("nbgrader" in cell.metadata for cell in student.cells)
        domains.add(solution.metadata["neural_labs"]["domain"])
        solution_hashes.add(hashlib.sha256((folder / "notebook_solution.ipynb").read_bytes()).hexdigest())
    assert len(solution_hashes) == len(list_labs())
    assert {"vision", "text", "time_series", "graph", "generative", "reinforcement", "tabular", "multimodal"} <= domains
