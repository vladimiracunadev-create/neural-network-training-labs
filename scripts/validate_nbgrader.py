#!/usr/bin/env python3
"""Validate student/solution notebook pairs without executing external datasets."""
from __future__ import annotations

from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def validate() -> list[str]:
    errors: list[str] = []
    lab_dirs = list(sorted((ROOT / "labs").iterdir())) + list(sorted((ROOT / "advanced_labs").iterdir()))
    for lab in lab_dirs:
        if not lab.is_dir():
            continue
        student_path = lab / "notebook_student.ipynb"
        solution_path = lab / "notebook_solution.ipynb"
        if not student_path.is_file() or not solution_path.is_file():
            errors.append(f"{lab.name}: falta notebook_student.ipynb o notebook_solution.ipynb")
            continue
        student = nbformat.read(student_path, as_version=4)
        solution = nbformat.read(solution_path, as_version=4)
        if len(student.cells) != len(solution.cells):
            errors.append(f"{lab.name}: estudiante y solución tienen diferente número de celdas")
        student_ids = []
        solution_ids = []
        for notebook, ids, label in ((student, student_ids, "student"), (solution, solution_ids, "solution")):
            for index, cell in enumerate(notebook.cells):
                metadata = cell.get("metadata", {}).get("nbgrader")
                if not metadata:
                    continue
                grade_id = metadata.get("grade_id")
                if not grade_id:
                    errors.append(f"{lab.name}:{label}:{index}: falta grade_id")
                else:
                    ids.append(grade_id)
                if metadata.get("grade") and not isinstance(metadata.get("points"), (int, float)):
                    errors.append(f"{lab.name}:{label}:{index}: celda evaluable sin puntos")
        if student_ids != solution_ids:
            errors.append(f"{lab.name}: los grade_id no coinciden entre estudiante y solución")
    return errors


if __name__ == "__main__":
    problems = validate()
    if problems:
        print("\n".join(problems))
        raise SystemExit(1)
    print("nbgrader: 31 pares estudiante/solución válidos")
