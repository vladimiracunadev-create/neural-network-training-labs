#!/usr/bin/env python3
"""Añade los cinco ejercicios evaluables a los cuadernos de las especializaciones.

Los cuadernos de `advanced_labs/` se escribieron a mano y no tienen generador, así
que este script **no los reescribe**: conserva su contenido tal cual y solo
gestiona el bloque de ejercicios, delimitado por celdas con identificadores
nbgrader conocidos. Es idempotente —retira el bloque anterior antes de insertar el
nuevo—, de modo que se puede volver a ejecutar cuando cambien los ejercicios.

También separa las tres variantes, que hasta ahora eran dos:

* `notebook.ipynb` — el recorrido, sin ejercicios.
* `notebook_student.ipynb` — con los cinco ejercicios por resolver.
* `notebook_solution.ipynb` — con los cinco resueltos.

    python scripts/generate_advanced_exercises.py
    python scripts/generate_advanced_exercises.py --check    # falla si algo quedó desfasado
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import nbformat
import yaml
from nbformat.v4 import new_code_cell, new_markdown_cell

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lab_exercises import exercises, graded_metadata, test_metadata  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TRACKS = yaml.safe_load((ROOT / "configs" / "advanced_tracks.yaml").read_text(encoding="utf-8"))["tracks"]
CLASS_PROFILES = {item["id"]: item for item in yaml.safe_load((ROOT / "configs" / "classes.yaml").read_text(encoding="utf-8"))["classes"]}

# Marcadores del bloque gestionado por este script.
INTRO_MARK = "<!-- ejercicios-evaluables -->"
WALKTHROUGH_MARK = "<!-- sin-ejercicios -->"
CLASS_INTRO_MARK = "<!-- identidad-de-clase -->"


def _managed_ids() -> set[str]:
    ids: set[str] = set()
    for exercise in exercises(metric="loss", baseline="x", experiment="x", dataset="x"):
        ids.add(exercise["id"])
        ids.add(f"{exercise['id']}_test")
    return ids


def strip_block(cells: list[Any]) -> list[Any]:
    """Quita el bloque de ejercicios insertado por una ejecución anterior."""
    managed = _managed_ids()
    kept = []
    for cell in cells:
        grade_id = (cell.get("metadata", {}).get("nbgrader") or {}).get("grade_id")
        source = "".join(cell.get("source") or "")
        cell_id = str(cell.get("id") or "")
        # Los enunciados son celdas de markdown sin metadatos nbgrader: se
        # reconocen por su identificador, que este script fija de forma estable.
        if (grade_id in managed
                or cell_id.startswith("ej-")
                or cell_id.startswith("ejercicios-")
                or INTRO_MARK in source
                or WALKTHROUGH_MARK in source
                or CLASS_INTRO_MARK in source):
            continue
        kept.append(cell)
    return kept


def add_class_intro(cells: list[Any], track_id: str) -> list[Any]:
    profile = CLASS_PROFILES[track_id]
    number = int(profile["class_number"])
    intro = new_markdown_cell(
        f"{CLASS_INTRO_MARK}\n## Clase {number:02d} · Antes de ejecutar\n\n"
        f"{profile['hook']}\n\n> **Pregunta esencial:** {profile['essential_question']}\n\n"
        f"![{profile['visual_title']}](assets/class-map.svg)\n\n"
        f"**Práctica propia:** {profile['practice']}\n\n"
        f"> **Error conceptual que debes poder detectar:** {profile['misconception']}"
    )
    intro.id = "identidad-de-clase"
    return cells[:1] + [intro] + cells[1:]


def exercise_cells(track: dict[str, Any], metric: str, *, solution: bool) -> list[Any]:
    # Los identificadores de celda se fijan a mano: nbformat asigna uno aleatorio a
    # cada celda nueva, y con eso el script nunca sería idempotente ni el diff
    # legible.
    intro = new_markdown_cell(
        f"{INTRO_MARK}\n## Ejercicios evaluables\n\nCinco ejercicios sobre el contrato experimental "
        "de esta especialización. Se resuelven con Python estándar: **no hace falta descargar el "
        "dataset ni entrenar**, y cada uno trae debajo una celda de comprobación que debe pasar sin "
        "error."
    )
    intro.id = "ejercicios-intro"
    cells = [intro]
    for exercise in exercises(
        metric=metric,
        baseline=str(track.get("baseline") or "la línea base declarada"),
        experiment=str(track.get("experiment") or track.get("objective") or ""),
        dataset=str(track.get("dataset") or ""),
    ):
        statement = new_markdown_cell(f"### {exercise['title']}\n\n{exercise['statement']}")
        statement.id = f"ej-{exercise['id']}-md"
        body = ("# SOLUCIÓN DE REFERENCIA\n" + exercise["solution"]) if solution \
            else ("# YOUR CODE HERE\n" + exercise["student"])
        answer = new_code_cell(
            body, metadata=graded_metadata(exercise["id"], exercise["points"], solution)
        )
        answer.id = f"ej-{exercise['id']}"
        check = new_code_cell(
            exercise["test"], metadata=test_metadata(f"{exercise['id']}_test", exercise["points"])
        )
        check.id = f"ej-{exercise['id']}-test"
        cells += [statement, answer, check]
    return cells


def selection_metric(track_id: str) -> str:
    config = yaml.safe_load(
        (ROOT / "advanced_labs" / track_id / "configs" / "baseline.yaml").read_text(encoding="utf-8")
    )
    return str((config or {}).get("selection_metric") or "loss")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="no escribe: falla si algún cuaderno quedó desfasado.")
    args = parser.parse_args()

    stale: list[str] = []
    written = 0
    for track in TRACKS:
        track_id = track["id"]
        folder = ROOT / "advanced_labs" / track_id
        metric = selection_metric(track_id)

        student = nbformat.read(folder / "notebook_student.ipynb", as_version=4)
        solution = nbformat.read(folder / "notebook_solution.ipynb", as_version=4)
        walkthrough = nbformat.read(folder / "notebook.ipynb", as_version=4)

        base_student = add_class_intro(strip_block(list(student.cells)), track_id)
        base_solution = add_class_intro(strip_block(list(solution.cells)), track_id)
        base_walkthrough = add_class_intro(strip_block(list(walkthrough.cells)), track_id)

        student.cells = base_student + exercise_cells(track, metric, solution=False)
        solution.cells = base_solution + exercise_cells(track, metric, solution=True)
        nota = new_markdown_cell(
            f"{WALKTHROUGH_MARK}\n## Ejercicios\n\nEste cuaderno es el recorrido de referencia y no "
            "trae ejercicios. La práctica está en `notebook_student.ipynb` —cinco ejercicios "
            "evaluables sobre el contrato experimental— y su corrección, en "
            "`notebook_solution.ipynb`."
        )
        nota.id = "ejercicios-nota"
        walkthrough.cells = base_walkthrough + [nota]

        for notebook, name in (
            (student, "notebook_student.ipynb"),
            (solution, "notebook_solution.ipynb"),
            (walkthrough, "notebook.ipynb"),
        ):
            path = folder / name
            # Se comparan las dos serializaciones con el mismo escritor: `write`
            # añade un salto final que `writes` no pone, y comparar el texto crudo
            # daría siempre distinto.
            new = nbformat.writes(notebook)
            if nbformat.writes(nbformat.read(path, as_version=4)) == new:
                continue
            if args.check:
                stale.append(f"advanced_labs/{track_id}/{name}")
            else:
                nbformat.write(notebook, path)
                written += 1

    if args.check:
        if stale:
            print("Cuadernos avanzados desfasados (ejecuta scripts/generate_advanced_exercises.py):")
            for item in stale:
                print(f"  - {item}")
            return 1
        print(f"Cuadernos avanzados al día: {len(TRACKS)} especializaciones.")
        return 0

    print(f"Cuadernos avanzados actualizados: {written} archivos en {len(TRACKS)} especializaciones.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
