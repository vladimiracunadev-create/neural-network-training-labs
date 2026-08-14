"""Los cinco ejercicios de cada laboratorio existen y su solución pasa.

Un ejercicio cuya solución de referencia no ejecuta es peor que no tenerlo: el
estudiante no puede saber si se equivocó él o el material. Por eso la prueba no
se limita a comprobar que las celdas están —ejecuta cada solución y su celda de
comprobación en un espacio de nombres limpio—.

Los ejercicios se resuelven con la biblioteca estándar, así que esto corre en
segundos y sin descargar ningún dataset.
"""
from __future__ import annotations

import contextlib
import io

import nbformat
import pytest

from neural_labs.catalog import ROOT

EJERCICIOS = [
    "auditoria_particiones",
    "regla_de_seleccion",
    "comparacion_linea_base",
    "sellado_del_test",
    "plan_experimental",
]


def _laboratorios() -> list:
    carpetas = [path for path in sorted((ROOT / "labs").iterdir()) if path.is_dir()]
    carpetas += [path for path in sorted((ROOT / "advanced_labs").iterdir()) if path.is_dir()]
    return carpetas


def _grade_ids(notebook) -> list[str]:
    return [
        cell.metadata["nbgrader"]["grade_id"]
        for cell in notebook.cells
        if "nbgrader" in cell.metadata
    ]


@pytest.mark.parametrize("carpeta", _laboratorios(), ids=lambda path: path.name)
def test_los_cinco_ejercicios_estan_en_ambas_versiones(carpeta) -> None:
    student = nbformat.read(carpeta / "notebook_student.ipynb", as_version=4)
    solution = nbformat.read(carpeta / "notebook_solution.ipynb", as_version=4)

    ids_student, ids_solution = _grade_ids(student), _grade_ids(solution)
    assert ids_student == ids_solution, "los identificadores nbgrader deben coincidir"
    for ejercicio in EJERCICIOS:
        assert ejercicio in ids_student, f"falta el ejercicio {ejercicio}"
        assert f"{ejercicio}_test" in ids_student, f"falta la comprobación de {ejercicio}"

    # La versión de estudiante deja el hueco; la de solución lo resuelve.
    for cell in student.cells:
        if cell.metadata.get("nbgrader", {}).get("grade_id") in EJERCICIOS:
            assert "# YOUR CODE HERE" in cell.source
    for cell in solution.cells:
        if cell.metadata.get("nbgrader", {}).get("grade_id") in EJERCICIOS:
            assert "# SOLUCIÓN DE REFERENCIA" in cell.source


@pytest.mark.parametrize("carpeta", _laboratorios(), ids=lambda path: path.name)
def test_la_solucion_de_referencia_ejecuta_y_pasa(carpeta) -> None:
    solution = nbformat.read(carpeta / "notebook_solution.ipynb", as_version=4)
    espacio: dict = {}
    ejecutadas = 0
    for cell in solution.cells:
        metadata = cell.metadata.get("nbgrader")
        if not metadata or cell.cell_type != "code":
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(cell.source, f"{carpeta.name}:{metadata['grade_id']}", "exec"), espacio)
        ejecutadas += 1
    assert ejecutadas >= len(EJERCICIOS) * 2


@pytest.mark.parametrize("carpeta", _laboratorios(), ids=lambda path: path.name)
def test_el_recorrido_no_es_la_solucion(carpeta) -> None:
    """El cuaderno de recorrido debe ser un documento distinto, sin ejercicios."""
    walkthrough = nbformat.read(carpeta / "notebook.ipynb", as_version=4)
    solution = nbformat.read(carpeta / "notebook_solution.ipynb", as_version=4)
    assert [cell.source for cell in walkthrough.cells] != [cell.source for cell in solution.cells]
    assert not any(
        cell.metadata.get("nbgrader", {}).get("grade_id") in EJERCICIOS
        for cell in walkthrough.cells
    )
