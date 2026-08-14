#!/usr/bin/env python3
"""Los cinco ejercicios evaluables que lleva cada laboratorio.

Se definen aquí una sola vez y los usan los dos generadores de cuadernos —el de
los laboratorios centrales y el de las especializaciones—, de modo que la
práctica sea la misma en las 31 rutas y sus identificadores nbgrader no puedan
divergir.

Los cinco ejercicios cubren el contrato experimental del repositorio, que es lo
que distingue a estos laboratorios de un tutorial: auditar la partición, decidir
con `validation`, compararse con una línea base, sellar antes de abrir `test` y
dejar el plan experimental por escrito. Cada uno se parametriza con los datos del
laboratorio —su métrica de selección, su línea base, su experimento propio— así
que el enunciado y la solución cambian de una ruta a otra.

Todos se resuelven con Python estándar y **no requieren descargar el dataset ni
entrenar**: se pueden ejecutar y corregir sin conexión y en segundos. Esa es una
decisión deliberada, para que la práctica del protocolo no dependa de tener una
GPU ni de que un proveedor esté disponible.
"""
from __future__ import annotations

from typing import Any

# Métricas donde un valor MENOR es mejor. El resto se interpreta al revés.
LOWER_IS_BETTER = (
    "loss", "mse", "rmse", "mae", "error", "wasserstein", "nt_xent",
    "ece", "brier", "perplexity", "noise",
)


def higher_is_better(metric: str) -> bool:
    """¿Un valor mayor de esta métrica significa un modelo mejor?"""
    name = str(metric).lower()
    return not any(token in name for token in LOWER_IS_BETTER)


def _quote(text: str) -> str:
    return repr(str(text))


def exercises(*, metric: str, baseline: str, experiment: str, dataset: str) -> list[dict[str, Any]]:
    """Devuelve los cinco ejercicios ya parametrizados para un laboratorio."""
    better = higher_is_better(metric)
    comparador = "max" if better else "min"
    sentido = "mayor es mejor" if better else "menor es mejor"
    signo = ">" if better else "<"

    return [
        {
            "id": "auditoria_particiones",
            "points": 5,
            "title": "Ejercicio 1 — Auditar la partición",
            "statement": (
                "Antes de entrenar hay que demostrar que ningún ejemplo aparece en dos particiones. "
                "Una sola fila compartida entre `train` y `test` infla la métrica final sin dar "
                "ningún síntoma.\n\n"
                "Escribe `sin_solapamiento(train_ids, validation_ids, test_ids)`, que devuelve `True` "
                "solo si los tres conjuntos de identificadores son disjuntos dos a dos."
            ),
            "solution": (
                "def sin_solapamiento(train_ids, validation_ids, test_ids):\n"
                "    train, validation, test = set(train_ids), set(validation_ids), set(test_ids)\n"
                "    return not (train & validation or train & test or validation & test)"
            ),
            "student": (
                "def sin_solapamiento(train_ids, validation_ids, test_ids):\n"
                "    # Pista: convierte cada lista en un conjunto y comprueba las tres intersecciones.\n"
                "    raise NotImplementedError"
            ),
            "test": (
                "assert sin_solapamiento([1, 2, 3], [4, 5], [6]) is True\n"
                "assert sin_solapamiento([1, 2, 3], [3, 4], [6]) is False   # solapa train y validation\n"
                "assert sin_solapamiento([1, 2], [4], [2]) is False         # solapa train y test\n"
                "assert sin_solapamiento([1], [2, 3], [3]) is False         # solapa validation y test\n"
                "assert sin_solapamiento([], [], []) is True\n"
                "print('Auditoría correcta.')"
            ),
        },
        {
            "id": "regla_de_seleccion",
            "points": 5,
            "title": "Ejercicio 2 — Elegir el checkpoint con `validation`",
            "statement": (
                f"Este laboratorio selecciona con **`{metric}`**, donde **{sentido}**. El modelo que "
                "se conserva es el de la época con mejor valor *en validación*, nunca en test.\n\n"
                "Escribe `mejor_epoca(historial)`, que recibe una lista de diccionarios con las "
                f"claves `epoch` y `{metric}` y devuelve el número de la mejor época."
            ),
            "solution": (
                f"SELECTION_METRIC = {_quote(metric)}\n"
                f"HIGHER_IS_BETTER = {better}\n"
                "\n"
                "def mejor_epoca(historial):\n"
                "    elegir = max if HIGHER_IS_BETTER else min\n"
                "    mejor = elegir(historial, key=lambda fila: fila[SELECTION_METRIC])\n"
                "    return mejor['epoch']"
            ),
            "student": (
                f"SELECTION_METRIC = {_quote(metric)}\n"
                "HIGHER_IS_BETTER = None   # ¿mayor es mejor para esta métrica?\n"
                "\n"
                "def mejor_epoca(historial):\n"
                f"    # Pista: usa {comparador}() con key=... sobre la métrica de selección.\n"
                "    raise NotImplementedError"
            ),
            "test": (
                "historial = [\n"
                f"    {{'epoch': 1, {_quote(metric)}: 0.40}},\n"
                f"    {{'epoch': 2, {_quote(metric)}: 0.65}},\n"
                f"    {{'epoch': 3, {_quote(metric)}: 0.55}},\n"
                "]\n"
                f"assert HIGHER_IS_BETTER is {better}\n"
                f"assert mejor_epoca(historial) == {2 if better else 1}\n"
                "print('Regla de selección correcta.')"
            ),
        },
        {
            "id": "comparacion_linea_base",
            "points": 5,
            "title": "Ejercicio 3 — Comparar contra la línea base",
            "statement": (
                f"La línea base de este laboratorio es **{baseline}**. Superarla por poco no basta: "
                "la diferencia tiene que ser mayor que la dispersión del propio modelo entre "
                "semillas, o no se puede distinguir de la suerte.\n\n"
                "Escribe `supera_linea_base(puntajes, base)`, que recibe la lista de puntajes del "
                "modelo (uno por semilla) y el puntaje de la línea base, y devuelve un diccionario "
                "con `media`, `desviacion` y `concluyente`."
            ),
            "solution": (
                "from statistics import mean, pstdev\n"
                "\n"
                "def supera_linea_base(puntajes, base):\n"
                "    media = mean(puntajes)\n"
                "    desviacion = pstdev(puntajes) if len(puntajes) > 1 else 0.0\n"
                f"    ventaja = media - base if {better} else base - media\n"
                "    return {\n"
                "        'media': media,\n"
                "        'desviacion': desviacion,\n"
                "        'concluyente': ventaja > desviacion,\n"
                "    }"
            ),
            "student": (
                "from statistics import mean, pstdev\n"
                "\n"
                "def supera_linea_base(puntajes, base):\n"
                "    # Pista: la ventaja es la diferencia en el sentido correcto de la métrica,\n"
                "    # y solo es concluyente si supera la dispersión entre semillas.\n"
                "    raise NotImplementedError"
            ),
            "test": (
                # Caso claro: gran ventaja y semillas muy juntas.
                # Caso dudoso: la ventaja (0,05) cabe dentro de la dispersión (≈0,082).
                ("claro = supera_linea_base([0.80, 0.81, 0.82], 0.60)\n"
                 "dudoso = supera_linea_base([0.55, 0.65, 0.75], 0.60)\n"
                 "assert round(claro['media'], 4) == 0.81\n"
                 if better else
                 "claro = supera_linea_base([0.20, 0.21, 0.22], 0.60)\n"
                 "dudoso = supera_linea_base([0.45, 0.55, 0.65], 0.60)\n"
                 "assert round(claro['media'], 4) == 0.21\n") +
                "assert claro['concluyente'] is True\n"
                "assert dudoso['concluyente'] is False   # la mejora cabe dentro del ruido\n"
                "print('Comparación correcta:', claro)"
            ),
        },
        {
            "id": "sellado_del_test",
            "points": 5,
            "title": "Ejercicio 4 — No abrir `test` sin sellar",
            "statement": (
                "El repositorio escribe `experiment.lock.json` con las semillas, la configuración y "
                "el checkpoint elegido **antes** de evaluar `test`. Sin ese archivo, cualquier "
                "número de test es inválido.\n\n"
                "Escribe `puede_abrir_test(run_dir)`, que devuelve `True` solo si el directorio de la "
                "ejecución contiene `experiment.lock.json`."
            ),
            "solution": (
                "from pathlib import Path\n"
                "\n"
                "def puede_abrir_test(run_dir):\n"
                "    return (Path(run_dir) / 'experiment.lock.json').is_file()"
            ),
            "student": (
                "from pathlib import Path\n"
                "\n"
                "def puede_abrir_test(run_dir):\n"
                "    # Pista: comprueba la existencia del archivo de sellado dentro del directorio.\n"
                "    raise NotImplementedError"
            ),
            "test": (
                "import tempfile\n"
                "from pathlib import Path\n"
                "\n"
                "with tempfile.TemporaryDirectory() as carpeta:\n"
                "    ruta = Path(carpeta)\n"
                "    assert puede_abrir_test(ruta) is False   # todavía no se selló\n"
                "    (ruta / 'experiment.lock.json').write_text('{}', encoding='utf-8')\n"
                "    assert puede_abrir_test(ruta) is True\n"
                "print('Sellado comprobado.')"
            ),
        },
        {
            "id": "plan_experimental",
            "points": 5,
            "title": "Ejercicio 5 — Dejar el plan por escrito",
            "statement": (
                f"El experimento propio de esta ruta es: **{str(experiment).rstrip('.')}**.\n\n"
                "Declara el plan antes de ejecutarlo: qué hipótesis pones a prueba, qué variable "
                "cambias, qué mantienes fijo, con qué semillas de entrenamiento y qué conclusión "
                "esperas poder escribir. Completa las cinco variables."
            ),
            "solution": (
                f"HIPOTESIS = {_quote(str(experiment).rstrip('.') + f' mejora {metric} frente a {baseline}.')}\n"
                f"VARIABLE_QUE_CAMBIA = {_quote(str(experiment).rstrip('.'))}\n"
                "VARIABLES_CONTROLADAS = ['misma partición (split_seed=42)', 'mismo presupuesto de épocas', "
                f"{_quote('mismo dataset: ' + str(dataset))}]\n"
                "SEMILLAS_DE_ENTRENAMIENTO = [41, 42, 43]\n"
                f"CONCLUSION = ('La diferencia solo es concluyente si supera la dispersión entre semillas; "
                f"si no, se reporta que no se distingue del ruido, y se compara siempre contra {baseline}.')"
            ),
            "student": (
                "HIPOTESIS = ''\n"
                "VARIABLE_QUE_CAMBIA = ''\n"
                "VARIABLES_CONTROLADAS = []\n"
                "SEMILLAS_DE_ENTRENAMIENTO = []\n"
                "CONCLUSION = ''"
            ),
            "test": (
                "assert len(HIPOTESIS) >= 30, 'La hipótesis debe poder resultar falsa; descríbela.'\n"
                "assert len(VARIABLE_QUE_CAMBIA) >= 10, 'Un experimento cambia una cosa a la vez.'\n"
                "assert len(VARIABLES_CONTROLADAS) >= 3, 'Enumera al menos tres variables controladas.'\n"
                "assert len(SEMILLAS_DE_ENTRENAMIENTO) >= 3, 'Con menos de tres semillas no hay dispersión.'\n"
                "assert len(CONCLUSION) >= 40, 'La conclusión debe hablar de magnitud e incertidumbre.'\n"
                "print('Plan experimental completo.')"
            ),
        },
    ]


def graded_metadata(grade_id: str, points: int, solution: bool) -> dict[str, Any]:
    return {
        "nbgrader": {
            "grade": False,
            "grade_id": grade_id,
            "locked": False,
            "points": points,
            "schema_version": 3,
            "solution": True,
            "task": False,
        }
    }


def test_metadata(grade_id: str, points: int) -> dict[str, Any]:
    return {
        "nbgrader": {
            "grade": True,
            "grade_id": grade_id,
            "locked": True,
            "points": points,
            "schema_version": 3,
            "solution": False,
            "task": False,
        }
    }
