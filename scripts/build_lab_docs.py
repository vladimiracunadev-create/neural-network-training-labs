#!/usr/bin/env python3
"""Construye el material de cada laboratorio y las páginas de parte en Markdown.

Cada laboratorio publica cuatro documentos. Tres se generan enteros desde los
datos del repositorio y uno es la fuente redactada a mano:

* `README.md` — la **clase completa**: qué se va a hacer, la teoría incrustada
  desde `theory.md` (idea, matemática y riesgos), los comandos explicados opción
  por opción con su equivalente en Python, el paso a paso del protocolo, cómo
  leer cada archivo que produce la ejecución, los errores frecuentes y la tabla
  que dice dónde comprobar cada afirmación.
* `experiments.md` — el **plan experimental**: qué se varía, qué se mantiene fijo
  y por qué, cómo ejecutar la serie multi-semilla y cómo decidir con los números.
* `assessment.md` — la **evaluación**: evidencias, preguntas con lo que se busca
  en cada respuesta, y la rúbrica explicada.
* `theory.md` — la explicación redactada del tema, con su bibliografía. **No se
  reescribe nunca**: es la fuente desde la que la guía incrusta su teoría.

Además genera las siete páginas de parte (`parts/*.md`) y su índice, y mantiene
la navegación —posición en el recorrido, parte, laboratorio anterior y siguiente,
y barra de documentos— en los cuatro archivos, delimitada con marcadores HTML
para que el script sea idempotente.

    python scripts/build_lab_docs.py            # aplica los cambios
    python scripts/build_lab_docs.py --check    # falla si algo quedó desfasado

`scripts/generate_site.py` elimina estos bloques al renderizar el sitio, porque
allí la navegación la aporta el paginador propio.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SITE = "https://vladimiracunadev-create.github.io/neural-network-training-labs"

# Emoji identificador por laboratorio. Debe coincidir con scripts/generate_site.py.
LAB_EMOJI = {
    "00_numpy_neuron": "🔢", "01_pytorch_perceptron": "🧩", "02_mlp_nonlinear": "🌀",
    "03_cnn_vision": "🖼️", "04_rnn_sequences": "🔁", "05_lstm_time_series": "📈",
    "06_autoencoder_anomaly": "🧬", "07_transformer_attention": "🔭",
    "08_gan_generation": "🎨", "09_gnn_graphs": "🕸️", "10_dqn_reinforcement": "🕹️",
    "11_transfer_learning": "♻️", "12_multimodal_fusion": "🔀",
    "13_hyperparameter_search": "🎛️", "14_knowledge_distillation": "⚗️",
    "15_federated_learning": "🌐", "16_backpropagation_manual": "∂",
    "17_activations_and_losses": "📐", "18_optimizers_and_schedulers": "⚙️",
    "19_regularization_dropout_batchnorm": "🛡️", "20_data_augmentation": "🔄",
    "21_explainability": "🔍", "22_uncertainty_calibration": "🎯",
    "23_model_export_and_inference": "📦", "24_capstone_real_project": "🏁",
    "25_transformer_finetuning": "🔧", "26_segmentation_unet": "🧷",
    "27_audio_speechcommands": "🎙️", "28_wgan_gp": "🖌️",
    "29_diffusion_ddpm": "🌫️", "30_self_supervised_simclr": "🪞",
}

# (archivo, emoji, etiqueta corta) de los cuatro documentos de cada laboratorio.
DOCS = [
    ("README.md", "📄", "Guía"),
    ("theory.md", "🧠", "Teoría"),
    ("experiments.md", "🔬", "Experimentos"),
    ("assessment.md", "📝", "Evaluación"),
]

# Las siete partes del recorrido. Son tramos CONTIGUOS de la secuencia 00 → 30:
# `first` y `last` son el prefijo numérico del primer y último laboratorio, de modo
# que ninguna clase queda fuera y ninguna aparece en dos partes.
PARTS = [
    {
        "num": 1, "slug": "01-fundamentos", "emoji": "🟢",
        "title": "Fundamentos: de la derivada a la primera red",
        "first": 0, "last": 2,
        "summary": (
            "Se construye una red desde cero antes de usar cualquier abstracción: primero la "
            "neurona a mano en NumPy, después el mismo cálculo delegado en autograd, y por "
            "último varias capas resolviendo un problema que una recta no separa."
        ),
        "outcome": "entiendes qué calcula, qué deriva y qué actualiza un entrenamiento.",
    },
    {
        "num": 2, "slug": "02-arquitecturas", "emoji": "🔵",
        "title": "Arquitecturas según la forma del dato",
        "first": 3, "last": 7,
        "summary": (
            "Cada estructura —imagen, secuencia, serie temporal, señal sin etiqueta, texto— pide "
            "su propio sesgo inductivo. Aquí se recorren las cinco familias que cubren la mayoría "
            "de los problemas reales, y se comparan contra una línea base honesta."
        ),
        "outcome": "eliges arquitectura por la forma del problema, no por la moda.",
    },
    {
        "num": 3, "slug": "03-familias-especializadas", "emoji": "🟣",
        "title": "Familias especializadas: generar, decidir, relacionar",
        "first": 8, "last": 12,
        "summary": (
            "Tres regímenes donde una métrica de acierto ya no cuenta toda la historia —generación, "
            "decisión secuencial y datos relacionales— más las dos formas de reutilizar y combinar "
            "información que ya existe."
        ),
        "outcome": "evalúas sistemas que no tienen una única etiqueta correcta.",
    },
    {
        "num": 4, "slug": "04-entrenamiento-eficiente", "emoji": "🟠",
        "title": "Entrenar mejor, más barato y sin centralizar datos",
        "first": 13, "last": 15,
        "summary": (
            "El modelo ya funciona: ahora hay que mejorarlo sin hacer trampas, encogerlo para que "
            "quepa donde debe correr, y entrenarlo cuando los datos no pueden salir de donde están."
        ),
        "outcome": "mejoras un modelo sin tocar `test` y sabes qué cuesta cada mejora.",
    },
    {
        "num": 5, "slug": "05-mecanica-fina", "emoji": "🔴",
        "title": "La mecánica fina, ahora en profundidad",
        "first": 16, "last": 20,
        "summary": (
            "Segunda pasada por el motor, ya con la experiencia de haber entrenado modelos reales: "
            "lo que en la ruta 00 era una fórmula, aquí es una decisión de diseño que se mide, se "
            "compara entre semillas y se justifica."
        ),
        "outcome": "explicas por qué un entrenamiento converge, se estanca o sobreajusta.",
    },
    {
        "num": 6, "slug": "06-confianza-y-despliegue", "emoji": "⚫",
        "title": "Confiar en el modelo y sacarlo del cuaderno",
        "first": 21, "last": 24,
        "summary": (
            "Un acierto sin explicación ni confianza calibrada no es evidencia, y un modelo que solo "
            "corre en un cuaderno no es un sistema. Esta parte cierra el ciclo hasta el artefacto "
            "desplegable y el proyecto integrador."
        ),
        "outcome": "respondes «¿por qué predijo esto?», «¿cuánto te fías?» y «¿cuánto tarda?».",
    },
    {
        "num": 7, "slug": "07-especializaciones-avanzadas", "emoji": "🔬",
        "title": "Especializaciones avanzadas",
        "first": 25, "last": 30,
        "summary": (
            "Mismo contrato de semillas, selección por validación y sellado del test, con "
            "arquitecturas de frontera y pesos preentrenados descargados de su proveedor. "
            "Se pueden tomar en cualquier orden una vez completadas las rutas 00–24."
        ),
        "outcome": "trabajas con arquitecturas actuales sin renunciar al protocolo.",
    },
]

TOP_RE = re.compile(r"\n?<!-- nav-top -->.*?<!-- /nav-top -->\n?", re.DOTALL)
BOTTOM_RE = re.compile(r"\n?<!-- nav-bottom -->.*?<!-- /nav-bottom -->\n?", re.DOTALL)
NAV_ANY_RE = re.compile(r"\n?<!-- nav-(top|bottom) -->.*?<!-- /nav-\1 -->\n?", re.DOTALL)

BADGE_COLORS = {
    "fundamentos": "3fb950", "básico": "3fb950",
    "intermedio": "1f6feb",
    "intermedio-avanzado": "8957e5", "avanzado": "8957e5",
    "experto": "d29922",
}

# Los catálogos avanzados declaran el nivel en inglés; el material es en español.
LEVEL_ES = {
    "beginner": "fundamentos", "basic": "fundamentos", "foundations": "fundamentos",
    "intermediate": "intermedio", "advanced": "avanzado", "expert": "experto",
}


# ──────────────────────────────────────────────────────────────────────────────
# Lectura de metadatos
# ──────────────────────────────────────────────────────────────────────────────

def _yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _first_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        match = re.match(r"^#\s+(.*)", line.strip())
        if match:
            return match.group(1).strip()
    return fallback


def _catalog() -> dict[str, dict]:
    """Une el catálogo central y el de especializaciones por identificador."""
    entries: dict[str, dict] = {}
    core = _yaml(ROOT / "configs/labs.yaml").get("labs") or []
    for item in core:
        entries[item["id"]] = item
    advanced = _yaml(ROOT / "configs/advanced_tracks.yaml").get("tracks") or []
    for item in advanced:
        entries[item["id"]] = item
    return entries


def part_for(num: int) -> dict:
    """Parte a la que pertenece un laboratorio, por su prefijo numérico."""
    for part in PARTS:
        if part["first"] <= num <= part["last"]:
            return part
    raise SystemExit(f"ERROR: la ruta {num:02d} no pertenece a ninguna parte; revisa PARTS.")


def collect_labs() -> list[dict]:
    catalog = _catalog()
    labs: list[dict] = []
    for base, category in (("labs", "Central"), ("advanced_labs", "Avanzada")):
        base_dir = ROOT / base
        if not base_dir.exists():
            continue
        for lab_dir in sorted(path for path in base_dir.iterdir() if path.is_dir()):
            readme = lab_dir / "README.md"
            if not readme.exists():
                continue
            slug = lab_dir.name
            lesson = _yaml(lab_dir / "lesson.yaml")
            labs.append({
                "slug": slug,
                "base": base,
                "num": slug.split("_", 1)[0],
                "part": part_for(int(slug.split("_", 1)[0])),
                "category": category,
                "dir": lab_dir,
                "emoji": LAB_EMOJI.get(slug, "🧠"),
                "title": _first_title(readme.read_text(encoding="utf-8"), slug),
                "lesson": lesson,
                "dataset": _yaml(lab_dir / "data/dataset.yaml"),
                "baseline_cfg": _yaml(lab_dir / "configs/baseline.yaml"),
                "improved_cfg": _yaml(lab_dir / "configs/improved.yaml"),
                "catalog": catalog.get(slug, {}),
            })
    return labs


# ──────────────────────────────────────────────────────────────────────────────
# Navegación
# ──────────────────────────────────────────────────────────────────────────────

def doc_link(lab: dict, doc: str) -> str:
    """Ruta relativa hacia un documento de otro laboratorio (o del mismo)."""
    return f'../../{lab["base"]}/{lab["slug"]}/{doc}'


def label(lab: dict) -> str:
    return f'{lab["emoji"]} {lab["title"]}'


def top_block(lab: dict, prev: dict | None, nxt: dict | None,
              index: int, total: int, current_doc: str) -> str:
    jumps = []
    if prev:
        jumps.append(f'[⬅️ {label(prev)}]({doc_link(prev, current_doc if (prev["dir"] / current_doc).exists() else "README.md")})')
    else:
        jumps.append("⬅️ *inicio del recorrido*")
    jumps.append("[🏠 Índice de rutas](../../parts/README.md)")
    if nxt:
        jumps.append(f'[{label(nxt)} ➡️]({doc_link(nxt, current_doc if (nxt["dir"] / current_doc).exists() else "README.md")})')
    else:
        jumps.append("*fin del recorrido* ➡️")

    tabs = []
    for doc, emoji, name in DOCS:
        if not (lab["dir"] / doc).exists():
            continue
        tabs.append(f"**{emoji} {name}**" if doc == current_doc else f"[{emoji} {name}]({doc})")

    part = lab["part"]
    part_link = f'[Parte {part["num"]} — {part["title"]}](../../parts/{part["slug"]}.md)'

    return (
        "<!-- nav-top -->\n"
        f"> 🧭 **Ruta {index + 1} / {total}** · {part['emoji']} {part_link}\n"
        ">\n"
        f"> {' · '.join(jumps)}\n"
        ">\n"
        f"> {' · '.join(tabs)}\n"
        "<!-- /nav-top -->"
    )


def bottom_block(lab: dict, prev: dict | None, nxt: dict | None, current_doc: str) -> str:
    prev_cell = f'[{label(prev)}]({doc_link(prev, "README.md")})' if prev else "*— inicio del recorrido*"
    next_cell = f'[{label(nxt)}]({doc_link(nxt, "README.md")})' if nxt else "*— fin del recorrido*"

    docs = []
    for doc, emoji, name in DOCS:
        if not (lab["dir"] / doc).exists():
            continue
        docs.append(f"**{emoji} {name}**" if doc == current_doc else f"[{emoji} {name}]({doc})")
    for notebook, emoji, name in (
        ("notebook.ipynb", "📓", "Recorrido"),
        ("notebook_student.ipynb", "✏️", "Estudiante"),
        ("notebook_solution.ipynb", "✅", "Solución"),
    ):
        if (lab["dir"] / notebook).exists():
            docs.append(f"[{emoji} {name}]({notebook})")

    part = lab["part"]
    # `index.html` lo genera scripts/generate_lab_html.py a partir de este Markdown;
    # el enlace es incondicional para que el orden de generación sea determinista.
    salidas = [
        f'{part["emoji"]} [Parte {part["num"]} — {part["title"]}](../../parts/{part["slug"]}.md)',
        "[🏠 Portada del repositorio](../../README.md)",
        f'[🌐 Sitio de estudio]({SITE}/labs/{lab["slug"]}/index.html)',
        "[🖥️ Página HTML local](index.html)",
    ]

    return (
        "<!-- nav-bottom -->\n"
        "## 🧭 Navegación del recorrido\n\n"
        "| ⬅️ Laboratorio anterior | 🏠 Índice | Laboratorio siguiente ➡️ |\n"
        "|---|:---:|---|\n"
        f"| {prev_cell} | [Las 31 rutas](../../parts/README.md) | {next_cell} |\n\n"
        f"**En este laboratorio:** {' · '.join(docs)}\n\n"
        f"{' · '.join(salidas)}\n"
        "<!-- /nav-bottom -->"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Comparación de configuraciones (la usa el plan de experimentos)
# ──────────────────────────────────────────────────────────────────────────────

CONFIG_KEYS = [
    ("epochs", "Épocas", "Cuántas pasadas completas sobre `train`."),
    ("batch_size", "Tamaño de lote", "Cuántos ejemplos se promedian antes de cada actualización de pesos."),
    ("learning_rate", "Tasa de aprendizaje", "Cuánto se mueve cada peso en la dirección del gradiente."),
    ("patience", "Paciencia", "Épocas sin mejorar en `validation` antes de detener el entrenamiento."),
    ("amp", "Precisión mixta", "Usar float16 donde se puede: acelera en GPU, no cambia el protocolo."),
    ("num_workers", "Procesos de carga", "Procesos que preparan los lotes en paralelo."),
    ("gradient_clip_norm", "Recorte de gradiente", "Techo a la norma del gradiente, para evitar pasos desbocados."),
]


def _show_value(value: object) -> str:
    if isinstance(value, bool):
        return "sí" if value else "no"
    return f"`{value}`"


def config_comparison(lab: dict) -> list[str]:
    """Tabla `baseline` vs `improved`, solo con los parámetros que difieren."""
    baseline, improved = lab["baseline_cfg"], lab["improved_cfg"]
    if not baseline or not improved:
        return []
    rows = [
        f"| {name} | {_show_value(baseline[key])} | {_show_value(improved[key])} | {meaning} |"
        for key, name, meaning in CONFIG_KEYS
        if key in baseline and key in improved and baseline[key] != improved[key]
    ]
    if not rows:
        return []
    return [
        "| Parámetro | `baseline.yaml` | `improved.yaml` | Qué controla |",
        "|---|---|---|---|",
        *rows,
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Aplicación de los bloques
# ──────────────────────────────────────────────────────────────────────────────

def render_doc(lab: dict, doc: str, index: int, total: int,
               prev: dict | None, nxt: dict | None) -> str:
    path = lab["dir"] / doc

    # Tres de los cuatro documentos se generan enteros desde los datos del
    # repositorio —la guía, el plan de experimentos y la evaluación—, porque su
    # contenido anterior era una plantilla derivada del catálogo y mantenerla a
    # mano solo abría la puerta a que dijera algo que el código ya no hace.
    # `theory.md` es la excepción: lleva la explicación redactada de cada tema y
    # es la fuente que la guía incrusta, así que nunca se reescribe.
    generated = {
        "README.md": (lab["title"], lambda: guia_doc(lab, index, total, prev, nxt)),
        "experiments.md": (f'Plan de experimentos — {lab["title"]}', lambda: experiments_doc(lab)),
        "assessment.md": (f'Evaluación — {lab["title"]}', lambda: assessment_doc(lab)),
    }
    if doc in generated:
        heading, builder = generated[doc]
        return "\n".join([
            f"# {heading}",
            "",
            top_block(lab, prev, nxt, index, total, doc),
            "",
            builder(),
            "",
            bottom_block(lab, prev, nxt, doc),
            "",
        ])

    text = path.read_text(encoding="utf-8")
    # Se retiran los bloques generados para volver a insertarlos actualizados.
    text = TOP_RE.sub("\n", text)
    text = BOTTOM_RE.sub("\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"

    header = top_block(lab, prev, nxt, index, total, doc)
    out: list[str] = []
    inserted = False
    for line in text.splitlines():
        out.append(line)
        if not inserted and re.match(r"^#\s+", line):
            out.append("")
            out.append(header)
            inserted = True
    if not inserted:
        out.insert(0, header)

    body = "\n".join(out).rstrip() + "\n\n"
    body += bottom_block(lab, prev, nxt, doc) + "\n"
    return body


# ──────────────────────────────────────────────────────────────────────────────
# Guía paso a paso (bloque <!-- guia --> del README de cada laboratorio)
#
# Todo lo que se afirma aquí sale de un archivo del repositorio, nunca de una
# suposición: el objetivo, la línea base y las métricas vienen del catálogo
# (`configs/labs.yaml` o `configs/advanced_tracks.yaml`); la procedencia y la
# licencia del dataset, de `data/dataset.yaml`; los hiperparámetros, de
# `configs/*.yaml`; y el orden de los pasos y los archivos que produce cada
# ejecución, del código que los escribe (`src/neural_labs/experiments.py` y
# `src/neural_labs/advanced/training.py`). La última sección de la guía deja esa
# correspondencia por escrito para que cualquiera pueda comprobarla.
# ──────────────────────────────────────────────────────────────────────────────

# Archivos adicionales que escribe cada arquitectura, además de los comunes.
# Verificado leyendo las funciones `_run_*` de src/neural_labs/experiments.py.
EXTRA_ARTIFACTS = {
    # `_run_numpy_logistic` guarda además el estado final; `_run_numpy_mlp` solo el mejor.
    "numpy_logistic": [("last_model.npz", "El estado de la última época, para contrastarlo con el mejor.")],
    "dcgan": [("generated_samples.png", "Rejilla de muestras generadas: la evidencia visual de si hay diversidad o colapso.")],
    "gcn": [("graph_model_comparison.json", "Puntaje de GCN, GraphSAGE y GAT, y cuál se seleccionó.")],
    "transfer_resnet18": [("transfer_comparison.json", "Extracción de características frente a fine-tuning frente a entrenar desde cero.")],
    "mlp_optuna": [("hyperparameter_trials.json", "Cada combinación probada, su puntaje y la ganadora.")],
    "augmentation_comparison": [("augmentation_comparison.json", "El mismo modelo con y sin aumento de datos.")],
    "activation_comparison": [("variant_comparison.json", "Una fila por variante comparada, con su métrica de validación.")],
    "optimizer_comparison": [("variant_comparison.json", "Una fila por variante comparada, con su métrica de validación.")],
    "regularization_comparison": [("variant_comparison.json", "Una fila por variante comparada, con su métrica de validación.")],
}

# Pasos finales que el código aplica solo a estos laboratorios (dispatch explícito
# por `lab_id` en run_lab).
LAB_SPECIFIC_STEPS = {
    "21_explainability": ("feature_attributions.csv",
                          "Integrated Gradients e importancia por permutación, ordenadas por atribución media."),
    "22_uncertainty_calibration": ("calibration.json",
                                   "Temperatura ajustada y error de calibración esperado (ECE)."),
    "23_model_export_and_inference": ("model.onnx",
                                      "El modelo exportado; `metrics.json` añade latencia, throughput y tamaño."),
}

# Tareas para las que el código NO produce ciertos artefactos. Sirve para explicar
# una ausencia en vez de dejar que parezca un error.
NO_PREDICTIONS_TASKS = {"node_classification", "reinforcement_learning", "generation", "anomaly_detection"}


def _theory_sections(lab: dict) -> dict[str, str]:
    """Devuelve las secciones de `theory.md` indexadas por su encabezado.

    La guía las incrusta en lugar de enlazarlas: un laboratorio se entiende mejor
    leído de corrido que saltando entre archivos. `theory.md` sigue siendo la
    fuente —se edita ahí y esta guía se regenera—, de modo que no hay dos textos
    que puedan contradecirse.
    """
    path = lab["dir"] / "theory.md"
    if not path.exists():
        return {}
    text = NAV_ANY_RE.sub("\n", path.read_text(encoding="utf-8"))
    sections: dict[str, str] = {}
    current: str | None = None
    buffer: list[str] = []
    for line in text.splitlines():
        heading = re.match(r"^##\s+(.*)", line)
        if heading:
            if current:
                sections[current] = "\n".join(buffer).strip()
            current = heading.group(1).strip()
            buffer = []
        elif current:
            buffer.append(line)
    if current:
        sections[current] = "\n".join(buffer).strip()
    return sections


def _theory_question(lab: dict) -> str | None:
    """Pregunta crítica que ya vive en el `theory.md` del laboratorio."""
    section = _theory_sections(lab).get("Pregunta crítica", "")
    match = re.match(r">\s*(.+)", section.strip())
    return match.group(1).strip() if match else None


def _quick_note(lab: dict) -> str | None:
    """Qué recorta exactamente `--quick` en este laboratorio, según su configuración."""
    quick = lab["baseline_cfg"].get("quick")
    if not isinstance(quick, dict):
        return None
    pieces = []
    if quick.get("max_train_samples"):
        pieces.append(f"{quick['max_train_samples']} ejemplos de entrenamiento")
    if quick.get("max_validation_samples"):
        pieces.append(f"{quick['max_validation_samples']} de validación")
    if quick.get("max_test_samples"):
        pieces.append(f"{quick['max_test_samples']} de test")
    if quick.get("epochs"):
        pieces.append(f"{quick['epochs']} épocas")
    return " · ".join(pieces) if pieces else None


def _notebook_stats(lab: dict) -> dict | None:
    """Mide los tres cuadernos del laboratorio leyéndolos, no suponiéndolos.

    Devuelve el número de celdas, cuántas son de código, cuántas cambian entre la
    versión de estudiante y la de solución —esos son los ejercicios— y si el
    cuaderno de recorrido coincide con el de solución.
    """
    import json

    def sources(name: str) -> list[str] | None:
        path = lab["dir"] / name
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return None
        return ["".join(cell.get("source") or []) for cell in payload.get("cells", [])]

    walkthrough = sources("notebook.ipynb")
    student = sources("notebook_student.ipynb")
    solution = sources("notebook_solution.ipynb")
    if not (walkthrough and student and solution):
        return None

    import json as _json
    cells = _json.loads((lab["dir"] / "notebook.ipynb").read_text(encoding="utf-8"))["cells"]
    # La primera celda solo cambia la etiqueta de versión; no cuenta como ejercicio.
    differing = [
        index for index, (a, b) in enumerate(zip(student, solution))
        if a != b and "**Versión:**" not in a
    ]
    return {
        "cells": len(cells),
        "code_cells": sum(cell.get("cell_type") == "code" for cell in cells),
        "practice_cells": len(student),
        "exercises": len(differing),
        "walkthrough_equals_solution": walkthrough == solution,
    }


def _notebooks_section(lab: dict) -> list[str]:
    """Explica los tres cuadernos: qué trae cada uno y cómo abrirlos."""
    stats = _notebook_stats(lab)
    if not stats:
        return []
    slug, base = lab["slug"], lab["base"]

    lines = [
        "## 📓 Los tres cuadernos",
        "",
        "El laboratorio se puede recorrer en Jupyter, y trae tres cuadernos con papeles distintos. "
        "Los tres siguen el mismo camino —descargar el dataset real, auditar la partición, "
        "entrenar, sellar el experimento y evaluar `test` una vez—; lo que cambia es qué te toca "
        "escribir a ti:",
        "",
        "| Cuaderno | Qué trae | Cuándo usarlo |",
        "|---|---|---|",
        f"| [📓 `notebook.ipynb`](notebook.ipynb) | El **recorrido de referencia**: {stats['cells']} celdas "
        f"({stats['code_cells']} de código) con **todo el código escrito y ejecutable**, intercalado con las "
        "explicaciones. No trae ejercicios. | Para leer y ejecutar de principio a fin. |",
        f"| [✏️ `notebook_student.ipynb`](notebook_student.ipynb) | El mismo recorrido más "
        f"**{stats['exercises']} ejercicios evaluables** ({stats['practice_cells']} celdas en total). Las celdas de "
        "ejercicio están marcadas con `# YOUR CODE HERE` y debajo de cada una hay una comprobación. | Para practicar. |",
        "| [✅ `notebook_solution.ipynb`](notebook_solution.ipynb) | Los mismos ejercicios **resueltos**, marcados con "
        "`# SOLUCIÓN DE REFERENCIA`. Cada solución se ejecuta en la integración continua, así que se sabe que pasa. "
        "| Para contrastar después de intentarlo. |",
        "",
        "### Qué se practica en los ejercicios",
        "",
        "Cinco de ellos no son de arquitectura sino del **contrato experimental**, que es lo que distingue a "
        "estos laboratorios de un tutorial: auditar la partición, decidir con `validation`, compararse con la línea "
        "base, sellar antes de abrir `test` y dejar el plan por escrito. Se resuelven con Python estándar —**sin "
        "descargar el dataset ni entrenar**—, así que se corrigen en segundos y sin GPU, y cada uno está "
        "parametrizado con los valores de este laboratorio: su métrica de selección, su línea base y su experimento "
        "propio.",
        "",
    ]

    if stats["walkthrough_equals_solution"]:
        lines += [
            "> **Aviso honesto sobre el estado actual.** Hoy `notebook.ipynb` y "
            "`notebook_solution.ipynb` tienen **el mismo contenido**. Está anotado en el "
            "[roadmap](../../ROADMAP.md) y se dice aquí para que nadie descubra el límite después "
            "de abrir el archivo.",
            "",
        ]

    lines += [
        "### Cómo abrirlos",
        "",
        "Los cuadernos necesitan el extra `notebooks`, que instala Jupyter junto con el paquete:",
        "",
        "```bash",
        'pip install -e ".[dev,notebooks]"',
        f"jupyter lab {base}/{slug}/notebook.ipynb",
        "```",
        "",
        "También se abren desde VS Code —con la extensión de Jupyter— haciendo doble clic en el "
        "archivo, o desde la interfaz clásica con `jupyter notebook`. El primer arranque descarga "
        "el dataset real desde su proveedor, así que la primera ejecución tarda más y **requiere "
        "conexión**.",
        "",
        "Si prefieres ejecutar sin abrir un cuaderno, `train.py` hace exactamente lo mismo desde la "
        "terminal, y la sección de comandos de arriba explica cada opción.",
        "",
    ]
    return lines


def _step(number: int, title: str, what: str, why: str,
          command: str | None = None, check: str | None = None) -> list[str]:
    block = [f"### Paso {number} — {title}", "", f"**Qué ocurre.** {what}", "", f"**Por qué.** {why}", ""]
    if command:
        block += ["```bash", command, "```", ""]
    if check:
        block += [f"**Cómo sabes que salió bien.** {check}", ""]
    return block


def _core_steps(lab: dict) -> list[str]:
    slug, catalog = lab["slug"], lab["catalog"]
    metric = lab["baseline_cfg"].get("selection_metric") or catalog.get("selection_metric") or "la métrica declarada"
    reference = catalog.get("baseline") or "la línea base declarada en el catálogo"
    dataset = lab["dataset"].get("name") or catalog.get("dataset") or "el dataset"
    task = lab["dataset"].get("task") or catalog.get("task") or ""
    architecture = str(lab["dataset"].get("architecture") or catalog.get("architecture") or "")
    # Las dos rutas que se implementan en NumPy no guardan un checkpoint de PyTorch.
    checkpoint = "best_model.npz" if architecture.startswith("numpy_") else "best_model.pt"
    # El código solo escribe predicciones, intervalos y subgrupos cuando hay una
    # predicción por ejemplo comparable contra su etiqueta.
    has_predictions = task not in NO_PREDICTIONS_TASKS

    lines: list[str] = []
    lines += _step(
        1, "Traer el dataset real y partirlo",
        f"Descarga `{dataset}` desde su proveedor y construye las tres particiones "
        f"—`train`, `validation` y `test`— con la semilla de partición que le pases.",
        "La partición se fija **antes** de entrenar y su semilla (`--split-seed`) es independiente de la del "
        "entrenamiento (`--training-seed`). Si ambas fueran la misma, no podrías distinguir si un cambio en el "
        "resultado viene de haber repartido los datos de otro modo o de haber inicializado los pesos de otro modo.",
        f"neural-labs dataset --lab {slug} --quick --split-seed 42",
        "El comando termina sin error y deja el dataset en caché. Si no hay red, **falla**: no se rellena con "
        "datos inventados (`data/dataset.yaml` declara `fallback_to_generated_data: false`).",
    )
    lines += _step(
        2, "Comprobar que las particiones no se tocan",
        "Recorre los identificadores de las tres particiones y verifica que ninguno aparece en dos de ellas.",
        "Una sola fila compartida entre `train` y `test` basta para que la métrica final quede inflada. "
        "Es el error más caro del oficio porque no produce ningún síntoma visible: el modelo simplemente "
        "«parece» mejor de lo que es.",
        f"neural-labs audit --lab {slug} --quick --split-seed 42",
        "La auditoría reporta cero solapamientos. Si reporta alguno, no sigas: el resto del laboratorio no "
        "significaría nada.",
    )
    lines += _step(
        3, "Mirar los datos antes de modelarlos",
        "Genera el informe de calidad (valores faltantes, balance de clases, rangos) y el de deriva entre "
        "particiones.",
        "Un desbalance fuerte o una diferencia de distribución entre `train` y `test` cambia qué métrica tiene "
        "sentido leer. Descubrirlo después de entrenar obliga a repetirlo todo.",
        f"neural-labs quality --lab {slug} --quick --split-seed 42",
        "Obtienes `data_quality.json` y `drift_report.json`; ábrelos antes de decidir la configuración.",
    )
    lines += _step(
        4, "Estudiar la teoría del laboratorio",
        "Leer [`theory.md`](theory.md): la idea central, el desarrollo matemático, los riesgos de "
        "interpretación y la bibliografía de la que sale todo eso.",
        "Sin esto, el entrenamiento es una caja que devuelve números. La teoría es lo que te permite decidir "
        "qué mirar y reconocer cuándo un resultado es sospechoso.",
        None,
        "Puedes responder, con tus palabras, qué calcula el modelo y por qué esa arquitectura encaja con "
        f"{'esta tarea' if not task else f'la tarea `{task}`'}.",
    )
    lines += _step(
        5, "Entrenar y seleccionar con `validation`",
        "El entrenamiento recorre las épocas midiendo en `validation` después de cada una, y conserva el "
        "checkpoint con el mejor valor de `" + str(metric) + "`.",
        "El conjunto de validación existe para tomar decisiones —arquitectura, hiperparámetros, cuándo parar—. "
        "Si esas decisiones se tomaran mirando `test`, `test` dejaría de ser una estimación de lo que pasará "
        "con datos nuevos y pasaría a ser parte del entrenamiento.",
        f"python labs/{slug}/train.py --quick\n# o, con control explícito de las dos semillas:\n"
        f"neural-labs train --lab {slug} --config baseline --split-seed 42 --training-seed 43",
        f"En `runs/{slug}/<ejecución>/` aparecen `history.csv` y `{checkpoint}`; la métrica de "
        "validación mejora respecto de la primera época.",
    )
    lines += _step(
        6, "Compararte con la línea base",
        f"El repositorio entrena por su cuenta **{reference}** y guarda su resultado, primero sobre "
        "`validation` y —solo al final— sobre `test`.",
        "Una métrica sola no dice si el modelo aporta algo. Puede que un método mucho más simple llegue igual "
        "de lejos, y entonces la complejidad añadida no está justificada. Esta comparación es la que convierte "
        "un número en un argumento.",
        None,
        "Comparas `metrics.json` con `baseline_metrics.json`. Si tu modelo no supera la línea base, el "
        "resultado del laboratorio es exactamente ese, y hay que reportarlo.",
    )
    lines += _step(
        7, "El sellado: `experiment.lock.json`",
        "Antes de tocar `test`, el código escribe un archivo que fija el laboratorio, las dos semillas, la "
        "configuración, la métrica de selección, el checkpoint elegido y el hash del dataset.",
        "Es la frontera del experimento. A partir de ahí, cualquier ajuste que hagas mirando `test` queda a la "
        "vista: el sello dice qué habías decidido *antes* de ver el resultado final. Sin ese archivo, nadie "
        "—incluido tú dentro de un mes— puede distinguir una predicción de una racionalización.",
        None,
        "El archivo existe y su contenido coincide con lo que creías haber ejecutado.",
    )
    lines += _step(
        8, "Evaluar `test` una sola vez y medir la incertidumbre",
        "Con el checkpoint congelado se evalúa `test`"
        + (", se calculan intervalos de confianza por bootstrap y se desglosan las métricas por subgrupo."
           if has_predictions else
           f". En esta ruta la tarea es `{task}`, así que el resultado se resume en las métricas propias de "
           "ese régimen y no en una predicción por ejemplo."),
        "Un número puntual esconde cuánto podría moverse. "
        + ("Los intervalos dicen si la diferencia con la línea base es real o cabe dentro del ruido; el "
           "desglose por subgrupo revela si el promedio está tapando un grupo donde el modelo funciona mucho "
           "peor." if has_predictions else
           "Por eso el paso siguiente —repetir con varias semillas— no es opcional aquí: es la única forma "
           "de saber cuánta de la diferencia observada es señal."),
        None,
        ("Tienes `metrics.json`, `confidence_intervals.json` y `subgroup_metrics.json`, y puedes decir la "
         "magnitud de la mejora **y** su incertidumbre." if has_predictions else
         "Tienes `metrics.json` con el resultado final, y sabes que la comparación honesta llega con las "
         "repeticiones del paso siguiente."),
    )
    lines += _step(
        9, "Repetir con varias semillas de entrenamiento",
        "Se repite el entrenamiento manteniendo **fija** la partición y cambiando solo la semilla de "
        "entrenamiento.",
        "Dos ejecuciones idénticas salvo por la inicialización pueden diferir bastante. Si no mides esa "
        "dispersión, corres el riesgo de celebrar una mejora que era una semilla afortunada.",
        f"neural-labs benchmark --lab {slug} --quick --split-seed 42 --training-seeds 41 42 43",
        "Obtienes media y dispersión entre semillas, no un único número.",
    )
    lines += _step(
        10, "Documentar y cerrar",
        "Cada ejecución deja `model_card.md` y `report.md`; el plan de experimentos vive en "
        "[`experiments.md`](experiments.md) y la rúbrica en [`assessment.md`](assessment.md).",
        "Un resultado sin su contexto —qué datos, qué decisiones, qué límites— no es reutilizable. La model "
        "card es lo que permite que otra persona sepa cuándo *no* debería usar tu modelo.",
        None,
        "Completaste la tabla multi-semilla de `experiments.md` y respondiste las preguntas de "
        "`assessment.md`.",
    )
    return lines


def _advanced_steps(lab: dict) -> list[str]:
    slug, catalog = lab["slug"], lab["catalog"]
    metric = lab["baseline_cfg"].get("selection_metric") or "la métrica declarada"
    lora = " --lora" if slug == "25_transformer_finetuning" else ""

    lines: list[str] = []
    lines += _step(
        1, "Estudiar la teoría antes de ejecutar nada",
        "Leer [`theory.md`](theory.md), que desarrolla " + str(catalog.get("math") or "el fundamento de la ruta")
        + " y cita las obras y papers de los que procede.",
        "Estas rutas usan arquitecturas donde un error de comprensión no se manifiesta como un fallo, sino "
        "como un número plausible pero equivocado.",
        None,
        "Puedes explicar qué mide `" + str(metric) + "` y por qué es la métrica de selección aquí.",
    )
    lines += _step(
        2, "Ejecutar la versión rápida",
        "Descarga el dataset y los pesos preentrenados desde su proveedor, entrena una versión reducida y "
        "escribe la ejecución en `runs-advanced/`.",
        "Antes de gastar horas de cómputo conviene comprobar que la descarga, el entorno y la ruta completa "
        "funcionan de extremo a extremo.",
        f"neural-labs train-advanced --track {slug} --quick{lora}",
        "Termina sin error y deja `metrics.json`, `history.json` y `best_model.pt` en el directorio de la "
        "ejecución.",
    )
    lines += _step(
        3, "Entrenar en serio y seleccionar con `validation`",
        "Se entrena el modelo completo conservando el checkpoint con el mejor valor de `" + str(metric)
        + "` en validación, y se sella el experimento antes de evaluar `test`.",
        "Igual que en las rutas centrales: `validation` decide, `test` solo confirma, y el sello deja por "
        "escrito qué se había decidido antes de mirar.",
        f"neural-labs train-advanced --track {slug} --split-seed 42 --training-seed 43{lora}",
        "Existe `experiment.lock.json` y `metrics.json` incluye tanto el valor de validación como el de test.",
    )
    lines += _step(
        4, "Repetir con otra semilla de entrenamiento",
        "Se repite el entrenamiento con la misma partición y distinta semilla de entrenamiento.",
        "Estas arquitecturas —adversariales, contrastivas, de difusión— son especialmente sensibles a la "
        "inicialización: una sola ejecución no permite distinguir una mejora de una casualidad.",
        f"neural-labs train-advanced --track {slug} --split-seed 42 --training-seed 44{lora}",
        "Puedes reportar el rango entre ejecuciones, no un único número.",
    )
    lines += _step(
        5, "Documentar los límites",
        "Registrar el resultado junto con la limitación declarada de la ruta y responder "
        "[`assessment.md`](assessment.md).",
        "En generación y aprendizaje autosupervisado las métricas son aproximaciones: sin declarar qué NO "
        "demuestran, invitan a conclusiones que los números no sostienen.",
        None,
        "Tu reporte dice qué mejoró, cuánto costó y en qué condiciones no esperarías el mismo resultado.",
    )
    return lines


def _artifact_table(lab: dict) -> list[str]:
    architecture = str(lab["dataset"].get("architecture") or lab["catalog"].get("architecture") or "")
    task = str(lab["dataset"].get("task") or "")
    advanced = lab["category"] == "Avanzada"

    if advanced:
        rows = [
            ("`config.json`", "Track, semillas, dispositivo y opciones con las que se lanzó."),
            ("`dataset_manifest.json`", "Fuente, licencia y número de ejemplos por partición."),
            ("`best_model.pt`", "El checkpoint seleccionado por validación."),
            ("`experiment.lock.json`", "El sello: qué se decidió antes de abrir `test`."),
            ("`history.json`", "La métrica de validación época a época."),
            ("`metrics.json`", "Resultado de validación y de test, ya con el modelo congelado."),
        ]
    else:
        rows = [
            ("`config.yaml` · `environment.json`", "La configuración exacta y el entorno (versiones, dispositivo) de la ejecución."),
            ("`dataset_manifest.json` · `dataset_card.md`", "Procedencia, licencia, hash y tamaño de cada partición."),
            ("`data_quality.json` · `drift_report.json`", "Calidad de los datos y diferencias de distribución entre particiones."),
            ("`baseline_validation_metrics.json`", "La línea base medida en `validation`, **antes** de entrenar."),
            ("`history.csv` · `history.png`", "Pérdida y métricas época a época: aquí se ve si el entrenamiento converge o sobreajusta."),
            ("`experiment.lock.json`", "El sello del experimento, escrito antes de tocar `test`."),
            ("`metrics.json`", "El resultado final en `test`, más tiempo, dispositivo y número de parámetros."),
            ("`baseline_metrics.json`", "La línea base en `test`, calculada después de tu evaluación final."),
        ]
        if architecture.startswith("numpy_"):
            rows.append(("`best_model.npz`", "El checkpoint elegido por validación. Esta ruta se implementa "
                                             "en NumPy, así que no hay un `.pt` de PyTorch."))
        else:
            rows.append(("`best_model.pt` · `last_model.pt`",
                         "El checkpoint elegido por validación y el último, para poder compararlos."))
        if task not in NO_PREDICTIONS_TASKS:
            rows += [
                ("`confidence_intervals.json`", "Intervalos por bootstrap: cuánto podría moverse cada métrica."),
                ("`subgroup_metrics.json`", "El mismo resultado desglosado por subgrupo, para ver qué esconde el promedio."),
                ("`predictions.csv`", "Predicción por ejemplo, para analizar los errores uno a uno."),
            ]
            if task and task != "regression":
                rows.append(("`confusion_matrix.png`", "Qué clases se confunden entre sí."))
        rows += [
            ("`model_spec.json` · `inference_contract.json`", "Qué entrada espera el modelo y qué devuelve: lo que necesita quien lo despliegue."),
            ("`model_card.md` · `report.md`", "La ficha del modelo y el informe legible de la ejecución."),
        ]

    for name, description in EXTRA_ARTIFACTS.get(architecture, []):
        rows.append((f"`{name}`", f"**Propio de esta ruta.** {description}"))
    if lab["slug"] in LAB_SPECIFIC_STEPS:
        name, description = LAB_SPECIFIC_STEPS[lab["slug"]]
        rows.append((f"`{name}`", f"**Propio de esta ruta.** {description}"))

    return ["| Archivo | Qué contiene y qué mirar |", "|---|---|"] + [
        f"| {name} | {description} |" for name, description in rows
    ]


def _pitfalls(lab: dict) -> list[str]:
    task = str(lab["dataset"].get("task") or "")
    items = []

    quick = _quick_note(lab)
    if quick:
        items.append(
            f"**`--quick` no es una versión pequeña del resultado, es una prueba de que todo corre.** "
            f"En esta ruta recorta a {quick}. Sirve para comprobar la instalación y la descarga; "
            f"cualquier conclusión sobre el modelo exige la ejecución completa."
        )
    items.append(
        "**Cambiar algo después de ver `test` invalida la comparación.** Si al mirar el resultado final se te "
        "ocurre una mejora, la ruta correcta es volver a `validation`, decidir allí, y sellar de nuevo."
    )
    items.append(
        "**Las dos semillas no son intercambiables.** `--split-seed` cambia *qué datos* caen en cada partición; "
        "`--training-seed` cambia *cómo se inicializa y baraja* el entrenamiento. Para comparar modelos se fija "
        "la primera y se varía la segunda."
    )
    if task in NO_PREDICTIONS_TASKS:
        items.append(
            f"**Aquí no vas a ver `predictions.csv` ni `confusion_matrix.png`, y no es un error.** La tarea es "
            f"`{task}`, y el código solo genera esos archivos cuando hay una predicción por ejemplo comparable "
            "contra una etiqueta."
        )
    elif task == "regression":
        items.append(
            "**No hay `confusion_matrix.png`, y no es un error.** Es una tarea de regresión: no existen clases "
            "que confundir."
        )
    if lab["dataset"].get("known_limitations") or lab["catalog"].get("notes"):
        limitation = (lab["catalog"].get("notes")
                      or (lab["dataset"].get("known_limitations") or [""])[0])
        items.append(f"**Límite declarado de este dataset.** {limitation}")
    return [f"- {item}" for item in items]


def _commands_section(lab: dict) -> list[str]:
    """Explica, opción por opción, los comandos que aparecen en la guía.

    Los valores por defecto salen de los parsers de `src/neural_labs/cli.py`.
    """
    advanced = lab["category"] == "Avanzada"
    slug = lab["slug"]

    lines = [
        "## 🖥️ Los comandos, explicados",
        "",
        "Todo el laboratorio se maneja con una sola herramienta de terminal, `neural-labs`, que se "
        "instala junto con el paquete (`pip install -e \".[dev,notebooks]\"`). Cada subcomando hace "
        "**una** cosa del protocolo, y por eso se pueden ejecutar por separado: preparar datos, "
        "auditar la partición, entrenar, repetir con varias semillas.",
        "",
        "La forma general es siempre la misma:",
        "",
        "```bash",
        "neural-labs <subcomando> --lab <identificador> [opciones]" if not advanced
        else "neural-labs <subcomando> --track <identificador> [opciones]",
        "```",
        "",
    ]

    if advanced:
        rows = [
            ("`--track`", f"`{slug}`", "obligatorio",
             "Qué especialización se entrena. Solo acepta los seis identificadores existentes."),
            ("`--quick`", "desactivado", "—",
             "Reduce datos y épocas para comprobar que la ruta corre de extremo a extremo."),
            ("`--split-seed N`", "`42`", "entero",
             "Semilla que decide **qué ejemplo cae en qué partición**. Se mantiene fija al comparar."),
            ("`--training-seed N`", "`42`", "entero",
             "Semilla de la inicialización de pesos y del barajado. Es la que se varía para medir dispersión."),
            ("`--device`", "`auto`", "`auto` · `cpu` · `cuda` · `mps`",
             "Dónde entrenar. `auto` elige GPU si la hay."),
            ("`--output-dir`", "`runs-advanced`", "ruta",
             "Dónde se escribe el directorio de la ejecución."),
        ]
        if slug == "25_transformer_finetuning":
            rows.append(("`--lora` / `--no-lora`", "`--no-lora`", "—",
                         "Con LoRA se entrenan unas pocas matrices de bajo rango en vez de todos los "
                         "pesos: el objetivo del laboratorio es comparar ambas."))
    else:
        rows = [
            ("`--lab`", f"`{slug}`", "obligatorio",
             "Qué laboratorio se ejecuta. Solo acepta los identificadores del catálogo."),
            ("`--quick`", "desactivado", "—",
             "Usa una fracción real del dataset y pocas épocas. Sirve para comprobar la instalación, "
             "no para concluir nada sobre el modelo."),
            ("`--split-seed N`", "`42`", "entero",
             "Semilla que decide **qué ejemplo cae en qué partición**. Se mantiene fija al comparar modelos."),
            ("`--training-seed N`", "`42`", "entero",
             "Semilla de la inicialización de pesos y del barajado de lotes. Es la que se varía para "
             "medir cuánta diferencia es simple azar."),
            ("`--config`", "`baseline`", "`baseline` · `improved`",
             "Cuál de las dos configuraciones del laboratorio se usa."),
            ("`--device`", "`auto`", "`auto` · `cpu` · `cuda` · `mps`",
             "Dónde entrenar. `auto` elige GPU si está disponible y cae a CPU si no."),
            ("`--training-seeds A B C`", "`41 42 43`", "enteros",
             "Solo en `benchmark`: la lista de semillas de entrenamiento que se repiten."),
            ("`--output-dir`", "`runs`", "ruta",
             "Dónde se escribe el directorio de la ejecución."),
        ]

    lines += ["| Opción | Valor por defecto | Valores | Qué hace y cuándo cambiarla |", "|---|---|---|---|"]
    lines += [f"| {name} | {default} | {values} | {meaning} |" for name, default, values, meaning in rows]
    lines += [""]

    if not advanced:
        lines += [
            "### El script del laboratorio",
            "",
            f"`labs/{slug}/train.py` no es un programa distinto: fija el `--lab` y delega en la misma "
            "herramienta, de modo que estas dos líneas hacen exactamente lo mismo.",
            "",
            "```bash",
            f"python labs/{slug}/train.py --quick",
            f"neural-labs train --lab {slug} --quick",
            "```",
            "",
            "### Lo mismo desde Python",
            "",
            "Si prefieres trabajar en un cuaderno o llamar al laboratorio desde tu propio código, la "
            "misma ejecución se lanza así. La función devuelve un objeto con el directorio de la "
            "ejecución, las métricas y el historial ya cargados:",
            "",
            "```python",
            "from neural_labs.experiments import run_lab",
            "",
            "resultado = run_lab(",
            f'    "{slug}",',
            "    quick=True,          # False para la ejecución completa",
            '    config_name="baseline",',
            "    split_seed=42,       # fija la partición",
            "    training_seed=43,    # varía la inicialización",
            ")",
            "",
            "print(resultado.run_dir)   # dónde quedaron los archivos",
            "print(resultado.metrics)   # el diccionario de métricas finales",
            "```",
            "",
            "Y para preparar el dataset sin entrenar —útil para inspeccionarlo antes—:",
            "",
            "```python",
            "from neural_labs.datasets import prepare_dataset",
            "",
            f'datos = prepare_dataset("{slug}", quick=True, seed=42)',
            "print(datos.summary)       # tamaño de cada partición y metadatos de la fuente",
            "```",
            "",
        ]
    else:
        lines += [
            "### Lo mismo desde Python",
            "",
            "```python",
            "from neural_labs.advanced.training import train_advanced",
            "",
            "resultado = train_advanced(",
            f'    "{slug}",',
            "    quick=True,",
            "    split_seed=42,",
            "    training_seed=43,",
            ")",
            "",
            'print(resultado["run_dir"])',
            'print(resultado["metrics"])',
            "```",
            "",
        ]
    return lines


def _theory_embedded(lab: dict) -> list[str]:
    """Incrusta la explicación de `theory.md` dentro de la guía.

    Se mantiene el texto tal cual —es la fuente— y solo se rebaja el nivel de los
    encabezados para que encajen bajo la sección de la guía.
    """
    sections = _theory_sections(lab)
    wanted = [
        ("Idea central", "De qué trata"),
        ("Fundamento matemático", "La matemática, paso a paso"),
        ("Visualización específica", "Qué conviene graficar"),
    ]
    lines: list[str] = []
    for key, heading in wanted:
        body = sections.get(key)
        if not body:
            continue
        lines += [f"### {heading}", "", body, ""]
    return lines


def guia_doc(lab: dict, index: int, total: int, prev: dict | None, nxt: dict | None) -> str:
    catalog, lesson = lab["catalog"], lab["lesson"]
    advanced = lab["category"] == "Avanzada"
    objective = str(catalog.get("objective") or lesson.get("title") or lab["title"]).strip()
    metrics = catalog.get("metrics") or []
    selection = lab["baseline_cfg"].get("selection_metric") or catalog.get("selection_metric")
    reference = catalog.get("baseline")
    dataset_name = lab["dataset"].get("name") or catalog.get("dataset") or "—"
    source = lab["dataset"].get("source") or catalog.get("source") or "—"
    license_text = lab["dataset"].get("license") or catalog.get("license") or "—"
    hours = lesson.get("estimated_hours") or catalog.get("estimated_hours")
    level = str(lesson.get("level") or catalog.get("level") or "")
    level = LEVEL_ES.get(level.lower(), level)
    part = lab["part"]

    parts: list[str] = ["## 🎯 Qué vas a hacer aquí", "", objective, ""]

    situar = (f'Es la **ruta {index + 1} de {total}** del recorrido y pertenece a {part["emoji"]} '
              f'la parte {part["num"]}, *{part["title"]}*.')
    if prev:
        situar += f' Llegas desde **{prev["title"]}**'
        situar += f' y lo que hagas aquí lo da por supuesto **{nxt["title"]}**.' if nxt else "."
    elif nxt:
        situar += f' Es el punto de partida; después viene **{nxt["title"]}**.'
    parts += [situar, ""]

    ficha = [f"Trabajarás con el dataset **`{dataset_name}`** ({source}, licencia: {license_text})"]
    if reference:
        ficha.append(f"y tendrás que superar la línea base **{reference}**")
    if selection:
        ficha.append(f"decidiendo con la métrica `{selection}` medida sobre `validation`")
    situacion = ", ".join(ficha) + "."
    if level or hours:
        detalle = " Nivel " + level if level else ""
        detalle += (f", unas **{hours} horas** de dedicación" if hours and level
                    else f" Unas **{hours} horas** de dedicación" if hours else "")
        situacion += detalle.rstrip() + "." if detalle else ""
    parts += [situacion, ""]

    if catalog.get("input"):
        parts += [f'**Qué recibe el modelo como entrada:** {catalog["input"]}.', ""]

    prerequisites = lesson.get("prerequisites") or []
    if prerequisites:
        parts += ["**Lo que conviene traer resuelto de las rutas anteriores:** "
                  + ", ".join(str(item) for item in prerequisites) + ".", ""]

    outcomes = lesson.get("learning_outcomes") or []
    if outcomes:
        parts += ["**Al terminar deberías ser capaz de:**", ""]
        parts += [f"- {item}" for item in outcomes]
        parts += [""]

    parts += ["## 🧠 La teoría de este laboratorio", ""]
    embedded = _theory_embedded(lab)
    if embedded:
        parts += [
            "Esta sección es la explicación completa del tema. No hace falta abrir otro archivo para "
            "entender lo que viene después: aquí está la idea, la matemática que la sostiene y sus "
            "límites. (El mismo texto vive en `theory.md`, que es la fuente desde la que se genera "
            "esta guía, junto con la bibliografía del final.)",
            "",
        ]
        parts += embedded
    question = _theory_question(lab)
    if question:
        parts += [f"> **La pregunta que deberías poder responder al terminar:** {question}", ""]

    if metrics:
        listed = ", ".join(f"`{metric}`" for metric in metrics)
        parts += [
            "### Qué se mide y con qué se decide",
            "",
            f"El laboratorio reporta {listed}."
            + (f" De todas ellas, la que **decide** qué modelo se conserva es `{selection}`, y se mide "
               "siempre sobre `validation`: es la única forma de que `test` siga siendo una estimación "
               "honesta de lo que pasará con datos nuevos." if selection else ""),
            "",
        ]

    parts += _notebooks_section(lab)
    parts += _commands_section(lab)

    parts += ["## 🪜 Paso a paso", ""]
    parts += [
        "Cada paso dice qué ocurre por dentro, por qué se hace en ese orden y cómo comprobar que "
        "salió bien. El orden no es una convención de estilo: es el que ejecuta el código, y "
        "alterarlo invalida el resultado.",
        "",
    ]
    parts += _advanced_steps(lab) if advanced else _core_steps(lab)

    parts += ["## 🔍 Cómo leer lo que produce la ejecución", ""]
    parts += [
        "Cada ejecución escribe su propio directorio con nombre único, de modo que dos corridas nunca "
        "se pisan. Esto es lo que encontrarás dentro:",
        "",
    ]
    parts += _artifact_table(lab)
    parts += [""]

    parts += ["## ⚠️ Dónde suele perderse la gente", ""]
    parts += _pitfalls(lab)
    parts += [""]

    risks = _theory_sections(lab).get("Riesgos de interpretación") or \
        _theory_sections(lab).get("Riesgo de interpretación")
    if risks:
        parts += ["### Riesgos al interpretar los resultados", "", risks, ""]

    criteria = lesson.get("success_criteria") or []
    deliverables = lesson.get("deliverables") or []
    parts += ["## ✅ Antes de darlo por terminado", ""]
    if criteria:
        parts += ["El laboratorio está aprobado cuando se cumplen estos criterios:", ""]
        parts += [f"- [ ] {item}" for item in criteria]
        parts += [""]
    if deliverables:
        parts += ["Y cuando tienes estos entregables:", ""]
        parts += [f"- [ ] {item}" for item in deliverables]
        parts += [""]
    parts += [
        "El plan experimental con la tabla que hay que completar está en `experiments.md`, y las "
        "preguntas con su rúbrica, en `assessment.md`. Ambos documentos se abren desde la barra de "
        "navegación de arriba.",
        "",
        "### Para ir más lejos",
        "",
        "- Cambia una decisión experimental y justifícala con el resultado en `validation`, no con la intuición.",
        "- Analiza los errores por clase o por segmento: casi siempre se concentran en un subconjunto reconocible.",
        "- Compara costo, precisión y latencia; el mejor modelo no siempre es el que gana por décimas.",
        "- Documenta sesgos, limitaciones y usos para los que **no** recomendarías este modelo.",
        "",
    ]

    references = _theory_sections(lab).get("🔗 Referencias")
    catalog_file = "configs/advanced_tracks.yaml" if advanced else "configs/labs.yaml"
    code_file = ("src/neural_labs/advanced/training.py" if advanced else "src/neural_labs/experiments.py")
    parts += ["## 📚 Fuentes", ""]
    if references:
        parts += [
            "La teoría de arriba no es original de este repositorio: se apoya en la literatura de "
            "referencia del área y en los papers originales de cada arquitectura. Estas son las obras "
            "concretas, y lo que aporta cada una:",
            "",
            references,
            "",
        ]
    resources = [
        ("📄 `README.md`", "README.md", "Esta guía."),
        ("🧠 `theory.md`", "theory.md", "La teoría completa con su bibliografía; es la fuente del apartado teórico de arriba."),
        ("🔬 `experiments.md`", "experiments.md", "El plan experimental y la tabla multi-semilla que hay que completar."),
        ("📝 `assessment.md`", "assessment.md", "Las preguntas de evaluación y la rúbrica con la que se corrigen."),
        ("📓 `notebook.ipynb`", "notebook.ipynb", "El recorrido completo con todo el código escrito y ejecutable."),
        ("✏️ `notebook_student.ipynb`", "notebook_student.ipynb", "El mismo recorrido con las celdas de ejercicio vacías."),
        ("✅ `notebook_solution.ipynb`", "notebook_solution.ipynb", "Los ejercicios resueltos, para contrastar."),
        ("🖥️ `train.py`", "train.py", "El mismo entrenamiento desde la terminal, sin abrir un cuaderno."),
        ("🎛️ `configs/baseline.yaml`", "configs/baseline.yaml", "Épocas, lote, tasa de aprendizaje y qué recorta `--quick`."),
        ("🎚️ `configs/improved.yaml`", "configs/improved.yaml", "La configuración ampliada que se compara contra la base."),
        ("🗄️ `data/dataset.yaml`", "data/dataset.yaml", "Fuente, licencia, política de partición y límites del dataset."),
        ("🧾 `lesson.yaml`", "lesson.yaml", "Nivel, prerrequisitos, resultados de aprendizaje y criterios."),
        ("🖥️ `index.html`", "index.html", "Esta misma clase como página autocontenida, para leerla sin conexión."),
    ]
    rows = [f"| [{name}]({path}) | {description} |"
            for name, path, description in resources if (lab["dir"] / path).exists()]

    parts += [
        "### Los archivos de este laboratorio",
        "",
        "Todo lo que necesitas está en esta carpeta. Cada enlace abre el archivo directamente:",
        "",
        "| Archivo | Qué es |",
        "|---|---|",
        *rows,
        "",
        "Y fuera de la carpeta, tres referencias que esta guía usa: el catálogo "
        f"`{catalog_file}` —de donde salen el objetivo, la línea base y las métricas—, el código "
        f"`{code_file}` —que define el orden de los pasos y los archivos que escribe cada "
        "ejecución— y `docs/experiment-protocol.md`, con la regla general del protocolo.",
        "",
        "Los datasets se descargan de su proveedor original y conservan su licencia; este repositorio "
        "no los redistribuye ni sustituye una descarga fallida por datos generados.",
    ]
    return "\n".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Plan de experimentos (experiments.md)
# ──────────────────────────────────────────────────────────────────────────────

def experiments_doc(lab: dict) -> str:
    catalog = lab["catalog"]
    slug = lab["slug"]
    advanced = lab["category"] == "Avanzada"
    reference = catalog.get("baseline") or "la línea base declarada en el catálogo"
    selection = lab["baseline_cfg"].get("selection_metric") or catalog.get("selection_metric") or "la métrica de selección"
    experiment = str(catalog.get("experiment") or "").strip()
    objective = str(catalog.get("objective") or lab["title"]).strip()

    parts: list[str] = [
        "## Qué significa «experimentar» aquí",
        "",
        "Entrenar un modelo y mirar el número que sale **no** es un experimento: es una observación. "
        "Un experimento empieza por una afirmación que podría resultar falsa, sigue fijando todo lo que "
        "no se está poniendo a prueba, y termina midiendo si la afirmación se sostiene cuando se repite.",
        "",
        "La diferencia importa porque el entrenamiento de una red tiene azar por dentro: la "
        "inicialización de los pesos y el orden en que se barajan los lotes cambian el resultado aunque "
        "no cambies nada más. Si comparas una sola corrida contra otra sola corrida, no sabes si la "
        "diferencia viene de tu idea o de la semilla. Por eso todo lo que sigue está organizado para "
        "separar **la señal** (el efecto de lo que cambiaste) del **ruido** (la variabilidad propia del "
        "entrenamiento).",
        "",
        "## La hipótesis de este laboratorio",
        "",
        f"> {objective}",
        "",
        f"Formulada como algo que se puede refutar: **el modelo de este laboratorio supera a "
        f"{reference} en `{selection}`, y la diferencia es mayor que la dispersión entre semillas.**",
        "",
        "Qué la haría falsa —y esto también es un resultado que hay que reportar—:",
        "",
        f"- El modelo no supera a {reference}.",
        "- Lo supera, pero por menos de lo que varía el propio modelo al cambiar la semilla de entrenamiento.",
        "- Lo supera solo con una semilla concreta y no con las demás.",
        "- Lo supera a costa de un tiempo o un tamaño que el problema no justifica.",
        "",
    ]

    if experiment:
        parts += [
            "### El experimento propio de esta ruta",
            "",
            f"Además de la comparación con la línea base, aquí interesa una pregunta específica: "
            f"**{experiment.rstrip('.')}**. Es la comparación que da sentido al tema de este "
            "laboratorio; la de la línea base solo dice si el modelo sirve, mientras que esta dice "
            "*qué parte* del diseño es la que aporta.",
            "",
        ]

    parts += [
        "## Qué se varía y qué se mantiene fijo",
        "",
        "Un experimento es interpretable cuando cambia **una** cosa a la vez. Esta tabla dice qué papel "
        "juega cada elemento y, sobre todo, por qué:",
        "",
        "| Elemento | En este experimento | Por qué |",
        "|---|---|---|",
        "| Partición de los datos (`--split-seed`) | **Fija** en 42 | Si cambiara entre corridas, no "
        "podrías saber si la diferencia viene de tu modelo o de que le tocaron datos distintos. |",
        "| Semilla de entrenamiento (`--training-seed`) | **Varía**: 41, 42, 43 | Es exactamente el ruido "
        "que quieres medir. Sin varias semillas no tienes con qué comparar la mejora. |",
        f"| Configuración (`--config`) | **Varía**: `baseline` y `improved` | Es la intervención bajo "
        "estudio: lo único que estás poniendo a prueba. |",
        "| Dataset y preprocesamiento | **Fijos** | Ajustar la normalización o el vocabulario con otros "
        "datos cambiaría el problema, no el modelo. |",
        "| Presupuesto de épocas y criterio de parada | **Fijos dentro de cada configuración** | Dar más "
        "épocas a una variante que a otra es comparar dos cosas distintas. |",
        "| Hardware y versiones | **Registrados** en `environment.json` | No siempre se pueden fijar, "
        "pero sí dejar anotados: explican diferencias de tiempo y, a veces, de resultado. |",
        f"| El conjunto `test` | **Intacto** hasta el final | Se abre una sola vez, después de que "
        f"`{selection}` sobre `validation` haya elegido al ganador. |",
        "",
    ]

    comparison = config_comparison(lab)
    if comparison:
        parts += [
            "## Las dos configuraciones que vas a comparar",
            "",
            "El laboratorio trae dos configuraciones ya escritas. Estos son los parámetros en los que "
            "**difieren** —el resto es idéntico, que es justamente lo que hace legible la comparación—:",
            "",
            *comparison,
            "",
            "Elegir entre ellas es una decisión de `validation`. Si `improved` gana en validación, se "
            "queda; si no, la configuración base es la respuesta y hay que decirlo así.",
            "",
        ]

    if advanced:
        parts += [
            "## Cómo ejecutar la serie",
            "",
            "En las especializaciones no hay comando de repetición automática: se lanza el entrenamiento "
            "una vez por semilla, manteniendo fija la partición.",
            "",
            "```bash",
            f"neural-labs train-advanced --track {slug} --split-seed 42 --training-seed 41",
            f"neural-labs train-advanced --track {slug} --split-seed 42 --training-seed 42",
            f"neural-labs train-advanced --track {slug} --split-seed 42 --training-seed 43",
            "```",
            "",
            "Cada corrida escribe su propio directorio en `runs-advanced/`, con `metrics.json` e "
            "`history.json`.",
            "",
        ]
    else:
        parts += [
            "## Cómo ejecutar la serie",
            "",
            "El comando `benchmark` hace exactamente esto: repite el entrenamiento manteniendo la "
            "partición fija y cambiando solo la semilla de entrenamiento.",
            "",
            "```bash",
            f"neural-labs benchmark --lab {slug} \\",
            "  --config baseline --split-seed 42 --training-seeds 41 42 43",
            "",
            f"neural-labs benchmark --lab {slug} \\",
            "  --config improved --split-seed 42 --training-seeds 41 42 43",
            "```",
            "",
            "Por dentro, `benchmark` no hace magia: llama a `run_lab` una vez por semilla con la misma "
            "partición y resume los resultados. Desde Python es literalmente eso:",
            "",
            "```python",
            "from neural_labs.experiments import run_lab",
            "from neural_labs.benchmarking import summarize_benchmark",
            "",
            "registros = []",
            "for semilla in (41, 42, 43):",
            "    resultado = run_lab(",
            f'        "{slug}",',
            '        config_name="baseline",',
            "        split_seed=42,          # la partición NO cambia",
            "        training_seed=semilla,  # solo cambia la inicialización",
            "    )",
            '    registros.append({"training_seed": semilla, "metrics": resultado.metrics})',
            "",
            "print(summarize_benchmark(registros))   # media y dispersión entre semillas",
            "```",
            "",
            "Empieza con `--quick` para comprobar que la serie corre entera; los números que se reportan "
            "salen de la ejecución completa.",
            "",
        ]

    parts += [
        "## La tabla que debes completar",
        "",
        "| Variante | Semilla | Métrica en `validation` | Métrica en `test` | Tiempo | Parámetros | Observación |",
        "|---|---:|---:|---:|---:|---:|---|",
        "| baseline | 41 | | | | | |",
        "| baseline | 42 | | | | | |",
        "| baseline | 43 | | | | | |",
        "| improved | 41 | | | | | |",
        "| improved | 42 | | | | | |",
        "| improved | 43 | | | | | |",
        "",
        "De dónde sale cada columna, para que no haya que adivinarlo:",
        "",
        f"- **Métrica en `validation`**: el mejor valor de `{selection}` durante el entrenamiento; está "
        "en `history.csv` y resumido en `metrics.json`.",
        "- **Métrica en `test`**: el valor final, en `metrics.json`, ya con el modelo congelado.",
        "- **Tiempo**: `wall_time_seconds` de `metrics.json`.",
        "- **Parámetros**: `parameters` de `metrics.json`; es la medida honesta del tamaño del modelo.",
        "- **Observación**: lo que viste y los números no dicen —una curva que se dispara, una clase que "
        "concentra los errores, una corrida que no convergió—.",
        "",
        "## Cómo decidir con esos números",
        "",
        "1. **Compara medias, pero decide con la dispersión.** Calcula media y desviación entre las tres "
        "semillas de cada variante. Una mejora de 0,3 puntos entre variantes cuya dispersión interna es "
        "de 1,2 puntos no es una mejora: es ruido con buena suerte.",
        "2. **Si los rangos se solapan, dilo.** «No se observó una diferencia distinguible del ruido con "
        "tres semillas» es una conclusión legítima y mucho más útil que un número inventado de "
        "confianza.",
        "3. **Decide con `validation`.** La columna de `test` sirve para reportar el resultado final una "
        "vez, no para elegir la variante ganadora.",
        "4. **El costo forma parte del resultado.** Si `improved` gana por poco y cuesta el triple de "
        "tiempo, la conclusión honesta menciona ambas cosas.",
        f"5. **Vuelve a la línea base.** La comparación contra {reference} es la que dice si todo el "
        "aparato de la red neuronal estaba justificado para este problema.",
        "",
        "## Qué debe decir tu conclusión",
        "",
        "Una conclusión completa contiene cinco cosas, y se puede escribir en un párrafo:",
        "",
        "- **Magnitud**: cuánto mejoró, en qué métrica y sobre qué conjunto.",
        "- **Incertidumbre**: cuánto varió entre semillas, y si la mejora sobrevive a esa variación.",
        "- **Costo**: tiempo, memoria o tamaño adicional que hubo que pagar.",
        "- **Errores**: dónde falla el modelo, no solo cuánto acierta.",
        "- **Condiciones**: en qué circunstancias no esperarías que este resultado se repitiera.",
        "",
        "## Errores de diseño que invalidan el experimento",
        "",
        "- **Cambiar dos cosas a la vez.** Si tocas la arquitectura y la tasa de aprendizaje en la misma "
        "corrida, el resultado no atribuye el efecto a ninguna de las dos.",
        "- **Comparar con particiones distintas.** Es el error más frecuente y el más difícil de "
        "detectar después: parece una mejora del modelo y es un reparto distinto de los datos.",
        "- **Quedarse con la mejor semilla.** Elegir la corrida más favorable y reportar solo esa es "
        "seleccionar el ruido. Se reportan todas.",
        "- **Ajustar mirando `test`.** En cuanto una decisión se toma con el resultado de `test`, ese "
        "conjunto deja de estimar el rendimiento con datos nuevos.",
        "- **No registrar el entorno.** Sin versiones ni hardware, una diferencia de tiempo o de "
        "resultado se vuelve inexplicable meses después.",
    ]
    return "\n".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Evaluación (assessment.md)
# ──────────────────────────────────────────────────────────────────────────────

RUBRIC = [
    ("Integridad de los datos", "20 %",
     "mezcla particiones o no puede demostrar que no lo hizo",
     "las tres particiones están separadas y auditadas",
     "además documenta hashes, política de partición y justifica la estrategia elegida"),
    ("Implementación", "20 %",
     "no ejecuta o produce resultados irreproducibles",
     "entrena y evalúa siguiendo el protocolo",
     "código claro y reutilizable, con las decisiones explicadas"),
    ("Diseño experimental", "20 %",
     "un solo resultado aislado",
     "comparación controlada contra la línea base",
     "varias semillas, dispersión reportada y variables controladas explícitas"),
    ("Análisis", "25 %",
     "repite las métricas sin interpretarlas",
     "interpreta los errores y su distribución",
     "identifica sesgos, límites y costo, y distingue evidencia de suposición"),
    ("Comunicación", "15 %",
     "incompleta o sin contexto",
     "reporte entendible y model card presente",
     "conclusiones verificables por un tercero a partir de los artefactos"),
]


def assessment_doc(lab: dict) -> str:
    catalog, lesson = lab["catalog"], lab["lesson"]
    math = str(catalog.get("math") or "").strip().rstrip(".")
    reference = catalog.get("baseline") or "la línea base del laboratorio"
    selection = lab["baseline_cfg"].get("selection_metric") or catalog.get("selection_metric") or "la métrica de selección"
    question = _theory_question(lab)
    dataset_name = lab["dataset"].get("name") or catalog.get("dataset") or "el dataset"

    parts: list[str] = [
        "## Cómo se evalúa este laboratorio",
        "",
        "No se evalúa el número final. Un modelo con una métrica alta obtenida mirando `test`, o sin "
        "compararse con nada, vale menos que uno modesto cuyo resultado se puede auditar. Lo que se "
        "califica es el **proceso**: si las particiones están limpias, si la decisión se tomó donde "
        "debía, si la conclusión distingue lo que se midió de lo que se supone.",
        "",
        "## Evidencias que debes entregar",
        "",
        "| Evidencia | Dónde vive | Por qué se pide |",
        "|---|---|---|",
        "| Dataset preparado y auditado | salida de `neural-labs audit` | Es la única prueba de que el "
        "resultado no está contaminado por una fuga entre particiones. |",
        "| Notebook ejecutado sin celdas omitidas | `notebook.ipynb` | Una celda saltada suele ser justo "
        "la que rompía el argumento. |",
        f"| Comparación contra {reference} | `metrics.json` y `baseline_metrics.json` | Sin línea base, "
        "una métrica no dice si el modelo aporta algo. |",
        "| Al menos tres semillas de entrenamiento, o la justificación de por qué no | salida de "
        "`benchmark` | Una sola corrida no permite separar mejora de azar. |",
        "| Análisis de errores y limitaciones | tu reporte | Saber *dónde* falla vale más que saber "
        "cuánto acierta. |",
        "| Model card actualizada | `model_card.md` | Es lo que permite a otra persona saber cuándo "
        "**no** debería usar tu modelo. |",
        "",
        "## Las preguntas, y qué se busca en tu respuesta",
        "",
        "No se corrige la longitud de la respuesta, sino si demuestra comprensión. Debajo de cada "
        "pregunta está lo que una buena respuesta debería contener.",
        "",
    ]

    questions: list[tuple[str, str]] = []
    if math:
        questions.append((
            f"Explica con tus palabras: {math}.",
            "Una buena respuesta conecta cuatro cosas —cómo se representa la entrada, qué calcula el "
            "modelo, qué mide la función de pérdida y cómo se actualizan los pesos— en vez de repetir "
            "la definición del libro. Si puedes explicarlo sin la fórmula delante, lo entendiste.",
        ))
    questions.append((
        "¿Qué información del dataset solo puede usarse durante el entrenamiento?",
        "Se espera que nombres casos concretos: las estadísticas de normalización, el vocabulario, la "
        "selección de variables, los umbrales. Todo eso se ajusta **solo** con `train`; calcularlo "
        "sobre el conjunto completo es una fuga silenciosa que infla el resultado sin dar ningún aviso.",
    ))
    questions.append((
        f"¿Por qué {reference} es una comparación razonable para este problema?",
        "Una buena respuesta explica qué captura la línea base y qué no, y por qué superarla —o no "
        "superarla— es informativo aquí. Si la línea base ya resuelve el problema, la conclusión "
        "correcta es que la red no estaba justificada.",
    ))
    if question:
        questions.append((
            question,
            "Esta es la pregunta propia del tema. Responde con evidencia de tu ejecución —predicciones, "
            "matriz de confusión, curvas, artefactos del directorio de la corrida—, no con una "
            "impresión general.",
        ))
    questions.append((
        f"¿Qué te dice `{selection}` que no te dirían las otras métricas?",
        "Cada métrica pondera distinto los errores. Se espera que expliques por qué esa es la que "
        "decide aquí y en qué situación sería una mala elección.",
    ))
    questions.append((
        "¿Qué cambiarías antes de usar este modelo fuera del laboratorio?",
        f"Aquí se evalúa el criterio, no la técnica: licencias y condiciones de uso de `{dataset_name}`, "
        "representatividad de la población, calibración de las probabilidades, vigilancia de la deriva, "
        "desempeño por subgrupo y supervisión humana. Un «funcionaría bien» sin condiciones se corrige "
        "como respuesta incompleta.",
    ))

    for number, (text, guidance) in enumerate(questions, start=1):
        parts += [f"**{number}. {text}**", "", f"*Qué se busca:* {guidance}", ""]

    parts += [
        "## La rúbrica, explicada",
        "",
        "| Criterio | Insuficiente | Adecuado | Excelente | Peso |",
        "|---|---|---|---|---:|",
    ]
    parts += [f"| {name} | {bad} | {ok} | {great} | {weight} |"
              for name, weight, bad, ok, great in RUBRIC]
    parts += [
        "",
        "La diferencia entre *adecuado* y *excelente* casi nunca está en la métrica: está en si el "
        "trabajo permite que otra persona llegue a la misma conclusión con los artefactos que dejaste. "
        "Un resultado peor, bien medido y bien explicado, se califica por encima de uno mejor que no se "
        "puede auditar.",
        "",
        "**La aprobación exige al menos 70 % y cero errores críticos de fuga de datos.** La fuga es "
        "eliminatoria por sí sola porque invalida todas las demás cifras del trabajo, por buenas que "
        "parezcan.",
        "",
        "## Autoevaluación antes de entregar",
        "",
        "- [ ] Puedo explicar el laboratorio a alguien que no lo hizo, sin leer el código.",
        "- [ ] Sé qué decisión tomé en cada paso y con qué evidencia la tomé.",
        f"- [ ] Miré `test` una sola vez, después de que existiera `experiment.lock.json`.",
        "- [ ] Mi conclusión dice magnitud, incertidumbre, costo, errores y condiciones.",
        "- [ ] Puedo señalar al menos una limitación real de mi resultado.",
    ]
    if lesson.get("success_criteria"):
        parts += [""]
        parts += [f"- [ ] {item}" for item in lesson["success_criteria"]]
    return "\n".join(parts)
# ──────────────────────────────────────────────────────────────────────────────
# Páginas de parte (parts/*.md)
# ──────────────────────────────────────────────────────────────────────────────

def _lab_summary(lab: dict) -> str:
    """Qué resuelve la clase, según el catálogo del repositorio."""
    text = str(lab["catalog"].get("objective") or lab["lesson"].get("title") or lab["title"]).strip()
    return text.rstrip(".")


def _lab_hours(lab: dict) -> int | None:
    hours = lab["lesson"].get("estimated_hours") or lab["catalog"].get("estimated_hours")
    return int(hours) if hours else None


def part_page(part: dict, labs: list[dict], prev: dict | None, nxt: dict | None) -> str:
    members = [lab for lab in labs if lab["part"] is part]
    positions = {lab["slug"]: index + 1 for index, lab in enumerate(labs)}
    hours = [h for h in (_lab_hours(lab) for lab in members) if h]
    # Se ordenan por dificultad creciente, no alfabéticamente.
    order = ["fundamentos", "básico", "intermedio", "intermedio-avanzado", "avanzado", "experto"]
    found = {LEVEL_ES.get(str(lab["lesson"].get("level") or lab["catalog"].get("level") or "").lower(),
                          str(lab["lesson"].get("level") or lab["catalog"].get("level") or ""))
             for lab in members} - {""}
    levels = sorted(found, key=lambda level: (order.index(level) if level in order else len(order), level))

    meta = [f"**Rutas:** {part['first']:02d}–{part['last']:02d}", f"**Clases:** {len(members)}"]
    if levels:
        meta.append(f"**Nivel:** {' · '.join(levels)}")
    if hours:
        total = sum(hours)
        meta.append(f"**Dedicación estimada:** ~{total} h"
                    + ("" if len(hours) == len(members) else " (las avanzadas no la declaran)"))

    nodes = "\n".join(
        f'    L{lab["num"]}["{lab["num"]}<br/>{lab["title"]}"]' for lab in members
    )
    edges = "\n".join(
        f'    L{members[i]["num"]} --> L{members[i + 1]["num"]}' for i in range(len(members) - 1)
    )

    rows = []
    for lab in members:
        hours_cell = f'{_lab_hours(lab)}' if _lab_hours(lab) else "—"
        rows.append(
            f'| {lab["num"]} | {lab["emoji"]} [{lab["title"]}](../{lab["base"]}/{lab["slug"]}/README.md) '
            f'| {_lab_summary(lab)} | `{lab["dataset"].get("name") or lab["catalog"].get("dataset") or "—"}` '
            f'| {hours_cell} |'
        )

    first = members[0]
    docs_row = " · ".join(
        f'[{emoji} {name}](../{first["base"]}/{first["slug"]}/{doc})'
        for doc, emoji, name in DOCS if (first["dir"] / doc).exists()
    )

    prev_cell = (f'[⬅️ Parte {prev["num"]} — {prev["title"]}]({prev["slug"]}.md)'
                 if prev else "⬅️ *primera parte*")
    next_cell = (f'[Parte {nxt["num"]} — {nxt["title"]} ➡️]({nxt["slug"]}.md)'
                 if nxt else "*última parte* ➡️")

    lines = [
        f'# {part["emoji"]} Parte {part["num"]} — {part["title"]}',
        "",
        f"> 🧭 {prev_cell} · [🏠 Índice de partes](README.md) · [📘 Portada](../README.md) · {next_cell}",
        "",
        " · ".join(meta),
        "",
        part["summary"],
        "",
        "## 🧭 Secuencia de la parte",
        "",
        "```mermaid",
        "flowchart LR",
        nodes,
        edges,
        "```",
        "",
        "## 📚 Clases de esta parte",
        "",
        "| # | Clase | Qué resuelve | Dataset | Horas |",
        "|---:|---|---|---|---:|",
        "\n".join(rows),
        "",
        f'> Empieza por {first["emoji"]} **[{first["title"]}](../{first["base"]}/{first["slug"]}/README.md)** '
        f'(ruta {positions[first["slug"]]} de {len(labs)}). Sus documentos: {docs_row}.',
        "",
        "## 🎯 Qué llevas al terminar",
        "",
        f'Al completar esta parte, {part["outcome"]}',
        "",
        "Todas las clases comparten el mismo contrato: los transformadores se ajustan solo con",
        "`train`, `validation` decide el modelo y `test` se abre una única vez tras escribir",
        "`experiment.lock.json`.",
        "",
        "---",
        "",
        f"{prev_cell} · [🏠 Índice de partes](README.md) · [📘 Portada del repositorio](../README.md) · {next_cell}",
        "",
    ]
    return "\n".join(lines)


def parts_index(labs: list[dict]) -> str:
    rows = []
    for index, part in enumerate(PARTS):
        members = [lab for lab in labs if lab["part"] is part]
        rows.append(
            f'| {part["emoji"]} **{part["num"]}** | [{part["title"]}]({part["slug"]}.md) '
            f'| {part["first"]:02d}–{part["last"]:02d} | {len(members)} | {part["outcome"]} |'
        )

    lines = [
        "# 🗺️ Índice del recorrido",
        "",
        "> 🧭 [📘 Portada del repositorio](../README.md) · "
        f'[🌐 Sitio de estudio]({SITE}/) · [🖥️ Índice HTML offline](../index.html)',
        "",
        f"Las **{len(labs)} rutas** se estudian en orden, de la **00** a la **{labs[-1]['num']}**.",
        "Las siete partes de abajo son tramos **contiguos** de esa misma secuencia: cada una agrupa",
        "las clases consecutivas que comparten propósito, y termina justo donde empieza la siguiente.",
        "",
        "| Parte | Título | Rutas | Clases | Qué llevas al terminar |",
        "|:---:|---|:---:|:---:|---|",
        "\n".join(rows),
        "",
        "## 📚 Todas las clases, en orden",
        "",
        "| # | Clase | Parte | Dataset |",
        "|---:|---|---|---|",
    ]
    for lab in labs:
        part = lab["part"]
        lines.append(
            f'| {lab["num"]} | {lab["emoji"]} [{lab["title"]}](../{lab["base"]}/{lab["slug"]}/README.md) '
            f'| {part["emoji"]} [{part["num"]}]({part["slug"]}.md) '
            f'| `{lab["dataset"].get("name") or lab["catalog"].get("dataset") or "—"}` |'
        )
    lines += [
        "",
        "---",
        "",
        f'[📘 Portada del repositorio](../README.md) · [▶️ Empezar por la ruta {labs[0]["num"]}]'
        f'(../{labs[0]["base"]}/{labs[0]["slug"]}/README.md)',
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="no escribe: falla si algún documento quedó desfasado.")
    args = parser.parse_args()

    labs = collect_labs()
    total = len(labs)
    stale: list[str] = []
    changed = 0

    pending: list[tuple[Path, str]] = []
    for index, lab in enumerate(labs):
        prev = labs[index - 1] if index > 0 else None
        nxt = labs[index + 1] if index < total - 1 else None
        for doc, _, _ in DOCS:
            path = lab["dir"] / doc
            if path.exists():
                pending.append((path, render_doc(lab, doc, index, total, prev, nxt)))

    parts_dir = ROOT / "parts"
    parts_dir.mkdir(exist_ok=True)
    pending.append((parts_dir / "README.md", parts_index(labs)))
    for index, part in enumerate(PARTS):
        prev = PARTS[index - 1] if index > 0 else None
        nxt = PARTS[index + 1] if index < len(PARTS) - 1 else None
        pending.append((parts_dir / f"{part['slug']}.md", part_page(part, labs, prev, nxt)))

    for path, content in pending:
        if path.exists() and path.read_text(encoding="utf-8") == content:
            continue
        if args.check:
            stale.append(str(path.relative_to(ROOT)).replace("\\", "/"))
        else:
            path.write_text(content, encoding="utf-8")
            changed += 1

    if args.check:
        if stale:
            print("Documentos desfasados (ejecuta scripts/build_lab_docs.py):")
            for item in stale:
                print(f"  - {item}")
            return 1
        print(f"Markdown al día: {total} laboratorios y {len(PARTS)} partes.")
        return 0

    print(f"Markdown actualizado: {changed} documentos "
          f"({total} laboratorios, {len(PARTS)} partes y su índice).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
