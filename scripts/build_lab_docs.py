#!/usr/bin/env python3
"""Construye la capa navegable y la ficha de cada laboratorio en Markdown.

Cada laboratorio publica cuatro documentos —`README.md`, `theory.md`,
`experiments.md` y `assessment.md`— y este script los mantiene enlazados entre sí
y con el resto del recorrido:

* **Navegación superior** (`<!-- nav-top -->`): posición en el recorrido
  (`Ruta 04 / 31`), salto al laboratorio anterior y siguiente, vuelta al índice y
  barra con los cuatro documentos del laboratorio, marcando el actual.
* **Ficha** (`<!-- ficha -->`, solo en `README.md`): distintivos, tabla de
  metadatos, resultados de aprendizaje, prerrequisitos, comparación
  `baseline` vs `improved` y tabla de recursos, todo enlazado.
* **Navegación inferior** (`<!-- nav-bottom -->`): tabla anterior / índice /
  siguiente, enlaces a los documentos y cuadernos del laboratorio, y salida hacia
  la portada, el sitio de estudio y la página HTML local.

Los tres bloques se delimitan con marcadores HTML, de modo que el script es
idempotente: se puede volver a ejecutar cuando cambien títulos, orden o
configuraciones. El contenido redactado a mano (teoría, experimentos, rúbricas)
nunca se reescribe.

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
FICHA_RE = re.compile(r"\n?<!-- ficha -->.*?<!-- /ficha -->\n?", re.DOTALL)

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
# Ficha del laboratorio (solo README.md)
# ──────────────────────────────────────────────────────────────────────────────

def _badge(label_text: str, value: str, color: str) -> str:
    def clean(text: str) -> str:
        return (str(text).replace("-", "--").replace("_", "__")
                .replace(" ", "%20").replace("|", "%7C"))
    return f"![{label_text}](https://img.shields.io/badge/{clean(label_text)}-{clean(value)}-{color}?style=flat-square)"


def _bullets(items: list, empty: str = "") -> str:
    rows = [f"- {str(item).strip()}" for item in items if str(item).strip()]
    return "\n".join(rows) if rows else empty


def ficha_block(lab: dict, index: int, total: int, prev: dict | None) -> str:
    lesson, dataset, catalog = lab["lesson"], lab["dataset"], lab["catalog"]
    baseline, improved = lab["baseline_cfg"], lab["improved_cfg"]

    level = str(lesson.get("level") or catalog.get("level") or "—")
    level = LEVEL_ES.get(level.lower(), level)
    hours = lesson.get("estimated_hours") or catalog.get("estimated_hours")
    dataset_name = dataset.get("name") or catalog.get("dataset") or lesson.get("dataset") or "—"
    source = dataset.get("source") or catalog.get("source") or "—"
    source_url = (dataset.get("source_url") or catalog.get("source_ref")
                  or (dataset.get("provenance") or {}).get("reference")
                  or (lesson.get("dataset") if str(lesson.get("dataset", "")).startswith("http") else "")
                  or "")
    license_text = dataset.get("license") or catalog.get("license") or "—"
    task = dataset.get("task") or catalog.get("task") or "—"
    domain = catalog.get("domain") or lesson.get("domain")
    architecture = dataset.get("architecture") or catalog.get("architecture") or catalog.get("model") or "—"
    metric = baseline.get("selection_metric") or catalog.get("selection_metric") or "—"
    reference_model = baseline.get("baseline") or catalog.get("baseline")

    badges = [
        _badge("ruta", f"{index + 1} de {total}", "7c5cff"),
        _badge("nivel", level, BADGE_COLORS.get(level.lower(), "6b7d92")),
        _badge("categoría", lab["category"], "2e8b57"),
    ]
    if hours:
        badges.append(_badge("horas", f"~{hours} h", "f0b429"))
    badges.append(_badge("dataset", dataset_name, "1f6feb"))
    badges.append(_badge("selección", metric, "8957e5"))

    rows = [
        ("🧭 Posición", f"Ruta **{index + 1} de {total}** del recorrido · categoría {lab['category'].lower()}"),
        ("🎚️ Nivel", level),
    ]
    if hours:
        rows.append(("⏱️ Dedicación estimada", f"{hours} horas"))
    if task != "—":
        rows.append(("🧩 Tarea", f"`{task}`"))
    if domain:
        rows.append(("🗺️ Dominio", f"`{domain}`"))
    rows.append(("🏗️ Arquitectura", f"`{architecture}`"))
    dataset_cell = f"[`{dataset_name}`]({source_url})" if source_url else f"`{dataset_name}`"
    rows.append(("🗄️ Dataset", f"{dataset_cell} — {source}"))
    rows.append(("⚖️ Licencia del dataset", license_text))
    rows.append(("🎯 Métrica de selección", f"`{metric}` sobre `validation`"))
    if reference_model:
        rows.append(("📏 Línea base a superar", reference_model))
    rows.append(("🔒 Política de `test`", "se abre una sola vez, tras escribir `experiment.lock.json`"))

    table = "\n".join(f"| {name} | {value} |" for name, value in rows)

    parts = [
        "<!-- ficha -->",
        "## 📋 Ficha del laboratorio",
        "",
        " ".join(badges),
        "",
        "| Campo | Valor |",
        "|---|---|",
        table,
        "",
    ]

    outcomes = lesson.get("learning_outcomes") or []
    if outcomes:
        parts += ["### 🎯 Qué vas a poder hacer al terminar", "", _bullets(outcomes), ""]

    prerequisites = lesson.get("prerequisites") or []
    if prerequisites:
        prev_hint = f" Viniendo de [{label(prev)}]({doc_link(prev, 'README.md')})." if prev else ""
        parts += [
            "### 🧩 Prerrequisitos",
            "",
            _bullets(prerequisites),
            "",
            f"> Si alguno te falta, retrocede antes de continuar.{prev_hint}",
            "",
        ]

    if baseline and improved:
        keys = [
            ("epochs", "Épocas"),
            ("batch_size", "Tamaño de lote"),
            ("learning_rate", "Tasa de aprendizaje"),
            ("patience", "Paciencia (early stopping)"),
            ("amp", "Precisión mixta (AMP)"),
            ("num_workers", "Procesos de carga"),
        ]
        def show(value: object) -> str:
            if isinstance(value, bool):
                return "sí" if value else "no"
            return f"`{value}`"

        config_rows = [
            f"| {name} | {show(baseline[key])} | {show(improved[key])} |"
            for key, name in keys
            if key in baseline and key in improved and baseline[key] != improved[key]
        ]
        if config_rows:
            parts += [
                "### ⚙️ `baseline` frente a `improved`",
                "",
                "| Parámetro | [`baseline.yaml`](configs/baseline.yaml) | [`improved.yaml`](configs/improved.yaml) |",
                "|---|---|---|",
                "\n".join(config_rows),
                "",
                "> Solo se muestran los parámetros en los que ambas configuraciones difieren. "
                "La elección entre una y otra se decide con `validation`, nunca con `test`.",
                "",
            ]

    deliverables = lesson.get("deliverables") or []
    criteria = lesson.get("success_criteria") or []
    if deliverables or criteria:
        parts += ["### 📦 Entregables y criterios de aceptación", ""]
        if deliverables:
            parts += ["**Entregables**", "", _bullets(deliverables), ""]
        if criteria:
            parts += ["**Criterios de éxito**", "", _bullets(criteria), ""]

    resources = [
        ("🧠 Teoría y referencias", "theory.md"),
        ("🔬 Plan de experimentos", "experiments.md"),
        ("📝 Evaluación y rúbrica", "assessment.md"),
        ("📓 Notebook de recorrido", "notebook.ipynb"),
        ("✏️ Notebook de estudiante", "notebook_student.ipynb"),
        ("✅ Notebook de solución", "notebook_solution.ipynb"),
        ("🖥️ Script de terminal", "train.py"),
        ("🎛️ Configuración base", "configs/baseline.yaml"),
        ("🎚️ Configuración ampliada", "configs/improved.yaml"),
        ("🗄️ Ficha del dataset", "data/dataset.yaml"),
        ("🧾 Metadatos de la lección", "lesson.yaml"),
    ]
    resource_rows = [
        f"| {name} | [`{path}`]({path}) |"
        for name, path in resources
        if (lab["dir"] / path).exists()
    ]
    if resource_rows:
        parts += [
            "### 🗂️ Recursos del laboratorio",
            "",
            "| Recurso | Archivo |",
            "|---|---|",
            "\n".join(resource_rows),
            "",
        ]

    parts.append("<!-- /ficha -->")
    return "\n".join(parts).rstrip()


# ──────────────────────────────────────────────────────────────────────────────
# Aplicación de los bloques
# ──────────────────────────────────────────────────────────────────────────────

def render_doc(lab: dict, doc: str, index: int, total: int,
               prev: dict | None, nxt: dict | None) -> str:
    path = lab["dir"] / doc

    # La guía (`README.md`) se genera entera desde los datos del repositorio: su
    # contenido anterior era una plantilla derivada del catálogo, y mantenerlo a
    # mano solo abría la puerta a que dijera algo que el código ya no hace. Los
    # demás documentos —teoría, experimentos y evaluación— llevan texto redactado
    # y solo se les añade la navegación.
    if doc == "README.md":
        return "\n".join([
            f'# {lab["title"]}',
            "",
            top_block(lab, prev, nxt, index, total, doc),
            "",
            ficha_block(lab, index, total, prev),
            "",
            guia_block(lab, index, total, prev, nxt),
            "",
            bottom_block(lab, prev, nxt, doc),
            "",
        ])

    text = path.read_text(encoding="utf-8")
    # Se retiran los bloques generados para volver a insertarlos actualizados.
    text = TOP_RE.sub("\n", text)
    text = BOTTOM_RE.sub("\n", text)
    text = FICHA_RE.sub("\n", text)
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


def _theory_question(lab: dict) -> str | None:
    """Pregunta crítica que ya vive en el `theory.md` del laboratorio."""
    path = lab["dir"] / "theory.md"
    if not path.exists():
        return None
    match = re.search(r"## Pregunta crítica\s*\n+>\s*(.+)", path.read_text(encoding="utf-8"))
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


def guia_block(lab: dict, index: int, total: int, prev: dict | None, nxt: dict | None) -> str:
    catalog, lesson = lab["catalog"], lab["lesson"]
    advanced = lab["category"] == "Avanzada"
    objective = str(catalog.get("objective") or lesson.get("title") or lab["title"]).strip()
    math = str(catalog.get("math") or "").strip()
    metrics = catalog.get("metrics") or []
    part = lab["part"]

    parts: list[str] = ["<!-- guia -->", "## 🎯 Qué vas a hacer aquí", "", objective, ""]

    context = [f'Es la **ruta {index + 1} de {total}** y pertenece a {part["emoji"]} '
               f'[la parte {part["num"]}, {part["title"]}](../../parts/{part["slug"]}.md).']
    if prev:
        context.append(f'Llegas desde [{label(prev)}]({doc_link(prev, "README.md")})')
        context[-1] += (f' y lo que aprendas aquí lo da por supuesto [{label(nxt)}]({doc_link(nxt, "README.md")}).'
                        if nxt else ".")
    elif nxt:
        context.append(f'Es el punto de partida del recorrido; sigue [{label(nxt)}]({doc_link(nxt, "README.md")}).')
    parts += [" ".join(context), ""]

    if catalog.get("input"):
        parts += [f'**Entrada del modelo:** {catalog["input"]}.', ""]

    parts += ["## 🧠 La idea que se pone a prueba", ""]
    if math:
        parts += [f"Este laboratorio trabaja **{math.rstrip('.')}**.", ""]
    parts += [
        "El desarrollo completo —qué calcula cada parte, de dónde sale la fórmula, qué riesgos tiene "
        "interpretarla mal y en qué libros y papers se estudia— está en [`theory.md`](theory.md). "
        "Léelo antes de entrenar: los pasos de abajo te dicen *qué* hacer, y la teoría, *por qué* "
        "funciona y cuándo deja de funcionar.",
        "",
    ]
    question = _theory_question(lab)
    if question:
        parts += [f"> **La pregunta que deberías poder responder al final:** {question}", ""]
    if metrics:
        listed = ", ".join(f"`{metric}`" for metric in metrics)
        selection = lab["baseline_cfg"].get("selection_metric") or catalog.get("selection_metric")
        parts += [
            f"**Métricas que se reportan:** {listed}."
            + (f" La selección del modelo se decide con `{selection}` sobre `validation`." if selection else ""),
            "",
        ]

    parts += ["## 🪜 Paso a paso", ""]
    parts += [
        "Cada paso dice qué ocurre, por qué se hace así y cómo comprobar que salió bien. El orden no es una "
        "convención: es el que ejecuta el código, y cambiarlo rompe la validez del resultado.",
        "",
    ]
    parts += _advanced_steps(lab) if advanced else _core_steps(lab)

    parts += ["## 🔍 Cómo leer lo que produce la ejecución", ""]
    parts += [
        "Cada ejecución escribe su propio directorio. Estos son los archivos que encontrarás y para qué sirve "
        "cada uno:",
        "",
    ]
    parts += _artifact_table(lab)
    parts += [""]

    parts += ["## ⚠️ Dónde suele perderse la gente", ""]
    parts += _pitfalls(lab)
    parts += [""]

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
        "Las preguntas y la rúbrica con la que se corrige están en [`assessment.md`](assessment.md); "
        "el plan de experimentos y la tabla multi-semilla que hay que completar, en "
        "[`experiments.md`](experiments.md).",
        "",
        "## 🧪 Para ir más lejos",
        "",
        "- Cambia una decisión experimental y justifícala con el resultado en `validation`, no con la intuición.",
        "- Analiza los errores por clase o por segmento: casi siempre se concentran en un subconjunto reconocible.",
        "- Compara costo, precisión y latencia; el mejor modelo no siempre es el que gana por décimas.",
        "- Documenta sesgos, limitaciones y usos para los que **no** recomendarías este modelo.",
        "",
    ]

    catalog_file = "configs/advanced_tracks.yaml" if advanced else "configs/labs.yaml"
    code_file = ("src/neural_labs/advanced/training.py" if advanced
                 else "src/neural_labs/experiments.py")
    parts += [
        "## 📚 De dónde sale cada cosa de esta guía",
        "",
        "Nada de lo anterior está escrito de memoria. Cada afirmación se puede comprobar en un archivo "
        "concreto del repositorio:",
        "",
        "| Lo que dice la guía | Dónde comprobarlo |",
        "|---|---|",
        f"| Objetivo, línea base, métricas y arquitectura | [`{catalog_file}`](../../{catalog_file}) |",
        "| Fuente, licencia, procedencia y límites del dataset | [`data/dataset.yaml`](data/dataset.yaml) |",
        "| Épocas, tamaño de lote, tasa de aprendizaje y recorte de `--quick` | "
        "[`configs/baseline.yaml`](configs/baseline.yaml) · [`configs/improved.yaml`](configs/improved.yaml) |",
        "| Nivel, prerrequisitos, resultados de aprendizaje y criterios | [`lesson.yaml`](lesson.yaml) |",
        f"| El orden de los pasos y los archivos que escribe cada ejecución | [`{code_file}`](../../{code_file}) |",
        "| La teoría, los papers y los libros de referencia | [`theory.md`](theory.md), sección 🔗 Referencias |",
        "| La regla general del protocolo | "
        "[`docs/experiment-protocol.md`](../../docs/experiment-protocol.md) |",
        "",
        "Los datasets se descargan de su proveedor original y conservan su propia licencia; este repositorio "
        "no los redistribuye ni sustituye una descarga fallida por datos generados.",
        "<!-- /guia -->",
    ]
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
