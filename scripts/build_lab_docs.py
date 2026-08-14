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
    text = path.read_text(encoding="utf-8")

    # Se retiran los bloques generados para volver a insertarlos actualizados.
    text = TOP_RE.sub("\n", text)
    text = BOTTOM_RE.sub("\n", text)
    text = FICHA_RE.sub("\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"

    header = top_block(lab, prev, nxt, index, total, doc)
    if doc == "README.md":
        header += "\n\n" + ficha_block(lab, index, total, prev)

    lines = text.splitlines()
    out: list[str] = []
    inserted = False
    for line in lines:
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
