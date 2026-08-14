#!/usr/bin/env python3
"""Genera la página HTML de cada laboratorio, junto a su Markdown.

A diferencia de `scripts/generate_site.py` —que construye el sitio de GitHub Pages
en `site/`— este script escribe una página **autocontenida** dentro de la propia
carpeta del laboratorio:

    labs/<NN_slug>/index.html
    advanced_labs/<NN_slug>/index.html
    index.html                          — portada offline con las 31 rutas

Se versionan en el repositorio para que cualquier aplicación (visor de escritorio,
app móvil, ZIP de estudio, servidor de aula) pueda abrirlas sin conexión y sin
proceso de compilación. Por eso el CSS va embebido y los enlaces son relativos:

* anterior / siguiente        → `../../<base>/<slug>/index.html`
* índice del recorrido        → `../../index.html`
* documentos del laboratorio  → anclas dentro de la misma página
* cuadernos, configs y datos  → rutas relativas al propio laboratorio

Fuente única: el Markdown del laboratorio, que mantiene `scripts/build_lab_docs.py`.
Ejecutar siempre en ese orden (primero el Markdown, después el HTML).

    python scripts/generate_lab_html.py            # escribe las páginas
    python scripts/generate_lab_html.py --check    # falla si quedaron desfasadas
"""
from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path

import markdown  # type: ignore

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_site import (  # noqa: E402
    BLOB,
    LAB_DOCS,
    LAB_EMOJI,
    NAV_RE,
    REPO,
    STYLES,
    _first_title,
)

ROOT = Path(__file__).resolve().parents[1]
SITE = "https://vladimiracunadev-create.github.io/neural-network-training-labs"

MD = markdown.Markdown(
    extensions=["extra", "toc", "sane_lists", "tables", "fenced_code", "admonition"],
    output_format="html5",
)

# Enlaces del Markdown hacia otro laboratorio: ../../<base>/<slug>/<doc>
CROSS_LAB_RE = re.compile(r"^\.\./\.\./(labs|advanced_labs)/([^/]+)/(.+)$")

# Distintivos de shields.io del Markdown: en el HTML local se convierten en chips
# CSS para que la página no dependa de la red.
BADGE_RE = re.compile(
    r'<img[^>]*?src="https://img\.shields\.io/badge/([^"?]+)(?:\?[^"]*)?"[^>]*?>'
)

CHIP_CSS = """
/* Distintivos: chips locales que reemplazan las imágenes de shields.io */
.chips { display: flex; flex-wrap: wrap; gap: 8px; margin: 2px 0 18px; }
.chip { display: inline-flex; align-items: stretch; border-radius: 999px; overflow: hidden;
  font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: 0.78rem; font-weight: 600;
  border: 1px solid var(--line); box-shadow: var(--shadow); }
.chip b { background: var(--bg-soft); color: var(--ink-soft); padding: 4px 10px; font-weight: 600; }
.chip span { padding: 4px 11px; color: #fff; }
"""


def _unescape_badge(part: str) -> str:
    """Deshace el escapado de shields.io: `--` → `-`, `__` → `_`, `%20` → espacio."""
    from urllib.parse import unquote
    return unquote(part).replace("--", "\0").replace("__", "\1") \
        .replace("-", " ").replace("\0", "-").replace("\1", "_")


def badges_to_chips(body: str) -> str:
    def repl(match: re.Match) -> str:
        pieces = match.group(1).split("-")
        if len(pieces) < 3:
            return match.group(0)
        color = pieces[-1]
        label = _unescape_badge("-".join(pieces[:-2]))
        value = _unescape_badge(pieces[-2])
        return (f'<span class="chip"><b>{html.escape(label)}</b>'
                f'<span style="background:#{html.escape(color)}">{html.escape(value)}</span></span>')

    body = BADGE_RE.sub(repl, body)
    # Los párrafos que solo contienen chips se muestran como una fila compacta.
    return re.sub(r'<p>((?:\s*<span class="chip">.*?</span>\s*)+)</p>',
                  r'<div class="chips">\1</div>', body, flags=re.DOTALL)


def collect_labs() -> list[dict]:
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
            labs.append({
                "slug": slug,
                "base": base,
                "num": slug.split("_", 1)[0],
                "category": category,
                "dir": lab_dir,
                "emoji": LAB_EMOJI.get(slug, "🧠"),
                "title": _first_title(readme.read_text(encoding="utf-8"), slug),
                "repo_path": f"{base}/{slug}",
            })
    return labs


def anchor_for(doc: str) -> str:
    return doc.replace(".md", "").replace("_", "-")


def rewrite_links_local(body: str) -> str:
    """Reescribe los enlaces para que funcionen desde el archivo local.

    Los documentos del propio laboratorio se convierten en anclas (están todos en
    esta misma página); los saltos a otro laboratorio apuntan a su `index.html`;
    la portada del repositorio apunta al índice offline. El resto de rutas
    relativas —cuadernos, configuraciones, fichas de dataset— se conservan tal
    cual, porque el archivo vive junto a la página.
    """
    docs = {doc for doc, _ in LAB_DOCS}

    def repl(match: re.Match) -> str:
        attr, url = match.group(1), match.group(2)
        if url.startswith(("http://", "https://", "#", "mailto:")):
            return match.group(0)

        target, _, fragment = url.partition("#")
        if not target:
            return match.group(0)

        if target in docs:  # documento del mismo laboratorio → ancla en la página
            return f'{attr}="#{anchor_for(target)}"'

        cross = CROSS_LAB_RE.match(target)
        if cross:  # otro laboratorio → su página HTML
            base, slug, _doc = cross.groups()
            return f'{attr}="../../{base}/{slug}/index.html"'

        if target in ("../../README.md", "../../README.md/"):  # portada → índice offline
            return f'{attr}="../../index.html"'

        return match.group(0)  # notebook.ipynb, configs/…, data/… quedan relativos

    return re.sub(r'(href|src)="([^"]+)"', repl, body)


def render_doc(path: Path) -> str:
    MD.reset()
    text = NAV_RE.sub("\n", path.read_text(encoding="utf-8"))
    return badges_to_chips(rewrite_links_local(MD.convert(text)))


def shell(title: str, description: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} · Neural Network Training Labs</title>
<meta name="description" content="{html.escape(description)}">
<meta name="theme-color" content="#7c5cff">
<style>
{STYLES}
{CHIP_CSS}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def pager(prev: dict | None, nxt: dict | None) -> str:
    def cell(lab: dict | None, direction: str, css: str) -> str:
        if not lab:
            return '<span class="pg pg-empty"></span>'
        href = f'../../{lab["base"]}/{lab["slug"]}/index.html'
        return (f'<a class="pg {css}" href="{href}"><span class="pg-dir">{direction}</span>'
                f'<span class="pg-title">{lab["emoji"]} {html.escape(lab["title"])}</span></a>')

    return (f'<nav class="pager">{cell(prev, "← Anterior", "pg-prev")}'
            f'{cell(nxt, "Siguiente →", "pg-next")}</nav>')


def doc_tabs(lab: dict) -> str:
    links = []
    for doc, heading in LAB_DOCS:
        if not (lab["dir"] / doc).exists():
            continue
        text = heading or "📄 Guía"
        links.append(f'<a class="btn btn-ghost" href="#{anchor_for(doc)}">{text}</a>')
    for notebook, text in (
        ("notebook.ipynb", "📓 Recorrido"),
        ("notebook_student.ipynb", "✏️ Estudiante"),
        ("notebook_solution.ipynb", "✅ Solución"),
    ):
        if (lab["dir"] / notebook).exists():
            links.append(f'<a class="btn btn-ghost" href="{notebook}">{text}</a>')
    return f'<div class="lab-actions">{"".join(links)}</div>'


def lab_page(lab: dict, index: int, total: int, prev: dict | None, nxt: dict | None) -> str:
    sections = []
    for doc, heading in LAB_DOCS:
        path = lab["dir"] / doc
        if not path.exists():
            continue
        rendered = render_doc(path)
        head = (f'<h2 class="doc-sep" id="{anchor_for(doc)}">{heading}</h2>'
                if heading else f'<span id="{anchor_for(doc)}"></span>')
        sections.append(head + rendered)

    body = f"""  <header class="topbar">
    <a class="brand" href="../../index.html"><span aria-hidden="true">🧠</span> Neural Network Training Labs</a>
    <nav class="crumbs"><a href="../../index.html">🏠 Índice</a><span class="sep">›</span>{lab["emoji"]} {html.escape(lab["title"])}</nav>
  </header>
  <main class="prose">
    <div class="lab-hero">
      <div class="lab-kicker">Laboratorio {lab["num"]} · {lab["category"]} · Ruta {index + 1} / {total}</div>
      <h1>{lab["emoji"]} {html.escape(lab["title"])}</h1>
      {doc_tabs(lab)}
    </div>
    {pager(prev, nxt)}
    <article class="content">
      {"".join(sections)}
    </article>
    {pager(prev, nxt)}
  </main>
  <footer class="foot">
    <p><a href="../../index.html">🏠 Índice del recorrido</a> · <a href="{SITE}/labs/{lab['slug']}/index.html">🌐 Sitio de estudio</a> · <a href="{REPO}/tree/main/{lab['repo_path']}">📂 Código en GitHub</a> · <a href="{BLOB}/{lab['repo_path']}/README.md">📄 Markdown</a></p>
    <p>Página autocontenida: se genera desde <code>{lab['repo_path']}</code> con <code>scripts/generate_lab_html.py</code>.</p>
  </footer>"""
    return shell(lab["title"], f'Laboratorio {lab["num"]} — {lab["title"]}', body)


def index_page(labs: list[dict]) -> str:
    def card(lab: dict, position: int) -> str:
        return (f'<a class="lab-card" href="{lab["base"]}/{lab["slug"]}/index.html">'
                f'<span class="lab-num">{lab["emoji"]} {lab["num"]} · ruta {position}</span>'
                f'<span class="lab-name">{html.escape(lab["title"])}</span></a>')

    core = "\n".join(card(lab, i + 1) for i, lab in enumerate(labs) if lab["category"] == "Central")
    adv = "\n".join(card(lab, i + 1) for i, lab in enumerate(labs) if lab["category"] == "Avanzada")
    first = labs[0]

    body = f"""  <header class="topbar">
    <a class="brand" href="index.html"><span aria-hidden="true">🧠</span> Neural Network Training Labs</a>
    <nav class="crumbs"><a href="{SITE}/">Sitio de estudio ↗</a><span class="sep">·</span><a href="{REPO}">Repositorio ↗</a></nav>
  </header>
  <main class="prose">
    <section class="home-hero">
      <div class="home-kicker">v1.0.0 · 31 rutas · 93 notebooks · offline</div>
      <h1>Índice del recorrido<br>de la neurona en NumPy al modelo desplegado</h1>
      <p class="lede">Portada local de las <strong>31 rutas</strong>. Cada una abre su página autocontenida, con teoría, plan de experimentos, evaluación y saltos <strong>anterior / siguiente</strong>. Funciona sin conexión: es la misma fuente que publica el sitio de estudio.</p>
      <div class="home-actions">
        <a class="btn" href="{first['base']}/{first['slug']}/index.html">▶ Empezar por la ruta 1</a>
        <a class="btn btn-ghost" href="README.md">📄 Portada en Markdown</a>
      </div>
    </section>
    <section>
      <h2 class="home-sec">🧪 Laboratorios centrales <span class="count">25</span></h2>
      <div class="lab-grid">{core}</div>
    </section>
    <section>
      <h2 class="home-sec">🚀 Especializaciones avanzadas <span class="count">6</span></h2>
      <div class="lab-grid">{adv}</div>
    </section>
  </main>
  <footer class="foot">
    <p>Semillas separadas · Selección por <code>validation</code> · Sellado de <code>test</code> · Datasets públicos reales</p>
    <p><a href="{REPO}">github.com/vladimiracunadev-create/neural-network-training-labs</a></p>
  </footer>"""
    return shell("Índice del recorrido",
                 "Índice offline de las 31 rutas de Neural Network Training Labs.", body)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="no escribe: falla si alguna página quedó desfasada.")
    args = parser.parse_args()

    labs = collect_labs()
    total = len(labs)
    pages: list[tuple[Path, str]] = [(ROOT / "index.html", index_page(labs))]
    for index, lab in enumerate(labs):
        prev = labs[index - 1] if index > 0 else None
        nxt = labs[index + 1] if index < total - 1 else None
        pages.append((lab["dir"] / "index.html", lab_page(lab, index, total, prev, nxt)))

    stale: list[str] = []
    written = 0
    for path, content in pages:
        current = path.read_text(encoding="utf-8") if path.exists() else None
        if current == content:
            continue
        if args.check:
            stale.append(str(path.relative_to(ROOT)).replace("\\", "/"))
        else:
            path.write_text(content, encoding="utf-8")
            written += 1

    if args.check:
        if stale:
            print("Páginas HTML desfasadas (ejecuta scripts/generate_lab_html.py):")
            for item in stale:
                print(f"  - {item}")
            return 1
        print(f"HTML al día: {total} laboratorios + índice.")
        return 0

    print(f"HTML generado: {written} páginas actualizadas ({total} laboratorios + índice).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
