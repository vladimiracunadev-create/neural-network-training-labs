"""Genera el sitio estático de GitHub Pages para Neural Network Training Labs.

Recorre `labs/` (25 laboratorios centrales) y `advanced_labs/` (6 especializaciones)
y renderiza, por cada laboratorio, su `README.md`, `theory.md`, `experiments.md` y
`assessment.md` en una sola página HTML. Produce:

    site/index.html                      — portada con las 31 rutas
    site/labs/<NN_slug>/index.html       — página del laboratorio con navegación
    site/styles.css                      — paleta y layout
    site/.nojekyll                       — evita el procesado Jekyll de Pages

Cada página de laboratorio incluye un paginador **anterior / siguiente** que crea
un flujo lineal de estudio: de la neurona en NumPy (00) a SimCLR (30). Es la única
fuente de verdad: el mismo Markdown del repositorio. Se vuelve a ejecutar en CI
antes de publicar el artefacto de Pages.
"""
from __future__ import annotations

import html
import re
import shutil
from pathlib import Path

import markdown  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "site"
REPO = "https://github.com/vladimiracunadev-create/neural-network-training-labs"
BLOB = f"{REPO}/blob/main"

# Documentos de cada laboratorio que se renderizan, en orden, dentro de la página.
LAB_DOCS = [
    ("README.md", None),
    ("theory.md", "🧠 Teoría"),
    ("experiments.md", "🔬 Experimentos"),
    ("assessment.md", "📝 Evaluación"),
]

# Emoji identificador por dominio, para la portada y el encabezado.
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

MD = markdown.Markdown(
    extensions=["extra", "toc", "sane_lists", "tables", "fenced_code", "admonition"],
    output_format="html5",
)


# ──────────────────────────────────────────────────────────────────────────────
# Recolección de laboratorios
# ──────────────────────────────────────────────────────────────────────────────

def _first_title(md_text: str, fallback: str) -> str:
    for line in md_text.splitlines():
        m = re.match(r"^#\s+(.*)", line.strip())
        if m:
            return m.group(1).strip()
    return fallback


def collect_labs() -> list[dict]:
    labs: list[dict] = []
    for base, category in (("labs", "Central"), ("advanced_labs", "Avanzada")):
        base_dir = ROOT / base
        if not base_dir.exists():
            continue
        for lab_dir in sorted(p for p in base_dir.iterdir() if p.is_dir()):
            readme = lab_dir / "README.md"
            if not readme.exists():
                continue
            slug = lab_dir.name
            num = slug.split("_", 1)[0]
            title = _first_title(readme.read_text(encoding="utf-8"), slug)
            labs.append({
                "slug": slug,
                "num": num,
                "title": title,
                "category": category,
                "emoji": LAB_EMOJI.get(slug, "🧠"),
                "src_dir": lab_dir,
                "repo_path": f"{base}/{slug}",
                "out_dir": OUT / "labs" / slug,
            })
    return labs


# ──────────────────────────────────────────────────────────────────────────────
# Render de Markdown → HTML (con reescritura de enlaces relativos)
# ──────────────────────────────────────────────────────────────────────────────

def rewrite_links(html_body: str, repo_path: str) -> str:
    """Los enlaces relativos apuntan a archivos del repo (notebooks, configs, docs).

    Desde Pages no existen, así que se reescriben hacia GitHub. Los anclas
    internas (#...) y las URLs absolutas se dejan intactas.
    """
    def repl(m: re.Match) -> str:
        attr, url = m.group(1), m.group(2)
        if url.startswith(("http://", "https://", "#", "mailto:")):
            return m.group(0)
        clean = url.lstrip("./")
        # Enlaces a otro documento del mismo laboratorio ya renderizado en la página:
        for doc, _ in LAB_DOCS:
            if clean == doc:
                anchor = doc.replace(".md", "").replace("_", "-")
                return f'{attr}="#{anchor}"'
        # Resto: hacia el blob de GitHub, resolviendo rutas ../
        if clean.startswith("../"):
            target = f"{BLOB}/{repo_path}/{clean}"
        else:
            target = f"{BLOB}/{repo_path}/{clean}"
        return f'{attr}="{target}"'

    return re.sub(r'(href|src)="([^"]+)"', repl, html_body)


def render_doc(path: Path, repo_path: str) -> str:
    MD.reset()
    body = MD.convert(path.read_text(encoding="utf-8"))
    return rewrite_links(body, repo_path)


# ──────────────────────────────────────────────────────────────────────────────
# Plantillas HTML
# ──────────────────────────────────────────────────────────────────────────────

def page_shell(title: str, description: str, root: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)} · Neural Network Training Labs</title>
  <meta name="description" content="{html.escape(description)}">
  <meta name="theme-color" content="#7c5cff">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="{root}styles.css">
</head>
<body>
{body}
</body>
</html>
"""


def breadcrumbs(root: str, extra: str | None = None) -> str:
    trail = f'<a href="{root}index.html">🧠 Inicio</a>'
    if extra:
        trail += f'<span class="sep">›</span>{extra}'
    return f'<nav class="crumbs">{trail}</nav>'


def pager(root: str, prev: dict | None, nxt: dict | None) -> str:
    left = (
        f'<a class="pg pg-prev" href="{root}labs/{prev["slug"]}/index.html">'
        f'<span class="pg-dir">← Anterior</span>'
        f'<span class="pg-title">{prev["emoji"]} {html.escape(prev["title"])}</span></a>'
        if prev else '<span class="pg pg-empty"></span>'
    )
    right = (
        f'<a class="pg pg-next" href="{root}labs/{nxt["slug"]}/index.html">'
        f'<span class="pg-dir">Siguiente →</span>'
        f'<span class="pg-title">{nxt["emoji"]} {html.escape(nxt["title"])}</span></a>'
        if nxt else '<span class="pg pg-empty"></span>'
    )
    return f'<nav class="pager">{left}{right}</nav>'


def lab_page(lab: dict, idx: int, total: int, prev: dict | None, nxt: dict | None) -> str:
    root = "../../"
    sections: list[str] = []
    for doc, heading in LAB_DOCS:
        path = lab["src_dir"] / doc
        if not path.exists():
            continue
        anchor = doc.replace(".md", "").replace("_", "-")
        rendered = render_doc(path, lab["repo_path"])
        head = f'<h2 class="doc-sep" id="{anchor}">{heading}</h2>' if heading else f'<span id="{anchor}"></span>'
        sections.append(head + rendered)

    crumb = breadcrumbs(root, f'{lab["emoji"]} {html.escape(lab["title"])}')
    hero = f"""
  <div class="lab-hero">
    <div class="lab-kicker">Laboratorio {lab["num"]} · {lab["category"]} · {idx + 1} / {total}</div>
    <h1>{lab["emoji"]} {html.escape(lab["title"])}</h1>
    <div class="lab-actions">
      <a class="btn" href="{BLOB}/{lab["repo_path"]}/notebook.ipynb">📓 Notebook</a>
      <a class="btn btn-ghost" href="{REPO}/tree/main/{lab["repo_path"]}">📂 Código en GitHub</a>
    </div>
  </div>"""

    body = f"""  <header class="topbar">
    <a class="brand" href="{root}index.html"><span aria-hidden="true">🧠</span> Neural Network Training Labs</a>
    {crumb}
  </header>
  <main class="prose">
    {hero}
    {pager(root, prev, nxt)}
    <article class="content">
      {"".join(sections)}
    </article>
    {pager(root, prev, nxt)}
  </main>
  <footer class="foot">
    <p>Fuente única: <a href="{REPO}/tree/main/{lab['repo_path']}"><code>{lab['repo_path']}</code></a> · <a href="{root}index.html">← Volver a la portada</a></p>
  </footer>"""
    return page_shell(lab["title"], f'Laboratorio {lab["num"]} — {lab["title"]}', root, body)


def index_page(labs: list[dict]) -> str:
    core = [l for l in labs if l["category"] == "Central"]
    adv = [l for l in labs if l["category"] == "Avanzada"]

    def card(lab: dict) -> str:
        return (
            f'<a class="lab-card" href="labs/{lab["slug"]}/index.html">'
            f'<span class="lab-num">{lab["emoji"]} {lab["num"]}</span>'
            f'<span class="lab-name">{html.escape(lab["title"])}</span></a>'
        )

    core_cards = "\n".join(card(l) for l in core)
    adv_cards = "\n".join(card(l) for l in adv)
    first = labs[0]

    body = f"""  <header class="topbar">
    <a class="brand" href="index.html"><span aria-hidden="true">🧠</span> Neural Network Training Labs</a>
    <nav class="crumbs"><a href="{REPO}">Repositorio ↗</a></nav>
  </header>
  <main class="prose">
    <section class="home-hero">
      <div class="home-kicker">v1.0.0 · 31 rutas · 93 notebooks</div>
      <h1>Aprende, entrena y despliega redes neuronales<br>con datasets públicos reales</h1>
      <p class="lede">Un recorrido lineal de la <strong>neurona en NumPy</strong> a <strong>difusión, transformers y aprendizaje autosupervisado</strong>. Cada laboratorio ancla su teoría en libros de referencia, separa las semillas de datos y entrenamiento, y sella el <code>test</code> antes de evaluar.</p>
      <div class="home-actions">
        <a class="btn" href="labs/{first['slug']}/index.html">▶ Empezar por el Laboratorio 00</a>
        <a class="btn btn-ghost" href="{REPO}">📂 Ver el repositorio</a>
      </div>
    </section>

    <section>
      <h2 class="home-sec">🧪 Laboratorios centrales <span class="count">25</span></h2>
      <div class="lab-grid">{core_cards}</div>
    </section>

    <section>
      <h2 class="home-sec">🚀 Especializaciones avanzadas <span class="count">6</span></h2>
      <div class="lab-grid">{adv_cards}</div>
    </section>
  </main>
  <footer class="foot">
    <p>Contenido anclado en libros de referencia · Semillas separadas · Sellado de <code>test</code> · Cadena de suministro verificable</p>
    <p><a href="{REPO}">github.com/vladimiracunadev-create/neural-network-training-labs</a></p>
  </footer>"""
    return page_shell("Portada", "31 rutas para aprender, entrenar y desplegar redes neuronales con datasets públicos reales.", "", body)


STYLES = """/* Neural Network Training Labs — sitio de estudio */
:root {
  --bg: #0d1117; --bg-soft: #131a24; --card: #161f2e; --line: #263041;
  --ink: #e6edf3; --ink-soft: #9fb0c3; --muted: #6b7d92;
  --accent: #7c5cff; --accent-soft: #9d86ff; --green: #2e8b57; --gold: #f0b429;
  --radius: 14px; --radius-sm: 9px;
  --shadow: 0 2px 10px rgba(0,0,0,0.35); --shadow-h: 0 8px 26px rgba(124,92,255,0.28);
}
@media (prefers-color-scheme: light) {
  :root {
    --bg: #f7f8fb; --bg-soft: #eef1f7; --card: #ffffff; --line: #e2e6ee;
    --ink: #1a2233; --ink-soft: #48566b; --muted: #7a889c;
    --shadow: 0 2px 10px rgba(20,30,50,0.08); --shadow-h: 0 10px 28px rgba(124,92,255,0.20);
  }
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body { margin: 0; background: var(--bg); color: var(--ink);
  font-family: 'Inter', system-ui, sans-serif; line-height: 1.65; }

.topbar { display: flex; align-items: center; justify-content: space-between; gap: 16px;
  padding: 14px 26px; background: color-mix(in srgb, var(--bg-soft) 88%, transparent);
  border-bottom: 1px solid var(--line); position: sticky; top: 0; z-index: 50; backdrop-filter: blur(10px); }
.brand { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.02rem;
  color: var(--ink); text-decoration: none; }
.crumbs { font-size: 0.9rem; color: var(--ink-soft); }
.crumbs a { color: var(--accent-soft); text-decoration: none; }
.crumbs a:hover { text-decoration: underline; }
.crumbs .sep { margin: 0 8px; opacity: 0.4; }

.prose { max-width: 900px; margin: 0 auto; padding: 34px 26px 72px; }

/* Portada */
.home-hero { text-align: center; padding: 30px 0 14px; }
.home-kicker { display: inline-block; font-family: 'JetBrains Mono', monospace; font-size: 0.82rem;
  color: var(--accent-soft); background: color-mix(in srgb, var(--accent) 14%, transparent);
  border: 1px solid color-mix(in srgb, var(--accent) 34%, transparent);
  padding: 5px 14px; border-radius: 999px; letter-spacing: 0.02em; }
.home-hero h1 { font-family: 'Space Grotesk', sans-serif; font-size: clamp(1.9rem, 4.6vw, 2.9rem);
  line-height: 1.14; margin: 18px 0 10px; letter-spacing: -0.02em;
  background: linear-gradient(120deg, var(--ink), var(--accent-soft));
  -webkit-background-clip: text; background-clip: text; color: transparent; }
.lede { max-width: 680px; margin: 0 auto; color: var(--ink-soft); font-size: 1.08rem; }
.home-actions, .lab-actions { display: flex; gap: 12px; flex-wrap: wrap; justify-content: center; margin: 24px 0 8px; }
.btn { display: inline-flex; align-items: center; gap: 8px; padding: 11px 20px; border-radius: 999px;
  background: var(--accent); color: #fff; font-weight: 600; text-decoration: none; font-size: 0.95rem;
  box-shadow: var(--shadow); transition: transform 0.15s, box-shadow 0.15s; }
.btn:hover { transform: translateY(-2px); box-shadow: var(--shadow-h); }
.btn-ghost { background: transparent; color: var(--ink); border: 1px solid var(--line); box-shadow: none; }
.btn-ghost:hover { border-color: var(--accent); }

.home-sec { font-family: 'Space Grotesk', sans-serif; font-size: 1.35rem; margin: 44px 0 18px;
  display: flex; align-items: center; gap: 12px; }
.home-sec .count { font-size: 0.8rem; font-weight: 600; color: var(--accent-soft);
  background: color-mix(in srgb, var(--accent) 15%, transparent); padding: 2px 11px; border-radius: 999px; }
.lab-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 14px; }
.lab-card { display: flex; flex-direction: column; gap: 6px; padding: 16px 18px; background: var(--card);
  border: 1px solid var(--line); border-radius: var(--radius); text-decoration: none; color: var(--ink);
  transition: transform 0.15s, border-color 0.15s, box-shadow 0.15s; box-shadow: var(--shadow); }
.lab-card:hover { transform: translateY(-3px); border-color: var(--accent); box-shadow: var(--shadow-h); }
.lab-num { font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: var(--accent-soft); font-weight: 600; }
.lab-name { font-weight: 600; font-size: 1.0rem; line-height: 1.3; }

/* Página de laboratorio */
.lab-hero { padding: 8px 0 6px; }
.lab-kicker { font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: var(--accent-soft);
  text-transform: uppercase; letter-spacing: 0.05em; }
.lab-hero h1 { font-family: 'Space Grotesk', sans-serif; font-size: clamp(1.7rem, 3.8vw, 2.4rem);
  line-height: 1.16; margin: 8px 0 16px; letter-spacing: -0.01em; }
.lab-actions { justify-content: flex-start; margin: 0 0 6px; }

/* Paginador anterior / siguiente */
.pager { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin: 22px 0; }
.pg { display: flex; flex-direction: column; gap: 4px; padding: 14px 18px; border-radius: var(--radius);
  background: var(--card); border: 1px solid var(--line); text-decoration: none; color: var(--ink);
  transition: transform 0.15s, border-color 0.15s, box-shadow 0.15s; box-shadow: var(--shadow); }
.pg-next { text-align: right; }
.pg:hover { border-color: var(--accent); transform: translateY(-2px); box-shadow: var(--shadow-h); }
.pg-empty { background: transparent; border: none; box-shadow: none; }
.pg-dir { font-size: 0.8rem; color: var(--accent-soft); font-weight: 600; }
.pg-title { font-weight: 600; font-size: 0.98rem; }

/* Contenido renderizado */
.content { margin-top: 8px; }
.content h1 { display: none; } /* el título ya está en el hero */
.doc-sep { font-family: 'Space Grotesk', sans-serif; margin: 2.4em 0 0.6em; padding-top: 1.2em;
  border-top: 2px solid var(--line); font-size: 1.5rem; color: var(--accent-soft); }
.content h2 { font-family: 'Space Grotesk', sans-serif; font-size: 1.4rem; margin: 1.8em 0 0.5em;
  padding-bottom: 6px; border-bottom: 1px solid var(--line); }
.content h3 { font-size: 1.14rem; margin: 1.5em 0 0.4em; }
.content p, .content li { color: var(--ink-soft); }
.content a { color: var(--accent-soft); text-decoration: underline; text-decoration-color: color-mix(in srgb, var(--accent) 45%, transparent); text-underline-offset: 3px; }
.content a:hover { text-decoration-color: var(--accent); }
.content code { font-family: 'JetBrains Mono', monospace; font-size: 0.88em;
  background: var(--bg-soft); border: 1px solid var(--line); padding: 1px 6px; border-radius: 5px; color: var(--ink); }
.content pre { background: #0b1220; color: #d7e0ee; padding: 16px 18px; border-radius: var(--radius-sm);
  overflow-x: auto; border: 1px solid var(--line); box-shadow: var(--shadow); }
.content pre code { background: transparent; border: none; padding: 0; color: inherit; }
.content blockquote { border-left: 4px solid var(--accent); background: var(--bg-soft);
  padding: 12px 18px; margin: 1.4em 0; border-radius: 0 var(--radius-sm) var(--radius-sm) 0; color: var(--ink); }
.content table { border-collapse: collapse; width: 100%; margin: 1.4em 0; background: var(--card);
  border-radius: var(--radius-sm); overflow: hidden; box-shadow: var(--shadow); display: block; overflow-x: auto; }
.content th, .content td { padding: 10px 14px; text-align: left; border-bottom: 1px solid var(--line); }
.content th { background: color-mix(in srgb, var(--accent) 20%, var(--card)); color: var(--ink); font-weight: 600; }
.content tr:last-child td { border-bottom: none; }
.content img { max-width: 100%; height: auto; border-radius: var(--radius-sm); }
.content hr { border: none; border-top: 1px solid var(--line); margin: 2.2em 0; }

.foot { max-width: 900px; margin: 0 auto; padding: 26px; border-top: 1px solid var(--line);
  color: var(--muted); font-size: 0.88rem; text-align: center; }
.foot a { color: var(--accent-soft); text-decoration: none; }
.foot code { font-family: 'JetBrains Mono', monospace; }

@media (max-width: 620px) {
  .pager { grid-template-columns: 1fr; }
  .pg-next { text-align: left; }
}
"""


def main() -> None:
    labs = collect_labs()
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    (OUT / ".nojekyll").write_text("", encoding="utf-8")
    (OUT / "styles.css").write_text(STYLES, encoding="utf-8")
    (OUT / "index.html").write_text(index_page(labs), encoding="utf-8")

    total = len(labs)
    for i, lab in enumerate(labs):
        prev = labs[i - 1] if i > 0 else None
        nxt = labs[i + 1] if i < total - 1 else None
        lab["out_dir"].mkdir(parents=True, exist_ok=True)
        (lab["out_dir"] / "index.html").write_text(lab_page(lab, i, total, prev, nxt), encoding="utf-8")

    print(f"Sitio generado: {total} laboratorios + portada en {OUT}")


if __name__ == "__main__":
    main()
