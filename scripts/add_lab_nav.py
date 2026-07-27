"""Inserta (o actualiza) la navegación anterior/siguiente en el README de cada laboratorio.

Crea un flujo lineal de estudio 00 → 30 directamente en el Markdown del repositorio:
una línea de navegación tras el título y un bloque de navegación al final de cada
`README.md`. Es idempotente: delimita el contenido con marcadores HTML
(`<!-- nav-top -->` / `<!-- nav-bottom -->`) y los reemplaza en cada ejecución, de
modo que puede volver a correrse cuando cambie el orden o los títulos.

El sitio de GitHub Pages (scripts/generate_site.py) elimina estos bloques al
renderizar, porque allí la navegación la aporta su propio paginador.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = "https://vladimiracunadev-create.github.io/neural-network-training-labs"

# Debe coincidir con el mapa de scripts/generate_site.py.
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

TOP_RE = re.compile(r"\n?<!-- nav-top -->.*?<!-- /nav-top -->\n?", re.DOTALL)
BOTTOM_RE = re.compile(r"\n?<!-- nav-bottom -->.*?<!-- /nav-bottom -->\n?", re.DOTALL)


def _first_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        m = re.match(r"^#\s+(.*)", line.strip())
        if m:
            return m.group(1).strip()
    return fallback


def collect_labs() -> list[dict]:
    labs: list[dict] = []
    for base in ("labs", "advanced_labs"):
        base_dir = ROOT / base
        if not base_dir.exists():
            continue
        for lab_dir in sorted(p for p in base_dir.iterdir() if p.is_dir()):
            readme = lab_dir / "README.md"
            if not readme.exists():
                continue
            slug = lab_dir.name
            labs.append({
                "slug": slug,
                "base": base,
                "title": _first_title(readme.read_text(encoding="utf-8"), slug),
                "emoji": LAB_EMOJI.get(slug, "🧠"),
                "readme": readme,
            })
    return labs


def link(lab: dict) -> str:
    # Ambos directorios cuelgan de la raíz; ../../<base>/<slug> resuelve para cualquier par.
    return f'../../{lab["base"]}/{lab["slug"]}/README.md'


def label(lab: dict) -> str:
    return f'{lab["emoji"]} {lab["title"]}'


def top_block(prev: dict | None, nxt: dict | None) -> str:
    parts = []
    if prev:
        parts.append(f'[⬅️ Anterior]({link(prev)})')
    parts.append('[🏠 Índice](../../README.md#laboratorios)')
    if nxt:
        parts.append(f'[Siguiente ➡️]({link(nxt)})')
    line = " · ".join(parts)
    return f"<!-- nav-top -->\n> 🧭 {line}\n<!-- /nav-top -->"


def bottom_block(lab: dict, prev: dict | None, nxt: dict | None) -> str:
    prev_cell = f'[{label(prev)}]({link(prev)})' if prev else '— (inicio del recorrido)'
    next_cell = f'[{label(nxt)}]({link(nxt)})' if nxt else '— (fin del recorrido)'
    site_url = f'{SITE}/labs/{lab["slug"]}/index.html'
    return (
        "<!-- nav-bottom -->\n"
        "## 🧭 Navegación del curso\n\n"
        "| ⬅️ Anterior | Siguiente ➡️ |\n"
        "|---|---|\n"
        f"| {prev_cell} | {next_cell} |\n\n"
        f"[🏠 Portada del repositorio](../../README.md) · "
        f"[🌐 Ver en el sitio de estudio]({site_url})\n"
        "<!-- /nav-bottom -->"
    )


def apply_nav(lab: dict, prev: dict | None, nxt: dict | None) -> bool:
    text = lab["readme"].read_text(encoding="utf-8")
    original = text

    # Quitar bloques previos para reinsertarlos actualizados.
    text = TOP_RE.sub("", text)
    text = BOTTOM_RE.sub("", text)
    text = text.rstrip() + "\n"

    lines = text.splitlines()
    # Insertar el bloque superior justo después del primer encabezado H1.
    out: list[str] = []
    inserted_top = False
    for line in lines:
        out.append(line)
        if not inserted_top and re.match(r"^#\s+", line):
            out.append("")
            out.append(top_block(prev, nxt))
            inserted_top = True
    if not inserted_top:  # sin H1: al principio
        out.insert(0, top_block(prev, nxt))

    body = "\n".join(out).rstrip() + "\n\n"
    body += bottom_block(lab, prev, nxt) + "\n"

    if body != original:
        lab["readme"].write_text(body, encoding="utf-8")
        return True
    return False


def main() -> None:
    labs = collect_labs()
    total = len(labs)
    changed = 0
    for i, lab in enumerate(labs):
        prev = labs[i - 1] if i > 0 else None
        nxt = labs[i + 1] if i < total - 1 else None
        if apply_nav(lab, prev, nxt):
            changed += 1
    print(f"Navegación aplicada a {changed}/{total} laboratorios.")


if __name__ == "__main__":
    main()
