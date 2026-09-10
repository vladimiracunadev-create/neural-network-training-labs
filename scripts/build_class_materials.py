#!/usr/bin/env python3
"""Sincroniza el contrato pedagógico y los recursos propios de las 31 clases."""
from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PROFILES_PATH = ROOT / "configs" / "classes.yaml"

MODULES = [
    (1, 3, 1, "Fundamentos", "#22c55e"),
    (4, 8, 2, "Arquitecturas", "#3b82f6"),
    (9, 13, 3, "Familias especializadas", "#a855f7"),
    (14, 16, 4, "Entrenamiento eficiente", "#f59e0b"),
    (17, 21, 5, "Mecánica fina", "#ef4444"),
    (22, 25, 6, "Confianza y despliegue", "#64748b"),
    (26, 31, 7, "Especializaciones avanzadas", "#06b6d4"),
]


def load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def module_for(number: int) -> tuple[int, str, str]:
    for first, last, module, title, color in MODULES:
        if first <= number <= last:
            return module, title, color
    raise ValueError(f"clase fuera del programa: {number}")


def folder_for(class_id: str) -> Path:
    base = "advanced_labs" if int(class_id.split("_", 1)[0]) >= 25 else "labs"
    return ROOT / base / class_id


def lesson_text(profile: dict, current: dict) -> str:
    number = int(profile["class_number"])
    module, module_title, _ = module_for(number)
    normalized = dict(current)
    normalized["schema_version"] = "3.0"
    normalized["id"] = profile["id"]
    normalized.pop("lab", None)
    normalized["class_number"] = number
    normalized["module_number"] = module
    normalized["module_title"] = module_title
    normalized["essential_question"] = profile["essential_question"]
    normalized["opening_hook"] = profile["hook"]
    normalized["class_practice"] = profile["practice"]
    normalized["visual_evidence"] = {
        "title": profile["visual_title"],
        "asset": "assets/class-map.svg",
        "steps": profile["visual_steps"],
    }
    normalized["common_misconception"] = profile["misconception"]
    normalized.setdefault("estimated_hours", 8 if number >= 9 else 6)
    normalized.setdefault("success_criteria", [
        "responde la pregunta esencial con evidencia de la ejecución",
        "interpreta la visualización propia de la clase",
        "distingue resultados observados de supuestos",
    ])
    return yaml.safe_dump(normalized, allow_unicode=True, sort_keys=False, width=1000)


def svg_text(profile: dict) -> str:
    number = int(profile["class_number"])
    module, module_title, color = module_for(number)
    steps = [html.escape(str(value)) for value in profile["visual_steps"]]
    title = html.escape(str(profile["visual_title"]))
    question = html.escape(str(profile["essential_question"]))
    boxes = []
    centers = [190, 600, 1010]
    for index, (x, label) in enumerate(zip(centers, steps), start=1):
        boxes.append(
            f'<rect x="{x - 150}" y="250" width="300" height="126" rx="22" fill="#111827" '
            f'stroke="{color}" stroke-width="3"/>'
            f'<circle cx="{x - 112}" cy="288" r="18" fill="{color}"/>'
            f'<text x="{x - 112}" y="295" text-anchor="middle" class="stepnum">{index}</text>'
            f'<text x="{x}" y="335" text-anchor="middle" class="step">{label}</text>'
        )
    arrows = (
        f'<path d="M350 313 H432" stroke="{color}" stroke-width="5" marker-end="url(#arrow)"/>'
        f'<path d="M760 313 H842" stroke="{color}" stroke-width="5" marker-end="url(#arrow)"/>'
    )
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="500" viewBox="0 0 1200 500" role="img" aria-labelledby="title desc">
<title id="title">Clase {number:02d}: {title}</title>
<desc id="desc">Mapa conceptual en tres pasos: {", ".join(steps)}.</desc>
<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="{color}"/></marker></defs>
<style>.kicker{{font:600 22px system-ui;fill:{color}}}.title{{font:700 38px system-ui;fill:#f8fafc}}.question{{font:400 21px system-ui;fill:#cbd5e1}}.step{{font:600 19px system-ui;fill:#f8fafc}}.stepnum{{font:700 16px system-ui;fill:#07111f}}</style>
<rect width="1200" height="500" rx="30" fill="#07111f"/>
<text x="60" y="62" class="kicker">MÓDULO {module:02d} · CLASE {number:02d} · {html.escape(module_title).upper()}</text>
<text x="60" y="120" class="title">{title}</text>
<text x="60" y="170" class="question">{question}</text>
{arrows}{''.join(boxes)}
<text x="600" y="445" text-anchor="middle" class="question">Observar → explicar → comprobar con evidencia</text>
</svg>
'''


def instructor_text(profile: dict, lesson: dict) -> str:
    number = int(profile["class_number"])
    module, module_title, _ = module_for(number)
    title = str(lesson.get("title") or profile["id"])
    outcomes = lesson.get("learning_outcomes") or []
    outcomes_md = "\n".join(f"- {item}" for item in outcomes)
    return f"""# Guía docente — Clase {number:02d}: {title}

> **Módulo {module}: {module_title}** · Esta guía acompaña a quien facilita la clase; no es un guion que deba leerse literalmente.

## La conversación que abre la clase

{profile['hook']}

Plantee primero esta pregunta y permita que el grupo formule una predicción antes de mostrar código:

> **{profile['essential_question']}**

## Qué debe quedar comprendido

{outcomes_md}

## Secuencia sugerida

| Momento | Duración | Acción docente | Evidencia del estudiante |
|---|---:|---|---|
| Activación | 10 min | Presentar el caso inicial y recoger predicciones. | Explica qué espera observar y por qué. |
| Construcción | 25 min | Conectar intuición, representación y matemática. | Dibuja o relata el mecanismo con sus propias palabras. |
| Demostración | 25 min | Recorrer el mapa visual y ejecutar el ejemplo mínimo. | Interpreta cada transición sin limitarse a describir la figura. |
| Investigación | 35 min | Guiar la práctica propia de esta clase. | Contrasta su predicción con una medida o gráfica. |
| Cierre | 15 min | Volver a la pregunta esencial y discutir límites. | Entrega una conclusión breve con evidencia y una duda abierta. |

## Práctica propia de esta materia

{profile['practice']}

![{profile['visual_title']}](assets/class-map.svg)

La figura es un punto de entrada. Pida al estudiante que explique qué cambia entre los tres pasos y qué resultado observable invalidaría su explicación.

## Dificultad conceptual que conviene provocar

> **Idea engañosa:** {profile['misconception']}

No la corrija inmediatamente. Solicite un ejemplo o contraejemplo, ejecute una comparación y recién entonces reconstruya la explicación con el grupo.

## Preguntas para acompañar sin resolver

- ¿Qué observación concreta respalda esa afirmación?
- ¿Qué variable está cambiando y cuáles permanecen controladas?
- ¿La gráfica muestra el mecanismo o solamente una correlación?
- ¿En qué caso dejaría de ser válida esta conclusión?

## Cierre verificable

La clase termina cuando el estudiante puede responder la pregunta esencial, interpretar su visualización y señalar al menos una limitación. Una métrica aislada no reemplaza ninguna de esas tres evidencias.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    profiles = load_yaml(PROFILES_PATH).get("classes") or []
    if len(profiles) != 31:
        raise SystemExit(f"se esperaban 31 perfiles de clase y hay {len(profiles)}")
    stale: list[str] = []
    for profile in profiles:
        folder = folder_for(profile["id"])
        lesson_path = folder / "lesson.yaml"
        current = load_yaml(lesson_path)
        outputs = {
            lesson_path: lesson_text(profile, current),
            folder / "assets" / "class-map.svg": svg_text(profile),
            folder / "instructor-guide.md": instructor_text(profile, current),
        }
        for path, expected in outputs.items():
            actual = path.read_text(encoding="utf-8") if path.exists() else None
            if actual == expected:
                continue
            if args.check:
                stale.append(str(path.relative_to(ROOT)))
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(expected, encoding="utf-8", newline="\n")
    if stale:
        print("Material de clases desfasado:")
        print("\n".join(f"  - {path}" for path in stale))
        return 1
    print(f"Material pedagógico al día: {len(profiles)} clases.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
