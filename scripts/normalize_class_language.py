#!/usr/bin/env python3
"""Normaliza la nomenclatura pública del programa sin tocar rutas de archivos."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = [ROOT / "README.md", ROOT / "ROADMAP.md"]
FILES += [ROOT / "scripts" / "build_lab_docs.py", ROOT / "scripts" / "generate_site.py"]
FILES += sorted((ROOT / "docs").glob("*.md"))
FILES += sorted((ROOT / "parts").glob("*.md"))
FILES += sorted((ROOT / "labs").glob("*/README.md"))
FILES += sorted((ROOT / "labs").glob("*/theory.md"))
FILES += sorted((ROOT / "labs").glob("*/experiments.md"))
FILES += sorted((ROOT / "labs").glob("*/assessment.md"))
FILES += sorted((ROOT / "advanced_labs").glob("*/README.md"))
FILES += sorted((ROOT / "advanced_labs").glob("*/theory.md"))
FILES += sorted((ROOT / "advanced_labs").glob("*/experiments.md"))
FILES += sorted((ROOT / "advanced_labs").glob("*/assessment.md"))

PHRASES = {
    "| # | Ruta |": "| # | Clase |",
    "Cada laboratorio publica": "Cada clase publica",
    "Página de la parte": "Página del módulo",
    "Cada parte tiene": "Cada módulo tiene",
    "su posición (`Ruta 4 / 31`), los saltos al laboratorio anterior y siguiente": "su posición (`Clase 04 / 31`), los saltos a la clase anterior y siguiente",
    "Parte 1": "Módulo 1",
    "Parte 2": "Módulo 2",
    "Parte 3": "Módulo 3",
    "Parte 4": "Módulo 4",
    "Parte 5": "Módulo 5",
    "Parte 6": "Módulo 6",
    "Parte 7": "Módulo 7",
    "parte 1": "módulo 1",
    "parte 2": "módulo 2",
    "parte 3": "módulo 3",
    "parte 4": "módulo 4",
    "parte 5": "módulo 5",
    "parte 6": "módulo 6",
    "parte 7": "módulo 7",
    "las siete partes": "los siete módulos",
    "Las siete partes": "Los siete módulos",
    "31 rutas de aprendizaje": "31 clases",
    "31 rutas del programa": "31 clases del programa",
    "31 rutas": "31 clases",
    "Las 31 rutas": "Las 31 clases",
    "las 31 rutas": "las 31 clases",
    "Cada ruta": "Cada clase",
    "cada ruta": "cada clase",
    "rutas anteriores": "clases anteriores",
    "rutas centrales": "clases centrales",
    "propio de esta ruta": "propio de esta clase",
    "Propio de esta ruta": "Propio de esta clase",
    "Ruta de aprendizaje": "Programa de clases",
    "ruta de aprendizaje": "programa de clases",
    "Índice de rutas": "Índice de clases",
    "Laboratorio anterior": "Clase anterior",
    "Laboratorio siguiente": "Clase siguiente",
    "En este laboratorio:": "Material de esta clase:",
}


def shift_reference(match: re.Match[str]) -> str:
    word, raw = match.groups()
    replacement = "Clase" if word[0].isupper() else "clase"
    return f"{replacement} {int(raw) + 1:02d}"


def shift_range(match: re.Match[str]) -> str:
    word, first, last = match.groups()
    replacement = "Clases" if word[0].isupper() else "clases"
    return f"{replacement} {int(first) + 1:02d}–{int(last) + 1:02d}"


def normalize(text: str, *, root_readme: bool) -> str:
    for old, new in PHRASES.items():
        text = text.replace(old, new)
    text = re.sub(r"\b(Rutas|rutas) (\d{2})–(\d{2})\b", shift_range, text)
    text = re.sub(r"\b(Ruta|ruta) (\d{2})\b", shift_reference, text)
    if root_readme:
        def table_number(match: re.Match[str]) -> str:
            body, technical_number = match.groups()
            return f"| {int(technical_number) + 1:02d} | {body}"
        text = re.sub(
            r"(?m)^\| \d{2} \| (.*?\]\((?:labs|advanced_labs)/(\d{2})_[^\n]+)$",
            table_number,
            text,
        )
    return text


def main() -> None:
    changed = 0
    for path in FILES:
        if not path.exists():
            continue
        before = path.read_text(encoding="utf-8")
        after = normalize(before, root_readme=path == ROOT / "README.md")
        if after != before:
            path.write_text(after, encoding="utf-8", newline="\n")
            changed += 1
    print(f"Lenguaje curricular normalizado en {changed} documentos.")


if __name__ == "__main__":
    main()
