#!/usr/bin/env python3
"""Extracción y normalización de citas de fuentes en las clases del programa.

Módulo compartido por `scripts/verify-sources` (offline, bloquea en CI) y
`scripts/refresh-sources` (en red, informativo). No hace ninguna llamada de red:
todo lo que hay aquí es determinista sobre el árbol de ficheros.

Una *clase* es cualquier `theory.md` de una ruta del programa (25 laboratorios
centrales en `labs/` y 6 especializaciones en `advanced_labs/`). Cada clase
termina con un bloque de fuentes; cada viñeta de ese bloque es una *cita*.

Una cita se identifica por una clave estable:

* si la viñeta trae una URL http(s), la clave es la URL normalizada;
* si no, la clave es el título en cursiva normalizado;
* las viñetas que apuntan a documentación interna del repositorio
  (`Consulte ...`) no son citas externas y se contabilizan aparte.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "sources" / "bibliography.json"
CLASS_ROOTS = ("labs", "advanced_labs")

SOURCES_HEADING = re.compile(r"^#{1,4}\s*.*([Ff]uentes|[Rr]eferencias|[Bb]ibliograf)")
BLOCK_END = re.compile(r"^<!--\s*nav-bottom")
BULLET = re.compile(r"^-\s+(.*\S)\s*$")
URL = re.compile(r"https?://[^\s)>\]]+")
ITALIC_TITLE = re.compile(r"\*([^*]+)\*")
INTERNAL_POINTER = re.compile(r"^Consulte\b")

# La glosa (el uso concreto que la clase hace de la fuente) va tras una raya larga.
GLOSS_SEPARATORS = ("—", "–")
MIN_GLOSS_CHARS = 25

VALID_TYPES = ("book", "paper", "standard", "reference", "dataset")
TYPES_WITH_URL_LOCATOR = ("standard", "reference", "dataset")
VALID_STATUS = ("verificada", "pendiente")


def normalize_text(value: str) -> str:
    """Minúsculas, sin acentos ni puntuación, espacios colapsados."""
    decomposed = unicodedata.normalize("NFKD", value)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    lowered = stripped.lower().replace("&", " and ")
    cleaned = re.sub(r"[^a-z0-9]+", " ", lowered)
    return re.sub(r"\s+", " ", cleaned).strip()


def normalize_url(value: str) -> str:
    """Clave estable de una URL: sin esquema, sin `www.`, sin barra final."""
    trimmed = value.strip().rstrip(".,;")
    without_scheme = re.sub(r"^https?://", "", trimmed)
    without_www = re.sub(r"^www\.", "", without_scheme)
    return without_www.rstrip("/")


def isbn13_is_valid(value: str) -> bool:
    digits = re.sub(r"[^0-9]", "", value or "")
    if len(digits) != 13:
        return False
    total = sum(int(d) * (1 if i % 2 == 0 else 3) for i, d in enumerate(digits[:12]))
    return (10 - total % 10) % 10 == int(digits[12])


def doi_is_wellformed(value: str) -> bool:
    return bool(re.fullmatch(r"10\.\d{4,9}/\S+", (value or "").strip()))


@dataclass(frozen=True)
class Citation:
    """Una viñeta del bloque de fuentes de una clase."""

    course_path: str
    line_number: int
    raw: str
    kind: str  # "url" | "work" | "internal"
    key: str
    gloss: str

    @property
    def declares_use(self) -> bool:
        """¿La viñeta dice para qué usa esa fuente *esta* clase?"""
        return len(self.gloss) >= MIN_GLOSS_CHARS


@dataclass
class SourceBlock:
    course_path: str
    heading: str
    citations: list[Citation] = field(default_factory=list)
    fingerprint: str = ""


def _gloss_of(text: str) -> str:
    """Texto tras la última raya larga *aislada*: el uso que la clase declara.

    Solo cuenta la raya rodeada de espacios. Sin esa condición, una raya interna
    como en «exploración–explotación» partiría la glosa por la mitad.
    """
    cut = max(text.rfind(f" {sep} ") for sep in GLOSS_SEPARATORS)
    if cut < 0:
        return ""
    return text[cut + 3 :].strip()


def _citation_from(course_path: str, line_number: int, text: str) -> Citation:
    if INTERNAL_POINTER.match(text):
        return Citation(course_path, line_number, text, "internal", "", "")
    found_url = URL.search(text)
    if found_url:
        key = normalize_url(found_url.group(0))
        return Citation(course_path, line_number, text, "url", key, _gloss_of(text))
    italic = ITALIC_TITLE.search(text)
    raw_key = italic.group(1) if italic else text
    return Citation(
        course_path, line_number, text, "work", normalize_text(raw_key), _gloss_of(text)
    )


def course_files(root: Path = ROOT) -> list[Path]:
    files: list[Path] = []
    for area in CLASS_ROOTS:
        files.extend(sorted((root / area).glob("*/theory.md")))
    return files


def read_block(path: Path, root: Path = ROOT) -> SourceBlock | None:
    course_path = path.relative_to(root).as_posix()
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next((i for i, line in enumerate(lines) if SOURCES_HEADING.match(line)), None)
    if start is None:
        return None
    block = SourceBlock(course_path=course_path, heading=lines[start].strip())
    body: list[str] = []
    for offset, line in enumerate(lines[start + 1 :], start=start + 2):
        if BLOCK_END.match(line):
            break
        body.append(line.strip())
        bullet = BULLET.match(line)
        if bullet:
            block.citations.append(_citation_from(course_path, offset, bullet.group(1)))
    block.fingerprint = normalize_text(" ".join(part for part in body if part))
    return block


def collect_blocks(root: Path = ROOT) -> tuple[list[SourceBlock], list[str]]:
    """Devuelve los bloques encontrados y las rutas de clase que no tienen bloque."""
    blocks: list[SourceBlock] = []
    missing: list[str] = []
    for path in course_files(root):
        block = read_block(path, root)
        if block is None:
            missing.append(path.relative_to(root).as_posix())
        else:
            blocks.append(block)
    return blocks, missing


def load_registry(path: Path = REGISTRY_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def entry_keys(entry: dict) -> set[str]:
    """Todas las claves de cita con las que una entrada puede ser referida."""
    keys: set[str] = set()
    for alias in entry.get("cited_as", []):
        keys.add(normalize_text(alias))
    for url in entry.get("cited_urls", []):
        keys.add(normalize_url(url))
    return keys


def build_key_index(registry: dict) -> dict[str, str]:
    """clave de cita -> id de entrada. Falla ruidosamente si dos entradas chocan."""
    index: dict[str, str] = {}
    for entry in registry.get("entries", []):
        for key in entry_keys(entry):
            if key in index and index[key] != entry["id"]:
                raise ValueError(
                    f"clave de cita duplicada {key!r}: {index[key]!r} y {entry['id']!r}"
                )
            index[key] = entry["id"]
    return index


def external_citations(blocks: list[SourceBlock]) -> list[Citation]:
    return [c for block in blocks for c in block.citations if c.kind != "internal"]


def used_in_from_structure(
    blocks: list[SourceBlock], index: dict[str, str]
) -> dict[str, list[str]]:
    """`used_in` derivado del árbol de clases, no escrito a mano."""
    usage: dict[str, set[str]] = {}
    for citation in external_citations(blocks):
        entry_id = index.get(citation.key)
        if entry_id is not None:
            usage.setdefault(entry_id, set()).add(citation.course_path)
    return {entry_id: sorted(paths) for entry_id, paths in usage.items()}


def counts(blocks: list[SourceBlock], registry: dict, index: dict[str, str]) -> dict:
    """Las cifras que el README publica. Se calculan, no se escriben."""
    external = external_citations(blocks)
    distinct = {c.key for c in external}
    covered = {key for key in distinct if key in index}
    entries = registry.get("entries", [])
    by_type: dict[str, int] = {}
    for entry in entries:
        by_type[entry["type"]] = by_type.get(entry["type"], 0) + 1
    datasets = [e for e in entries if e["type"] == "dataset"]
    return {
        "courses": len(blocks),
        "citations": len(external),
        "citations_internal": sum(
            1 for b in blocks for c in b.citations if c.kind == "internal"
        ),
        "distinct_works": len(distinct),
        "distinct_urls": len({c.key for c in external if c.kind == "url"}),
        "distinct_titles": len({c.key for c in external if c.kind == "work"}),
        "covered_works": len(covered),
        "coverage_pct": round(100.0 * len(covered) / len(distinct), 1) if distinct else 0.0,
        "entries": len(entries),
        "entries_by_type": by_type,
        "entries_verified": sum(1 for e in entries if e.get("status") == "verificada"),
        "entries_pending": sum(1 for e in entries if e.get("status") == "pendiente"),
        "datasets": len(datasets),
        "datasets_with_checksum": sum(
            1 for d in datasets if (d.get("integrity") or {}).get("artifacts")
        ),
        "dataset_artifacts_hashed": sum(
            len((d.get("integrity") or {}).get("artifacts") or []) for d in datasets
        ),
        "datasets_with_license": sum(1 for d in datasets if d.get("license")),
    }
