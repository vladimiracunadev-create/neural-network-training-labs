"""Contratos del registro de fuentes: los que se pueden comprobar sin red."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import sources_lib as sources  # noqa: E402

REGISTRY = json.loads((ROOT / "sources" / "bibliography.json").read_text(encoding="utf-8"))
ENTRIES = REGISTRY["entries"]


def test_normalize_text_ignora_acentos_puntuacion_y_ampersand() -> None:
    assert sources.normalize_text("Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow") == (
        "hands on machine learning with scikit learn keras and tensorflow"
    )
    assert sources.normalize_text("Géron") == "geron"


def test_normalize_url_quita_esquema_www_y_barra_final() -> None:
    assert sources.normalize_url("https://www.robots.ox.ac.uk/~vgg/data/pets/") == "robots.ox.ac.uk/~vgg/data/pets"
    assert sources.normalize_url("http://onnx.ai") == "onnx.ai"


@pytest.mark.parametrize(
    ("isbn", "valid"),
    [
        ("9780262035613", True),
        ("9798411463330", True),
        ("9780262035614", False),  # dígito de control alterado
        ("978026203561", False),  # longitud incorrecta
        ("", False),
    ],
)
def test_isbn13_comprueba_el_digito_de_control(isbn: str, valid: bool) -> None:
    assert sources.isbn13_is_valid(isbn) is valid


@pytest.mark.parametrize(
    ("doi", "wellformed"),
    [("10.1038/323533a0", True), ("10.48550/arxiv.1706.03762", True), ("doi:10.1038/x", False), ("", False)],
)
def test_doi_bien_formado(doi: str, wellformed: bool) -> None:
    assert sources.doi_is_wellformed(doi) is wellformed


def test_la_glosa_no_se_parte_en_una_raya_interna() -> None:
    """«exploración–explotación» no debe cortar el uso que declara la clase."""
    citation = sources._citation_from(
        "labs/10_dqn_reinforcement/theory.md",
        1,
        "Sutton & Barto — *Reinforcement Learning: An Introduction* (2.ª ed., MIT Press) — "
        "texto canónico: Bellman, Q-learning y equilibrio exploración–explotación.",
    )
    assert citation.gloss.startswith("texto canónico")
    assert citation.declares_use


def test_toda_ruta_del_programa_tiene_bloque_de_fuentes() -> None:
    blocks, missing = sources.collect_blocks(ROOT)
    assert missing == []
    assert len(blocks) == 31


def test_ningun_bloque_de_fuentes_se_repite_entre_clases() -> None:
    blocks, _ = sources.collect_blocks(ROOT)
    fingerprints = [block.fingerprint for block in blocks]
    assert len(set(fingerprints)) == len(fingerprints)


def test_cada_cita_declara_el_uso_que_la_clase_hace_de_ella() -> None:
    blocks, _ = sources.collect_blocks(ROOT)
    sin_uso = [
        f"{c.course_path}:{c.line_number}"
        for c in sources.external_citations(blocks)
        if not c.declares_use
    ]
    assert sin_uso == []


def test_las_claves_de_cita_no_chocan_entre_entradas() -> None:
    index = sources.build_key_index(REGISTRY)
    assert len(index) >= len(ENTRIES)


def test_cobertura_total_de_las_obras_citadas() -> None:
    blocks, _ = sources.collect_blocks(ROOT)
    index = sources.build_key_index(REGISTRY)
    figures = sources.counts(blocks, REGISTRY, index)
    assert figures["coverage_pct"] == 100.0
    assert figures["covered_works"] == figures["distinct_works"]


def test_los_localizadores_usan_su_forma_canonica() -> None:
    for entry in ENTRIES:
        if entry["status"] == "pendiente":
            assert entry.get("pending_reason"), entry["id"]
            assert "locator" not in entry, entry["id"]
            continue
        if entry["type"] == "book":
            assert sources.isbn13_is_valid(entry["isbn13"]), entry["id"]
            assert entry["locator"] == f"https://openlibrary.org/isbn/{entry['isbn13']}"
        elif entry["type"] == "paper":
            assert sources.doi_is_wellformed(entry["doi"]), entry["id"]
            assert entry["locator"] == f"https://doi.org/{entry['doi']}"
        else:
            assert entry["locator"].startswith("https://"), entry["id"]
            assert entry["accessed"], entry["id"]


def test_cada_dataset_declara_procedencia_licencia_version_y_cita() -> None:
    datasets = [e for e in ENTRIES if e["type"] == "dataset"]
    assert datasets
    for entry in datasets:
        for field in ("provenance", "license", "version", "cite_as", "authority"):
            assert entry.get(field), f"{entry['id']}: falta {field}"
        integrity = entry["integrity"]
        if integrity.get("artifacts"):
            for artifact in integrity["artifacts"]:
                assert len(artifact["sha256"]) == 64, entry["id"]
                assert artifact["url"].startswith(("http://", "https://")), entry["id"]
        else:
            # Un hueco es admisible; un hueco sin declarar, no.
            assert integrity.get("status") == "pendiente", entry["id"]
            assert integrity.get("pending_reason"), entry["id"]


def test_ninguna_entrada_del_registro_queda_sin_usar() -> None:
    sin_uso = [e["id"] for e in ENTRIES if not e.get("used_in")]
    assert sin_uso == []


def test_el_verificador_pasa_de_extremo_a_extremo() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "verify-sources")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr
