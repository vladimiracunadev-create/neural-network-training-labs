from __future__ import annotations

import re
from pathlib import Path

import yaml

from .catalog import ROOT


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    if not normalized:
        raise ValueError("El nombre no produce un identificador válido.")
    return normalized


def create_lab_scaffold(name: str, *, title: str | None = None, root: Path = ROOT) -> Path:
    existing = sorted(path.name for path in (root / "labs").iterdir() if path.is_dir())
    next_number = max([int(item.split("_", 1)[0]) for item in existing if item[:2].isdigit()] + [-1]) + 1
    lab_id = f"{next_number:02d}_{_slug(name)}"
    lab_dir = root / "labs" / lab_id
    if lab_dir.exists():
        raise FileExistsError(lab_dir)
    (lab_dir / "configs").mkdir(parents=True)
    (lab_dir / "data").mkdir()
    display_title = title or name.replace("_", " ").title()
    (lab_dir / "README.md").write_text(f"# {display_title}\n\nComplete el objetivo, dataset, protocolo y criterios de éxito.\n", encoding="utf-8")
    (lab_dir / "theory.md").write_text(f"# Teoría — {display_title}\n\nDesarrolle el fundamento matemático y las decisiones de arquitectura.\n", encoding="utf-8")
    (lab_dir / "experiments.md").write_text(f"# Experimentos — {display_title}\n\nDefina hipótesis, variables controladas y tabla de resultados.\n", encoding="utf-8")
    (lab_dir / "assessment.md").write_text(f"# Evaluación — {display_title}\n\nIncluya preguntas, ejercicios y una rúbrica verificable.\n", encoding="utf-8")
    lesson = {
        "schema_version": "2.0",
        "lab": lab_id,
        "title": display_title,
        "estimated_hours": 4,
        "prerequisites": [],
        "learning_outcomes": [],
        "deliverables": ["notebook ejecutado", "reporte experimental", "model card"],
    }
    (lab_dir / "lesson.yaml").write_text(yaml.safe_dump(lesson, sort_keys=False, allow_unicode=True), encoding="utf-8")
    (lab_dir / "train.py").write_text(
        "from pathlib import Path\nimport sys\n\nROOT = Path(__file__).resolve().parents[2]\nsys.path.insert(0, str(ROOT / 'src'))\n\nfrom neural_labs.cli import run_fixed_lab\n\nif __name__ == '__main__':\n    run_fixed_lab('" + lab_id + "')\n",
        encoding="utf-8",
    )
    return lab_dir
