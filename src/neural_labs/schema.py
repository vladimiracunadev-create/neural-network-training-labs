from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class SchemaError(ValueError):
    """Raised when a lab or dataset manifest is incomplete or inconsistent."""


@dataclass(frozen=True)
class ValidationIssue:
    path: str
    message: str
    severity: str = "error"

    def as_dict(self) -> dict[str, str]:
        return {"path": self.path, "message": self.message, "severity": self.severity}


LAB_REQUIRED = {
    "id",
    "title",
    "level",
    "task",
    "architecture",
    "dataset",
    "source",
    "source_type",
    "source_id",
    "source_ref",
    "license",
    "baseline",
    "metrics",
    "objective",
    "math",
    "notes",
}

DATASET_REQUIRED = {
    "schema_version",
    "name",
    "task",
    "source",
    "source_type",
    "source_id",
    "source_url",
    "license",
    "provenance",
    "retrieval",
    "split",
    "integrity",
    "redistribution",
    "known_limitations",
}

CONFIG_REQUIRED = {
    "lab",
    "title",
    "task",
    "architecture",
    "dataset",
    "seed",
    "device",
    "epochs",
    "batch_size",
    "learning_rate",
    "patience",
    "quick",
    "selection_metric",
    "test_policy",
    "baseline",
    "metrics",
}


def _missing(mapping: dict[str, Any], required: set[str]) -> list[str]:
    return sorted(required.difference(mapping))


def validate_lab_definition(lab: dict[str, Any], *, path: str = "lab") -> list[ValidationIssue]:
    issues = [ValidationIssue(path, f"Falta el campo requerido: {key}") for key in _missing(lab, LAB_REQUIRED)]
    if "metrics" in lab and not isinstance(lab["metrics"], list):
        issues.append(ValidationIssue(f"{path}.metrics", "Debe ser una lista."))
    if "source_ref" in lab and not str(lab["source_ref"]).startswith(("http://", "https://")):
        issues.append(ValidationIssue(f"{path}.source_ref", "Debe ser una URL pública de procedencia."))
    if lab.get("level") not in {"fundamentos", "intermedio", "avanzado", "produccion", "proyecto"}:
        issues.append(ValidationIssue(f"{path}.level", "Nivel desconocido.", "warning"))
    return issues


def validate_dataset_manifest(manifest: dict[str, Any], *, path: str = "dataset") -> list[ValidationIssue]:
    issues = [ValidationIssue(path, f"Falta el campo requerido: {key}") for key in _missing(manifest, DATASET_REQUIRED)]
    provenance = manifest.get("provenance", {})
    if provenance.get("generated_data") is not False:
        issues.append(ValidationIssue(f"{path}.provenance.generated_data", "Debe ser false: el repositorio exige datos reales."))
    if provenance.get("real_world_data") is not True:
        issues.append(ValidationIssue(f"{path}.provenance.real_world_data", "Debe ser true."))
    retrieval = manifest.get("retrieval", {})
    if retrieval.get("fallback_to_generated_data") is not False:
        issues.append(ValidationIssue(f"{path}.retrieval.fallback_to_generated_data", "No se permite reemplazar una descarga fallida por datos inventados."))
    split = manifest.get("split", {})
    if split.get("transform_fit_scope") != "train_only":
        issues.append(ValidationIssue(f"{path}.split.transform_fit_scope", "Los transformadores deben ajustarse solo con train."))
    if split.get("selection_scope") != "validation_only":
        issues.append(ValidationIssue(f"{path}.split.selection_scope", "La selección debe usar validation solamente."))
    if split.get("test_policy") not in {"evaluate_once_after_model_selection", "official_test_evaluate_once", "held_out_test_evaluate_once"}:
        issues.append(ValidationIssue(f"{path}.split.test_policy", "Debe declarar una política explícita de evaluación final."))
    return issues


def validate_training_config(config: dict[str, Any], *, path: str = "config") -> list[ValidationIssue]:
    issues = [ValidationIssue(path, f"Falta el campo requerido: {key}") for key in _missing(config, CONFIG_REQUIRED)]
    for key in ("epochs", "batch_size", "patience"):
        if key in config and int(config[key]) <= 0:
            issues.append(ValidationIssue(f"{path}.{key}", "Debe ser mayor que cero."))
    if "learning_rate" in config and float(config["learning_rate"]) <= 0:
        issues.append(ValidationIssue(f"{path}.learning_rate", "Debe ser mayor que cero."))
    if config.get("test_policy") != "evaluate_once_after_model_selection":
        issues.append(ValidationIssue(f"{path}.test_policy", "La política recomendada es evaluate_once_after_model_selection."))
    return issues


def validate_repository(root: Path) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    catalog_path = root / "configs" / "labs.yaml"
    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    seen: set[str] = set()
    for index, lab in enumerate(catalog.get("labs", [])):
        lab_id = str(lab.get("id", f"index-{index}"))
        if lab_id in seen:
            issues.append(ValidationIssue(f"configs/labs.yaml[{index}].id", f"ID duplicado: {lab_id}"))
        seen.add(lab_id)
        issues.extend(validate_lab_definition(lab, path=f"configs/labs.yaml:{lab_id}"))
        lab_dir = root / "labs" / lab_id
        for relative in ("README.md", "notebook.ipynb", "train.py", "lesson.yaml", "theory.md", "experiments.md", "assessment.md"):
            if not (lab_dir / relative).is_file():
                issues.append(ValidationIssue(str(lab_dir / relative), "Archivo educativo requerido ausente."))
        manifest_path = lab_dir / "data" / "dataset.yaml"
        if manifest_path.is_file():
            manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            issues.extend(validate_dataset_manifest(manifest, path=str(manifest_path)))
        else:
            issues.append(ValidationIssue(str(manifest_path), "Manifiesto de dataset ausente."))
        for config_name in ("baseline", "improved"):
            config_path = lab_dir / "configs" / f"{config_name}.yaml"
            if not config_path.is_file():
                issues.append(ValidationIssue(str(config_path), "Configuración requerida ausente."))
                continue
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            issues.extend(validate_training_config(config, path=str(config_path)))
            if config.get("lab") != lab_id:
                issues.append(ValidationIssue(f"{config_path}.lab", f"Debe coincidir con {lab_id}."))
    return issues


def raise_for_errors(issues: list[ValidationIssue]) -> None:
    errors = [issue for issue in issues if issue.severity == "error"]
    if errors:
        detail = "\n".join(f"- {issue.path}: {issue.message}" for issue in errors[:25])
        raise SchemaError(f"Se detectaron {len(errors)} errores de esquema:\n{detail}")
