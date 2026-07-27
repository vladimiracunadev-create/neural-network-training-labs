from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml

from .catalog import get_dataset, get_lab
from .runtime import environment_info, save_json


@dataclass
class ExperimentResult:
    lab_id: str
    run_dir: Path
    metrics: dict[str, Any]
    history: pd.DataFrame | None = None
    artifacts: dict[str, str] = field(default_factory=dict)


def initialize_run(
    run_dir: Path,
    lab_id: str,
    config: dict[str, Any],
    dataset_manifest: dict[str, Any],
    device: torch.device | None,
) -> None:
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    save_json(run_dir / "environment.json", environment_info(device))
    save_json(run_dir / "dataset_manifest.json", dataset_manifest)
    write_dataset_card(run_dir, lab_id, dataset_manifest)


def save_predictions(
    run_dir: Path,
    ids: list[str],
    y_true: Any,
    y_pred: Any,
    probabilities: Any | None = None,
) -> Path:
    frame = pd.DataFrame({"sample_id": ids, "y_true": list(y_true), "y_pred": list(y_pred)})
    if probabilities is not None:
        import numpy as np

        probs = np.asarray(probabilities)
        if probs.ndim == 1:
            frame["probability"] = probs
        else:
            for index in range(probs.shape[1]):
                frame[f"probability_{index}"] = probs[:, index]
    path = run_dir / "predictions.csv"
    frame.to_csv(path, index=False)
    return path


def write_dataset_card(run_dir: Path, lab_id: str, manifest: dict[str, Any]) -> Path:
    lab = get_lab(lab_id)
    dataset = get_dataset(lab_id)
    limitations = manifest.get("known_limitations") or dataset.get("known_limitations") or [lab.get("notes", "No declaradas.")]
    if isinstance(limitations, str):
        limitations = [limitations]
    content = f"""# Dataset Card — {lab['dataset']}

## Identificación

- Laboratorio: `{lab_id}`
- Dataset: `{lab['dataset']}`
- Fuente: {dataset['source']}
- Referencia: {dataset['source_ref']}
- Licencia/condiciones: {dataset['license']}
- Tipo de tarea: `{lab['task']}`

## Procedencia

Los archivos se obtienen en tiempo de ejecución desde la fuente declarada. El repositorio no redistribuye el dataset ni sustituye errores de descarga con datos generados.

## Particiones

- Transformadores y vocabularios: `train` solamente.
- Selección de modelo: `validation` solamente.
- Evaluación final: `test` después de congelar decisiones.
- Identidad: IDs y fingerprints registrados por ejecución.

## Resumen de esta ejecución

```json
{json.dumps(manifest.get('summary', {}), indent=2, ensure_ascii=False, default=str)}
```

## Limitaciones conocidas

{chr(10).join(f'- {item}' for item in limitations)}

## Uso responsable

Revise representatividad, privacidad, sesgos, licencia y restricciones de redistribución antes de reutilizar los datos. Esta ficha no reemplaza la documentación del proveedor.
"""
    path = run_dir / "dataset_card.md"
    path.write_text(content, encoding="utf-8")
    return path


def write_model_card(
    run_dir: Path,
    lab_id: str,
    metrics: dict[str, Any],
    baseline: dict[str, Any] | None,
    config: dict[str, Any],
) -> Path:
    lab = get_lab(lab_id)
    dataset = get_dataset(lab_id)
    content = f"""# Model Card — {lab['title']}

## Resumen

- Laboratorio: `{lab_id}`
- Arquitectura: `{lab['architecture']}`
- Tarea: `{lab['task']}`
- Dataset: `{lab['dataset']}`
- Fuente: {dataset['source']}
- Licencia/condiciones: {dataset['license']}
- Configuración: `{config.get('_config_name', 'desconocida')}`
- Semilla: `{config.get('seed')}`
- Dispositivo solicitado: `{config.get('device')}`

## Uso previsto

Aprendizaje, experimentación y comparación reproducible. Puede utilizarse como punto de partida para investigación o prototipos después de una validación independiente.

## Usos fuera de alcance

No usar directamente en decisiones médicas, financieras, laborales, legales, de seguridad o de alto impacto. No asumir que las métricas se transfieren a otra población, período, idioma, sensor o dominio.

## Datos y protocolo

- Entrenamiento: `train`.
- Selección: `validation`.
- Evaluación final: `test`, después de congelar decisiones.
- Línea base: {lab['baseline']}.
- Ficha completa: `dataset_card.md` y `dataset_manifest.json`.

## Resultados finales

```json
{json.dumps(metrics, indent=2, ensure_ascii=False, default=str)}
```

## Línea base

```json
{json.dumps(baseline or {}, indent=2, ensure_ascii=False, default=str)}
```

## Incertidumbre, segmentos y costo

Consulte `confidence_intervals.json`, `subgroup_metrics.json`, `data_quality.json`, `drift_report.json` y las métricas de perfil dentro de `metrics.json`. Una diferencia pequeña puede no ser estable ni justificar mayor complejidad.

## Riesgos y limitaciones

{dataset.get('notes', lab.get('notes', 'Consulte la ficha del dataset.'))}

Las explicaciones, probabilidades y métricas pueden ser incorrectas fuera de distribución. La ausencia de una brecha observada entre grupos no demuestra equidad.

## Reproducibilidad

Revise `config.yaml`, `environment.json`, `tracking.jsonl`, fingerprints y commit Git. PyTorch no garantiza identidad numérica entre todas las versiones y plataformas.
"""
    path = run_dir / "model_card.md"
    path.write_text(content, encoding="utf-8")
    return path


def write_report(
    run_dir: Path,
    lab_id: str,
    metrics: dict[str, Any],
    baseline: dict[str, Any] | None,
) -> Path:
    lab = get_lab(lab_id)
    content = f"""# Reporte experimental — {lab['title']}

## Pregunta

{lab['objective']}

## Hipótesis y evidencia

La red debe compararse contra **{lab['baseline']}**. Que la pérdida disminuya no demuestra por sí solo que el modelo sea útil ni que supere una alternativa más sencilla.

## Dataset

`{lab['dataset']}` — {lab['source']}. Consulte `dataset_card.md`, `dataset_manifest.json`, `data_quality.json` y `drift_report.json`.

## Línea base

```json
{json.dumps(baseline or {}, indent=2, ensure_ascii=False, default=str)}
```

## Modelo neuronal

```json
{json.dumps(metrics, indent=2, ensure_ascii=False, default=str)}
```

## Incertidumbre y errores

- Intervalos: `confidence_intervals.json`.
- Predicciones: `predictions.csv`.
- Matriz de confusión: `confusion_matrix.png`, cuando corresponde.
- Segmentos: `subgroup_metrics.json`, cuando hay variables crudas alineadas.

## Interpretación requerida

1. Magnitud de la diferencia frente a la línea base.
2. Estabilidad entre semillas e intervalo de confianza.
3. Clases, segmentos o períodos con mayor error.
4. Costo de entrenamiento e inferencia.
5. Límites de generalización y usos no recomendados.

## Reproducibilidad

La carpeta contiene configuración, entorno, métricas, historial, predicciones, checkpoints, seguimiento y fichas. El resultado debe poder reconstruirse sin usar el test para tomar decisiones.
"""
    path = run_dir / "report.md"
    path.write_text(content, encoding="utf-8")
    return path
