# Arquitectura

```text
src/neural_labs/
├── core/
│   ├── protocol.py       # semillas y test lock
│   └── registry.py       # factorías extensibles
├── domains/
│   ├── tabular/
│   ├── vision/
│   ├── text/
│   ├── time_series/
│   ├── graphs/
│   ├── generative/
│   └── reinforcement/
├── datasets.py           # adaptadores de fuentes reales
├── experiments.py        # orquestación compatible
├── inference.py          # contrato y carga de ejecución
├── exporting.py          # ONNX, INT8 y ExecuTorch
├── model_registry.py     # versiones y alias locales
├── mlflow_registry.py    # backend MLflow opcional
├── deployment/api.py     # API FastAPI
├── distributed.py        # contexto DDP/FSDP2
├── distributed_training.py
├── telemetry.py
└── supply_chain.py
```

El registro desacopla nombres de arquitectura de sus implementaciones. Agregar un modelo requiere registrar una factoría en su dominio, sin ampliar una cadena central de `if/elif`.

La orquestación histórica permanece para compatibilidad, pero las nuevas capacidades se construyen sobre contratos separados y módulos por dominio.

## Estructura de un laboratorio

Cada laboratorio (`labs/NN_slug/` o `advanced_labs/NN_slug/`) es autocontenido:

```text
labs/03_cnn_vision/
├── README.md            # objetivo, dataset, protocolo, ejecución y navegación anterior/siguiente
├── theory.md            # fundamento matemático + sección "🔗 Referencias" (libros y papers)
├── experiments.md       # hipótesis, variables controladas y tabla multi-semilla
├── assessment.md        # preguntas y rúbrica de evaluación
├── lesson.yaml          # resultados de aprendizaje, prerrequisitos y entregables
├── train.py             # interfaz de terminal que usa el mismo código del cuaderno
├── notebook.ipynb       # recorrido completo · notebook_student · notebook_solution
├── configs/             # baseline.yaml e improved.yaml
└── data/dataset.yaml    # procedencia, licencia y política de partición
```

La teoría de cada laboratorio se ancla en la literatura de referencia del tema y en los papers seminales de su arquitectura; la lista de fuentes por área vive en el README raíz y la cita concreta, en la sección `🔗 Referencias` de cada `theory.md`.

## Scripts

```text
scripts/
├── validate_repository.py           # valida estructura, catálogo y contratos de notebooks
├── build_lab_docs.py                # ficha + navegación en los 4 documentos de cada laboratorio (idempotente)
├── generate_lab_html.py             # página HTML autocontenida por laboratorio + índice offline
├── generate_site.py                 # genera el sitio de estudio (GitHub Pages) con navegación
├── generate_specialized_notebooks.py
├── prepare_datasets.py · audit_splits.py · check_dataset_sources.py
├── run_lab.py · smoke_test.py · clean_runs.py
├── dvc_smoke.py
└── validate_nbgrader.py
```

`build_lab_docs.py`, `generate_lab_html.py` y `generate_site.py` mantienen el recorrido de estudio en sus tres superficies —Markdown, HTML versionado y sitio de Pages— a partir de una única fuente: el Markdown de los laboratorios. Si cambian los títulos o el orden, se ejecutan en ese orden y todo queda sincronizado; los dos primeros aceptan `--check` y la CI los usa para impedir que las superficies se desincronicen. Consulta [Sitio de estudio y navegación](study-site.md).

## Flujos de integración continua

```text
.github/workflows/
├── ci.yml               # markdown lint + calidad (3.11/3.12/3.13) + empaquetado
├── deploy-pages.yml     # genera y publica el sitio de estudio en GitHub Pages
├── security.yml         # gitleaks + bandit + pip-audit + evidencia de cadena de suministro
├── docs.yml             # mkdocs build --strict
├── notebooks.yml        # contratos de notebooks y nbgrader
├── release.yml          # artefactos firmables por tag
├── benchmark.yml · distributed.yml · edge.yml · api.yml
├── advanced-smoke.yml · real-data-smoke.yml
```
