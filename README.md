<div align="center">

# 🧠 Neural Network Training Labs

## **31 rutas · 93 notebooks · el ciclo completo de ingeniería de IA, con datasets públicos reales**

**Plataforma evolutiva para aprender, experimentar, validar, exportar y desplegar redes neuronales — de la neurona en NumPy a difusión, transformers y aprendizaje autosupervisado — con semillas separadas, sellado del `test`, model/dataset cards, registro champion/challenger, API de inferencia, exportación ONNX/INT8/edge, entrenamiento distribuido y cadena de suministro verificable.**

[![CI](https://github.com/vladimiracunadev-create/neural-network-training-labs/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/vladimiracunadev-create/neural-network-training-labs/actions/workflows/ci.yml)
[![Notebooks](https://github.com/vladimiracunadev-create/neural-network-training-labs/actions/workflows/notebooks.yml/badge.svg?branch=main)](https://github.com/vladimiracunadev-create/neural-network-training-labs/actions/workflows/notebooks.yml)
[![Security](https://github.com/vladimiracunadev-create/neural-network-training-labs/actions/workflows/security.yml/badge.svg?branch=main)](https://github.com/vladimiracunadev-create/neural-network-training-labs/actions/workflows/security.yml)
[![Docs](https://github.com/vladimiracunadev-create/neural-network-training-labs/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/vladimiracunadev-create/neural-network-training-labs/actions/workflows/docs.yml)

[![Rutas](https://img.shields.io/badge/rutas-31%20·%2025%20labs%20%2B%206%20avanzadas-7c5cff?style=for-the-badge)](#laboratorios)
[![Notebooks](https://img.shields.io/badge/notebooks-93-2e8b57?style=for-the-badge)](#notebooks-evaluables)
[![Python](https://img.shields.io/badge/python-3.11%20·%203.12%20·%203.13-3776ab?style=for-the-badge)](pyproject.toml)
[![Versión](https://img.shields.io/badge/versión-1.0.0-orange?style=for-the-badge)](CHANGELOG.md)
[![License](https://img.shields.io/badge/license-MIT-3fb950?style=for-the-badge)](LICENSE)

[📚 Documentación](docs/index.md) · [🧭 Ruta de aprendizaje](docs/learning-path.md) · [🔬 Protocolo de experimento](docs/experiment-protocol.md) · [🗺️ Roadmap](ROADMAP.md) · [🤝 Contribuir](CONTRIBUTING.md) · [🔐 Seguridad](SECURITY.md)

</div>

---

> 🧭 **Estado del proyecto (v1.0.0).** Las **31 rutas están construidas**: 25 laboratorios centrales y 6 especializaciones avanzadas, con **93 notebooks** (recorrido, versión estudiante y solución por ruta). Todas comparten el mismo contrato de ingeniería: `split_seed` y `training_seed` independientes, `validation` decide el modelo, `test` se abre solo tras sellar `experiment.lock.json`.
>
> **Qué verifica una máquina y qué no**, para que sepas de qué te fías: la **CI** compila el código, ejecuta las pruebas sin red (`pytest -m "not network and not slow"`), valida la estructura del repositorio y los contratos de notebooks/nbgrader, y corre `ruff` y una auditoría de cadena de suministro (SBOM + SHA-256). Los **datasets son descargas reales** desde sus proveedores (UCI, Torchvision, Torchaudio, Hugging Face, PyG, Kaggle): no se generan datos sintéticos para tapar una descarga fallida, y los entrenamientos largos y las descargas quedan en pruebas externas marcadas para no esconder fallos de red.

## Qué resuelve

Cada laboratorio cubre el ciclo completo:

```text
fuente pública real
      ↓
descarga y licencia
      ↓
train / validation / test
      ↓
línea base en validation
      ↓
selección del modelo
      ↓
experiment.lock.json
      ↓
evaluación final en test
      ↓
model card + dataset card
      ↓
registro champion/challenger
      ↓
API, ONNX, INT8 o edge
```

## Principios

- No se generan datasets sintéticos para reemplazar una descarga fallida.
- Los transformadores, escaladores y vocabularios se ajustan solo con `train`.
- `validation` decide arquitectura, hiperparámetros, umbrales y checkpoint.
- `test` se abre después de generar `experiment.lock.json`.
- `split_seed` y `training_seed` son independientes.
- Los benchmarks usan una partición fija y varias semillas de entrenamiento.
- Los datasets grandes no se guardan dentro del repositorio.
- Cada modelo desplegable conserva su contrato de inferencia.

## Laboratorios

El repositorio ofrece **31 rutas de aprendizaje**: 25 laboratorios centrales y 6 especializaciones avanzadas. Los centrales cubren fundamentos, CNN, RNN/LSTM, transformer desde cero, DCGAN, GNN, Double Dueling DQN, transferencia, destilación, federado, explicabilidad, calibración, exportación y proyecto final.

Las especializaciones agregan fine-tuning de DistilBERT con LoRA, segmentación U-Net, audio SpeechCommands, WGAN-GP, difusión DDPM y aprendizaje autosupervisado SimCLR.

```bash
neural-labs catalog
neural-labs advanced
neural-labs models
```

En total se incluyen **93 notebooks**: recorrido completo, versión estudiante y solución para cada ruta. Los datasets proceden de UCI, Torchvision, Torchaudio, Hugging Face, PyTorch Geometric, Kaggle y repositorios académicos. Ejemplos: CIFAR-10, Fashion-MNIST, IMDb, AG News, Cora, SpeechCommands, UCI HAR, Online Retail, Oxford-IIIT Pet, Adult Census, Wine Quality e Iranian Churn.

## Especializaciones avanzadas

```bash
neural-labs train-advanced --track 25_transformer_finetuning --quick --lora
neural-labs train-advanced --track 26_segmentation_unet --quick
neural-labs train-advanced --track 27_audio_speechcommands --quick
neural-labs train-advanced --track 28_wgan_gp --quick
neural-labs train-advanced --track 29_diffusion_ddpm --quick
neural-labs train-advanced --track 30_self_supervised_simclr --quick
```

Estas rutas mantienen el mismo contrato de semillas, selección por validación y sellado del test. Los modelos preentrenados y datasets se descargan desde sus proveedores; no se incluyen pesos ni datos grandes dentro del ZIP.

## Estructura de cada laboratorio

```text
labs/03_cnn_vision/
├── README.md
├── theory.md
├── experiments.md
├── assessment.md
├── lesson.yaml
├── train.py
├── notebook.ipynb
├── notebook_student.ipynb
├── notebook_solution.ipynb
├── configs/
│   ├── baseline.yaml
│   └── improved.yaml
└── data/
    └── dataset.yaml
```

Los cuadernos no son copias de una misma plantilla. Visión incluye mapas de activación y errores por clase; texto trabaja tokenización, padding y atención; series temporales incorporan backtesting; grafos comparan GCN, GraphSAGE y GAT; generación analiza diversidad y colapso; refuerzo compara políticas e indicadores de inventario.

## Instalación

### Entorno básico

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev,notebooks]"
```

### Entorno completo

```bash
pip install -e ".[full,dev,serving,mlops,data-versioning]"
```

Se recomienda usar Python 3.11 o 3.12 para la mayor compatibilidad de extras científicos.

## Flujo de entrenamiento

```bash
neural-labs dataset \
  --lab 03_cnn_vision \
  --quick \
  --split-seed 42

neural-labs audit \
  --lab 03_cnn_vision \
  --quick \
  --split-seed 42

neural-labs train \
  --lab 03_cnn_vision \
  --config improved \
  --split-seed 42 \
  --training-seed 43 \
  --device auto
```

Cada ejecución genera, según el laboratorio:

```text
best_model.pt
last_model.pt
experiment.lock.json
inference_contract.json
preprocessor.joblib
vocabulary.json
model_spec.json
metrics.json
baseline_validation_metrics.json
baseline_metrics.json
confidence_intervals.json
subgroup_metrics.json
data_quality.json
drift_report.json
predictions.csv
history.csv
history.png
confusion_matrix.png
model_card.md
report.md
tracking.jsonl
```

## Benchmark sin mezclar variabilidades

```bash
neural-labs benchmark \
  --lab 02_mlp_nonlinear \
  --split-seed 42 \
  --training-seeds 41 42 43 44 45 \
  --config improved
```

La partición permanece fija. Solo cambia la aleatoriedad del entrenamiento.

Para problemas tabulares:

```bash
neural-labs cross-validate \
  --lab 24_capstone_real_project \
  --folds 5 \
  --split-seed 42 \
  --training-seeds 41 42 43
```

La validación cruzada trabaja sobre desarrollo y no utiliza el conjunto final de test.

## Notebooks evaluables

```bash
pip install -e ".[education,notebooks]"
nbgrader validate assignments/source/03_cnn_vision/notebook.ipynb
```

- `notebook_student.ipynb`: ejercicios incompletos.
- `notebook_solution.ipynb`: solución docente.
- `assignments/source`: material del instructor.
- `assignments/release`: material para estudiantes.

## Registro de modelos

### Registro local

```bash
neural-labs registry register \
  --name cifar-classifier \
  --run runs/03_cnn_vision/<run> \
  --alias challenger

neural-labs registry promote \
  --name cifar-classifier \
  --version 2 \
  --alias champion \
  --metric macro_f1 \
  --minimum 0.80 \
  --max-latency-ms 20

neural-labs registry resolve \
  --name cifar-classifier \
  --alias champion
```

### MLflow

```bash
neural-labs registry register \
  --backend mlflow \
  --tracking-uri http://localhost:5000 \
  --name cifar-classifier \
  --run runs/03_cnn_vision/<run> \
  --alias challenger
```

## Inferencia externa

```bash
neural-labs predict \
  --lab 03_cnn_vision \
  --run latest \
  --input sample.png
```

También admite JSON, CSV y NumPy cuando el contrato del modelo corresponde.

## API de inferencia

Registrar un modelo con alias `champion` y ejecutar:

```bash
export NEURAL_LABS_REGISTRY=model-registry.json
export NEURAL_LABS_MODEL=cifar-classifier
export NEURAL_LABS_MODEL_REFERENCE=champion
neural-labs serve --host 0.0.0.0 --port 8000
```

Endpoints:

```text
GET  /health
GET  /model
GET  /drift
POST /predict
POST /predict-batch
GET  /metrics
```

La API expone métricas en formato Prometheus, registra resúmenes estadísticos de predicción sin conservar las entradas crudas y activa OpenTelemetry cuando están instalados sus SDK.

Inferencia por lotes y panel operativo:

```bash
neural-labs batch-predict --lab 03_cnn_vision --run latest --input batch.npy --output predictions.csv
neural-labs monitor
neural-labs dashboard
```

## Exportación y edge

```bash
neural-labs export --lab 23_model_export_and_inference --run latest --format onnx --verify
neural-labs export --lab 23_model_export_and_inference --run latest --format int8
neural-labs export --lab 23_model_export_and_inference --run latest --format benchmark
neural-labs export --lab 23_model_export_and_inference --run latest --format executorch
```

El flujo ONNX usa el exportador basado en `torch.export`, puede verificar equivalencia con ONNX Runtime y guarda un informe. La cuantización dinámica registra tamaño y latencia. ExecuTorch es opcional porque requiere dependencias y toolchains específicas de la plataforma objetivo.

## Entrenamiento distribuido

Diagnóstico:

```bash
neural-labs distributed
```

DDP:

```bash
torchrun --standalone --nproc-per-node=2 \
  -m neural_labs train-distributed \
  --lab 03_cnn_vision \
  --strategy ddp \
  --quick
```

FSDP2:

```bash
torchrun --standalone --nproc-per-node=2 \
  -m neural_labs train-distributed \
  --lab 07_transformer_attention \
  --strategy fsdp2
```

Cada rango guarda su shard y el rango cero genera el manifiesto de ejecución.

## Seguridad y procedencia

```bash
neural-labs supply-chain
```

Genera:

```text
dist/security/SHA256SUMS
dist/security/sbom.cdx.json
dist/security/provenance.json
```

Los workflows de release incluyen instrucciones para firmar artefactos con Cosign y publicar procedencia.

## Calidad

```bash
pytest -m "not network and not slow" --cov=neural_labs --cov-report=term-missing
python -m ruff check src scripts tests
python -m compileall -q src labs scripts tests
neural-labs validate --warnings-as-errors
```

La cobertura obligatoria del núcleo de producción es 80 %. Los adaptadores de descarga y los entrenamientos largos tienen pruebas externas o programadas separadas para no esconder fallos de red.

## Reproducibilidad con uv

El proyecto está preparado para:

```bash
uv lock
uv sync --locked --extra dev --extra notebooks
```

Si el índice de paquetes no está disponible, no se fabrica un lockfile incompleto. `requirements/core-tested.txt` conserva el entorno validado y el workflow comprueba que `uv.lock` esté actualizado cuando pueda resolverse desde un índice normal.

## Documentación

```bash
pip install -e ".[docs]"
mkdocs serve
```

Consulta especialmente:

- `docs/experiment-protocol.md`
- `docs/education.md`
- `docs/model-registry-and-serving.md`
- `docs/export-and-edge.md`
- `docs/distributed-training.md`
- `docs/supply-chain-security.md`
- `docs/architecture.md`
- `docs/datasets.md`

## Alcance responsable

Los laboratorios sirven para aprendizaje, investigación y prototipos. No convierten automáticamente un modelo en una solución apta para decisiones médicas, financieras, laborales o de seguridad. Cada despliegue debe revisar licencias, privacidad, representatividad, calibración, deriva, subgrupos, seguridad y supervisión humana.
