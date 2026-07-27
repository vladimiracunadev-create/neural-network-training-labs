# Changelog

Este proyecto sigue [Versionado Semántico](https://semver.org/lang/es/).

## 1.0.0 — 2026-07-27

Primer release público. Consolida las 31 rutas de aprendizaje y toda la infraestructura de ingeniería de IA.

- Separación formal entre `split_seed` y `training_seed`.
- Sellado de cada experimento antes de abrir el conjunto de test.
- Refactorización por dominios y registro extensible de 23 factorías de modelos.
- DCGAN convolucional, RNN con padding enmascarado, transformer inspeccionable, TCN, GCN/GraphSAGE/GAT y Double Dueling DQN.
- 31 rutas de aprendizaje y 93 notebooks diferenciados con nbgrader.
- Seis especializaciones: DistilBERT/LoRA, U-Net, SpeechCommands, WGAN-GP, DDPM y SimCLR.
- Registro local y MLflow opcional con alias champion/challenger y puertas de promoción.
- API FastAPI, inferencia por lotes, Prometheus, OpenTelemetry, deriva y panel Streamlit.
- ONNX moderno, verificación ONNX Runtime, cuantización TorchAO/INT8 y ExecuTorch opcional.
- Entrenamiento DDP/FSDP2 y checkpoints distribuidos portables.
- SBOM, procedencia, hashes y flujo de release preparado para Cosign.
- 68 pruebas locales, 2 externas separadas y 84,62 % de cobertura.

## Pre-releases de desarrollo

Hitos internos previos al primer release público. Se conservan como referencia histórica.

### 0.3.0 — 2026-07-24

- Datasets públicos reales, sellado de experimentos y especializaciones avanzadas.
- API FastAPI, exportación ONNX/INT8/ExecuTorch, entrenamiento DDP/FSDP2 y cadena de suministro.

### 0.2.0 — 2026-07-24

- Profundización de los 25 laboratorios con teoría, planes experimentales, evaluación y metadatos pedagógicos.
- Nuevos notebooks que reutilizan el núcleo del repositorio.
- Validación estructurada de catálogo, configuración y manifiestos.
- Informe automático de calidad de datos, solapamientos y deriva entre particiones.
- Intervalos de confianza por bootstrap y métricas por subgrupo cuando existe alineación.
- Perfil de latencia, percentil 95 y throughput para modelos compatibles.
- Benchmark multi-semilla, comparación de runs y leaderboard.
- Seguimiento JSONL local e integración opcional con MLflow.
- Pipeline opcional con DVC, `params.yaml` y etapas reproducibles.
- Diagnóstico de entorno mediante `neural-labs doctor`.
- Precisión mixta CUDA, workers configurables y compilación experimental.
- Dockerfile, Dev Container, servidor MLflow local y documentación MkDocs.

### 0.1.0 — 2026-07-23

- Sustitución completa de datos generados por 19 datasets públicos reales.
- Ampliación a 25 laboratorios y 25 notebooks.
- Adaptadores para UCI, Torchvision, Hugging Face, PyTorch Geometric, scikit-learn y KaggleHub.
- Protocolo estricto train/validation/test y auditoría de fuga de datos.
- Líneas base, métricas específicas, model cards, reportes y artefactos reproducibles.
