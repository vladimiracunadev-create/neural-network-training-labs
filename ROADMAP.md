# 🗺️ Roadmap

El detalle vive en [`docs/roadmap.md`](docs/roadmap.md). Este resumen orienta hacia dónde va el proyecto.

## ✅ Entregado en 1.0.0

- **31 rutas de aprendizaje** (25 laboratorios centrales + 6 especializaciones avanzadas) y **93 archivos de cuaderno** con contrato nbgrader. Con una precisión que conviene declarar: hoy `notebook.ipynb` y `notebook_solution.ipynb` tienen el mismo contenido en los 31 laboratorios, y la versión de estudiante deja **2 celdas** por completar. El código del laboratorio está completo y es ejecutable en los tres; la práctica guiada es lo que falta.
- Contrato de experimento reproducible: `split_seed` y `training_seed` independientes, selección por `validation`, sellado de `test` con `experiment.lock.json`.
- Registro de modelos local y MLflow opcional con alias champion/challenger y puertas de promoción.
- API de inferencia FastAPI con métricas Prometheus, OpenTelemetry y endpoint de deriva.
- Exportación ONNX moderna, cuantización INT8/TorchAO y ExecuTorch opcional.
- Entrenamiento distribuido DDP/FSDP2 con checkpoints portables.
- Cadena de suministro: SBOM, procedencia, SHA-256 y flujo de release preparado para Cosign.

## 🚧 Próximas líneas

- **Cuadernos con práctica real.** Ampliar los ejercicios de la versión de estudiante —de 2 celdas a una secuencia que cubra las decisiones propias de cada laboratorio— y separar el cuaderno de recorrido del de solución, hoy idénticos. Es la brecha más visible entre lo que el material promete y lo que entrega.
- Orquestación Kubernetes y pruebas multinodo reales.
- Feature store con gobierno y control de acceso.
- Evaluación continua con feedback verificado.
- Privacidad diferencial y aprendizaje federado seguro.

Las propuestas se discuten en *issues*; consulta [CONTRIBUTING.md](CONTRIBUTING.md).
