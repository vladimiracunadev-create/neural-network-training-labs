# Neural Network Training Labs 3

Plataforma para aprender y construir redes neuronales con datasets públicos reales, protocolo científico, notebooks evaluables, registro de modelos, API, exportación y entrenamiento distribuido.

## Inicio rápido

```bash
pip install -e ".[dev,notebooks]"
neural-labs catalog
neural-labs train --lab 00_numpy_neuron --quick --split-seed 42 --training-seed 43
```

## Componentes principales

- 25 laboratorios centrales y 6 especializaciones avanzadas.
- 19 fuentes públicas reales.
- 93 notebooks especializados: recorrido, estudiante y solución.
- Separación train/validation/test y lock experimental.
- Registro local y MLflow.
- FastAPI, métricas y telemetría.
- ONNX, INT8, ExecuTorch, DDP y FSDP2.
- SBOM, procedencia, hashes y firma opcional.
