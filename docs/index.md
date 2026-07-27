# Neural Network Training Labs

Plataforma para aprender y construir redes neuronales con datasets públicos reales, protocolo científico, notebooks evaluables, registro de modelos, API, exportación y entrenamiento distribuido. Versión **1.0.0**.

## Inicio rápido

```bash
pip install -e ".[dev,notebooks]"
neural-labs catalog
neural-labs train --lab 00_numpy_neuron --quick --split-seed 42 --training-seed 43
```

## Sitio de estudio

Además de esta documentación técnica (MkDocs), el repositorio publica un **sitio de estudio** en GitHub Pages con un recorrido lineal de los 31 laboratorios y navegación **anterior / siguiente**:

- 🌐 <https://vladimiracunadev-create.github.io/neural-network-training-labs/>

Cada laboratorio ancla su teoría en libros de referencia y papers seminales. Consulta [Sitio de estudio y navegación](study-site.md) para saber cómo se genera y cómo se mantiene la coherencia entre el Markdown del repositorio y el sitio.

## Componentes principales

- 25 laboratorios centrales y 6 especializaciones avanzadas.
- 19 fuentes públicas reales.
- 93 notebooks especializados: recorrido, estudiante y solución.
- Teoría anclada en libros de referencia (Géron, Goodfellow-Bengio-Courville, Bishop, Prince…) y papers seminales por arquitectura.
- Separación train/validation/test y lock experimental.
- Registro local y MLflow.
- FastAPI, métricas y telemetría.
- ONNX, INT8, ExecuTorch, DDP y FSDP2.
- SBOM, procedencia, hashes y firma opcional.
- Sitio de estudio en GitHub Pages con navegación anterior/siguiente.
