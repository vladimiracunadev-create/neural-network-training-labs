# Educación y evaluación

Cada laboratorio ofrece cuaderno de estudiante y cuaderno de solución. Los ejercicios contienen metadatos compatibles con nbgrader, pruebas visibles y espacio para evaluación escrita.

## Flujo docente

```bash
pip install -e ".[education,notebooks]"
python scripts/generate_specialized_notebooks.py
nbgrader validate assignments/source/03_cnn_vision/notebook.ipynb
nbgrader generate_assignment 03_cnn_vision
```

Los cuadernos se especializan por dominio:

- Visión: imágenes, activaciones, Grad-CAM, errores y robustez.
- Texto: tokenización, padding, longitud, atención y truncamiento.
- Series: orden temporal, ventanas, backtesting y horizonte.
- Grafos: vecindarios, embeddings y ablación de aristas.
- Generación: diversidad, interpolación y colapso.
- Refuerzo: trayectorias, política, retorno y estabilidad.
- Multimodal: ablación por sensor y modality dropout.
- Tabular: calibración, subgrupos, importancia y comparación con modelos simples.
