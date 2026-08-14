# Educación y evaluación

Cada laboratorio ofrece tres cuadernos con papeles distintos: `notebook.ipynb` es el recorrido de referencia y no lleva ejercicios; `notebook_student.ipynb` añade la práctica; `notebook_solution.ipynb` la trae resuelta. Los ejercicios llevan metadatos nbgrader, pruebas visibles y espacio para evaluación escrita.

## Los cinco ejercicios evaluables

Cada laboratorio incluye **cinco ejercicios de 5 puntos**, iguales en estructura para las 31 rutas y parametrizados con los datos de cada una —su métrica de selección, su línea base y su experimento propio—:

| Identificador | Qué se practica |
|---|---|
| `auditoria_particiones` | Comprobar que `train`, `validation` y `test` no comparten ningún ejemplo. |
| `regla_de_seleccion` | Elegir el checkpoint con la métrica del laboratorio, sabiendo si mayor o menor es mejor. |
| `comparacion_linea_base` | Decidir si la ventaja sobre la línea base supera la dispersión entre semillas. |
| `sellado_del_test` | No abrir `test` sin `experiment.lock.json`. |
| `plan_experimental` | Dejar por escrito hipótesis, variable que cambia, controles, semillas y conclusión. |

Cubren el contrato experimental en vez de la arquitectura, que es lo que estos laboratorios enseñan y un tutorial no. Se resuelven con la biblioteca estándar —sin descargar datasets ni entrenar—, así que la corrección es de segundos y no necesita GPU. Cada celda de ejercicio va seguida de una celda de comprobación bloqueada que debe pasar sin error.

La solución de referencia de los 31 laboratorios **se ejecuta en la integración continua** (`tests/test_notebook_exercises_v3.py`): un ejercicio cuya solución no corre es peor que no tenerlo, porque el estudiante no puede saber si el fallo es suyo o del material.

## Flujo docente

```bash
pip install -e ".[education,notebooks]"

# Regenerar los cuadernos: centrales y especializaciones
python scripts/generate_specialized_notebooks.py
python scripts/generate_advanced_exercises.py

nbgrader validate assignments/source/03_cnn_vision/notebook.ipynb
nbgrader generate_assignment 03_cnn_vision
```

Los enunciados y las soluciones se definen una sola vez en [`scripts/lab_exercises.py`](https://github.com/vladimiracunadev-create/neural-network-training-labs/blob/main/scripts/lab_exercises.py), de donde los toman los dos generadores; así los identificadores nbgrader no pueden divergir entre laboratorios.

Los cuadernos se especializan por dominio:

- Visión: imágenes, activaciones, Grad-CAM, errores y robustez.
- Texto: tokenización, padding, longitud, atención y truncamiento.
- Series: orden temporal, ventanas, backtesting y horizonte.
- Grafos: vecindarios, embeddings y ablación de aristas.
- Generación: diversidad, interpolación y colapso.
- Refuerzo: trayectorias, política, retorno y estabilidad.
- Multimodal: ablación por sensor y modality dropout.
- Tabular: calibración, subgrupos, importancia y comparación con modelos simples.
