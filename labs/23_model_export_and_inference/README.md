# Exportación e inferencia

## Objetivo

Exportar ONNX, validar paridad y medir latencia por lotes.

## Dataset real

- **Dataset:** `cifar10`
- **Fuente:** Torchvision / University of Toronto
- **Referencia:** https://www.cs.toronto.edu/~kriz/cifar.html
- **Licencia/condiciones:** Consultar términos CIFAR-10
- **Uso:** los datos se descargan desde la fuente; no hay ejemplos sintéticos ni archivos inventados.

Incluye predicción, exportación y benchmark reproducible.

## Fundamento matemático

Paridad numérica y costo de inferencia.

## Protocolo experimental

1. Descargar y verificar la procedencia.
2. Conservar o crear una partición reproducible.
3. Ajustar transformaciones únicamente con `train`.
4. Seleccionar modelo e hiperparámetros usando `validation`.
5. Evaluar `test` una sola vez tras congelar la decisión.
6. Comparar con la línea base: **PyTorch eager**.
7. Guardar configuración, entorno, métricas, predicciones, gráficos y modelo.

## Ejecución

```bash
python labs/23_model_export_and_inference/train.py --quick
python labs/23_model_export_and_inference/train.py --config improved
```

Preparar únicamente el dataset:

```bash
python -m neural_labs.cli dataset --lab 23_model_export_and_inference
```

Inferencia y exportación:

```bash
neural-labs predict --lab 23_model_export_and_inference --run latest --input sample.json
neural-labs export --lab 23_model_export_and_inference --run latest --format onnx --verify
```

## Métricas

accuracy, latency_ms, throughput, model_size_mb.

## Archivos

- `notebook.ipynb`: recorrido completo y ejecutable.
- `notebook_student.ipynb`: actividades evaluables sin soluciones.
- `notebook_solution.ipynb`: resolución docente y pruebas de referencia.
- `train.py`: interfaz de terminal que usa el mismo código del cuaderno.
- `configs/baseline.yaml`: configuración base.
- `configs/improved.yaml`: configuración ampliada.
- `data/dataset.yaml`: procedencia, licencia y política de partición.

## Ejercicios

- Cambiar una decisión experimental y justificarla.
- Analizar errores por clase o segmento.
- Comparar costo, precisión y latencia.
- Documentar sesgos, limitaciones y usos no recomendados.


## Material formativo v3

- [`theory.md`](theory.md): fundamento, protocolo y riesgos de interpretación.
- [`experiments.md`](experiments.md): hipótesis, variables controladas y tabla multi-semilla.
- [`assessment.md`](assessment.md): preguntas y rúbrica de evaluación.
- [`lesson.yaml`](lesson.yaml): resultados de aprendizaje, prerrequisitos y entregables.

## Comandos profesionales

```bash
neural-labs quality --lab 23_model_export_and_inference --quick
neural-labs benchmark --lab 23_model_export_and_inference --quick --split-seed 42 --training-seeds 41 42 43
neural-labs leaderboard
```

## Sellado del experimento

La partición se controla con `split_seed`; la inicialización y el entrenamiento con `training_seed`. El conjunto `test` se abre solamente después de seleccionar el checkpoint mediante validación y escribir `experiment.lock.json`.
