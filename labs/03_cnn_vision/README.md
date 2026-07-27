# CNN para visión

## Objetivo

Entrenar una CNN y analizar errores sobre fotografías reales de diez clases.

## Dataset real

- **Dataset:** `cifar10`
- **Fuente:** Torchvision / University of Toronto
- **Referencia:** https://www.cs.toronto.edu/~kriz/cifar.html
- **Licencia/condiciones:** Consultar términos CIFAR-10
- **Uso:** los datos se descargan desde la fuente; no hay ejemplos sintéticos ni archivos inventados.

CIFAR-10 contiene 60.000 imágenes reales de 32×32.

## Fundamento matemático

Convolución, pooling, normalización y clasificación.

## Protocolo experimental

1. Descargar y verificar la procedencia.
2. Conservar o crear una partición reproducible.
3. Ajustar transformaciones únicamente con `train`.
4. Seleccionar modelo e hiperparámetros usando `validation`.
5. Evaluar `test` una sola vez tras congelar la decisión.
6. Comparar con la línea base: **Clasificador lineal sobre píxeles**.
7. Guardar configuración, entorno, métricas, predicciones, gráficos y modelo.

## Ejecución

```bash
python labs/03_cnn_vision/train.py --quick
python labs/03_cnn_vision/train.py --config improved
```

Preparar únicamente el dataset:

```bash
python -m neural_labs.cli dataset --lab 03_cnn_vision
```

Inferencia y exportación:

```bash
neural-labs predict --lab 03_cnn_vision --run latest --input sample.json
neural-labs export --lab 03_cnn_vision --run latest --format onnx --verify
```

## Métricas

accuracy, balanced_accuracy, macro_precision, macro_recall, macro_f1.

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
neural-labs quality --lab 03_cnn_vision --quick
neural-labs benchmark --lab 03_cnn_vision --quick --split-seed 42 --training-seeds 41 42 43
neural-labs leaderboard
```

## Sellado del experimento

La partición se controla con `split_seed`; la inicialización y el entrenamiento con `training_seed`. El conjunto `test` se abre solamente después de seleccionar el checkpoint mediante validación y escribir `experiment.lock.json`.
