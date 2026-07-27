# Perceptrón con PyTorch

<!-- nav-top -->
> 🧭 [⬅️ Anterior](../../labs/00_numpy_neuron/README.md) · [🏠 Índice](../../README.md#laboratorios) · [Siguiente ➡️](../../labs/02_mlp_nonlinear/README.md)
<!-- /nav-top -->

## Objetivo

Aprender tensores, autograd, optimizadores y un clasificador lineal.

## Dataset real

- **Dataset:** `banknote_authentication`
- **Fuente:** UCI
- **Referencia:** https://archive.ics.uci.edu/dataset/267/banknote+authentication
- **Licencia/condiciones:** Consultar ficha UCI
- **Uso:** los datos se descargan desde la fuente; no hay ejemplos sintéticos ni archivos inventados.

Características extraídas de imágenes reales de billetes.

## Fundamento matemático

z=xW+b; BCEWithLogitsLoss.

## Protocolo experimental

1. Descargar y verificar la procedencia.
2. Conservar o crear una partición reproducible.
3. Ajustar transformaciones únicamente con `train`.
4. Seleccionar modelo e hiperparámetros usando `validation`.
5. Evaluar `test` una sola vez tras congelar la decisión.
6. Comparar con la línea base: **Regresión logística**.
7. Guardar configuración, entorno, métricas, predicciones, gráficos y modelo.

## Ejecución

```bash
python labs/01_pytorch_perceptron/train.py --quick
python labs/01_pytorch_perceptron/train.py --config improved
```

Preparar únicamente el dataset:

```bash
python -m neural_labs.cli dataset --lab 01_pytorch_perceptron
```

Inferencia y exportación:

```bash
neural-labs predict --lab 01_pytorch_perceptron --run latest --input sample.json
neural-labs export --lab 01_pytorch_perceptron --run latest --format onnx --verify
```

## Métricas

accuracy, balanced_accuracy, precision, recall, f1, roc_auc, pr_auc.

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
neural-labs quality --lab 01_pytorch_perceptron --quick
neural-labs benchmark --lab 01_pytorch_perceptron --quick --split-seed 42 --training-seeds 41 42 43
neural-labs leaderboard
```

## Sellado del experimento

La partición se controla con `split_seed`; la inicialización y el entrenamiento con `training_seed`. El conjunto `test` se abre solamente después de seleccionar el checkpoint mediante validación y escribir `experiment.lock.json`.

<!-- nav-bottom -->
## 🧭 Navegación del curso

| ⬅️ Anterior | Siguiente ➡️ |
|---|---|
| [🔢 Neurona con NumPy](../../labs/00_numpy_neuron/README.md) | [🌀 MLP multiclase](../../labs/02_mlp_nonlinear/README.md) |

[🏠 Portada del repositorio](../../README.md) · [🌐 Ver en el sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/01_pytorch_perceptron/index.html)
<!-- /nav-bottom -->
