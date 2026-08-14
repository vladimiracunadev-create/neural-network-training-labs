# Regularización

<!-- nav-top -->
> 🧭 **Ruta 20 / 31** · 🔴 [Parte 5 — La mecánica fina, ahora en profundidad](../../parts/05-mecanica-fina.md)
>
> [⬅️ ⚙️ Optimizadores y schedulers](../../labs/18_optimizers_and_schedulers/README.md) · [🏠 Índice de rutas](../../parts/README.md) · [🔄 Aumento de datos ➡️](../../labs/20_data_augmentation/README.md)
>
> **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

<!-- ficha -->
## 📋 Ficha del laboratorio

![ruta](https://img.shields.io/badge/ruta-20%20de%2031-7c5cff?style=flat-square) ![nivel](https://img.shields.io/badge/nivel-intermedio-1f6feb?style=flat-square) ![categoría](https://img.shields.io/badge/categoría-Central-2e8b57?style=flat-square) ![horas](https://img.shields.io/badge/horas-~6%20h-f0b429?style=flat-square) ![dataset](https://img.shields.io/badge/dataset-fashion__mnist-1f6feb?style=flat-square) ![selección](https://img.shields.io/badge/selección-macro__f1-8957e5?style=flat-square)

| Campo | Valor |
|---|---|
| 🧭 Posición | Ruta **20 de 31** del recorrido · categoría central |
| 🎚️ Nivel | intermedio |
| ⏱️ Dedicación estimada | 6 horas |
| 🧩 Tarea | `multiclass_classification` |
| 🏗️ Arquitectura | `regularization_comparison` |
| 🗄️ Dataset | [`fashion_mnist`](https://github.com/zalandoresearch/fashion-mnist) — Torchvision / Zalando Research |
| ⚖️ Licencia del dataset | MIT |
| 🎯 Métrica de selección | `macro_f1` sobre `validation` |
| 📏 Línea base a superar | MLP sin regularización |
| 🔒 Política de `test` | se abre una sola vez, tras escribir `experiment.lock.json` |

### 🎯 Qué vas a poder hacer al terminar

- Medir dropout, weight decay y batch normalization.
- Preparar y auditar el dataset real fashion_mnist sin fuga de datos.
- Entrenar y evaluar dropout, batch normalization y weight decay.
- Comparar contra la línea base: MLP sin regularización.
- Interpretar intervalos de confianza, errores y limitaciones.

### 🧩 Prerrequisitos

- PyTorch básico
- particiones train/validation/test
- métricas de evaluación

> Si alguno te falta, retrocede antes de continuar. Viniendo de [⚙️ Optimizadores y schedulers](../../labs/18_optimizers_and_schedulers/README.md).

### ⚙️ `baseline` frente a `improved`

| Parámetro | [`baseline.yaml`](configs/baseline.yaml) | [`improved.yaml`](configs/improved.yaml) |
|---|---|---|
| Épocas | `20` | `50` |
| Tasa de aprendizaje | `0.001` | `0.0005` |
| Paciencia (early stopping) | `5` | `8` |
| Precisión mixta (AMP) | no | sí |
| Procesos de carga | `0` | `2` |

> Solo se muestran los parámetros en los que ambas configuraciones difieren. La elección entre una y otra se decide con `validation`, nunca con `test`.

### 📦 Entregables y criterios de aceptación

**Entregables**

- notebook ejecutado
- reporte experimental
- model card
- comparación con línea base
- respuesta a preguntas críticas

**Criterios de éxito**

- cero solapamiento entre train, validation y test
- selección basada únicamente en validation
- métricas finales acompañadas por incertidumbre
- conclusiones que distinguen evidencia de suposición

### 🗂️ Recursos del laboratorio

| Recurso | Archivo |
|---|---|
| 🧠 Teoría y referencias | [`theory.md`](theory.md) |
| 🔬 Plan de experimentos | [`experiments.md`](experiments.md) |
| 📝 Evaluación y rúbrica | [`assessment.md`](assessment.md) |
| 📓 Notebook de recorrido | [`notebook.ipynb`](notebook.ipynb) |
| ✏️ Notebook de estudiante | [`notebook_student.ipynb`](notebook_student.ipynb) |
| ✅ Notebook de solución | [`notebook_solution.ipynb`](notebook_solution.ipynb) |
| 🖥️ Script de terminal | [`train.py`](train.py) |
| 🎛️ Configuración base | [`configs/baseline.yaml`](configs/baseline.yaml) |
| 🎚️ Configuración ampliada | [`configs/improved.yaml`](configs/improved.yaml) |
| 🗄️ Ficha del dataset | [`data/dataset.yaml`](data/dataset.yaml) |
| 🧾 Metadatos de la lección | [`lesson.yaml`](lesson.yaml) |

<!-- /ficha -->

## Objetivo

Medir dropout, weight decay y batch normalization.

## Dataset real

- **Dataset:** `fashion_mnist`
- **Fuente:** Torchvision / Zalando Research
- **Referencia:** https://github.com/zalandoresearch/fashion-mnist
- **Licencia/condiciones:** MIT
- **Uso:** los datos se descargan desde la fuente; no hay ejemplos sintéticos ni archivos inventados.

Prendas reales normalizadas en 28×28.

## Fundamento matemático

Regularización explícita e implícita; brecha train-validation.

## Protocolo experimental

1. Descargar y verificar la procedencia.
2. Conservar o crear una partición reproducible.
3. Ajustar transformaciones únicamente con `train`.
4. Seleccionar modelo e hiperparámetros usando `validation`.
5. Evaluar `test` una sola vez tras congelar la decisión.
6. Comparar con la línea base: **MLP sin regularización**.
7. Guardar configuración, entorno, métricas, predicciones, gráficos y modelo.

## Ejecución

```bash
python labs/19_regularization_dropout_batchnorm/train.py --quick
python labs/19_regularization_dropout_batchnorm/train.py --config improved
```

Preparar únicamente el dataset:

```bash
python -m neural_labs.cli dataset --lab 19_regularization_dropout_batchnorm
```

Inferencia y exportación:

```bash
neural-labs predict --lab 19_regularization_dropout_batchnorm --run latest --input sample.json
neural-labs export --lab 19_regularization_dropout_batchnorm --run latest --format onnx --verify
```

## Métricas

accuracy, macro_f1, generalization_gap.

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
neural-labs quality --lab 19_regularization_dropout_batchnorm --quick
neural-labs benchmark --lab 19_regularization_dropout_batchnorm --quick --split-seed 42 --training-seeds 41 42 43
neural-labs leaderboard
```

## Sellado del experimento

La partición se controla con `split_seed`; la inicialización y el entrenamiento con `training_seed`. El conjunto `test` se abre solamente después de seleccionar el checkpoint mediante validación y escribir `experiment.lock.json`.

<!-- nav-bottom -->
## 🧭 Navegación del recorrido

| ⬅️ Laboratorio anterior | 🏠 Índice | Laboratorio siguiente ➡️ |
|---|:---:|---|
| [⚙️ Optimizadores y schedulers](../../labs/18_optimizers_and_schedulers/README.md) | [Las 31 rutas](../../parts/README.md) | [🔄 Aumento de datos](../../labs/20_data_augmentation/README.md) |

**En este laboratorio:** **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🔴 [Parte 5 — La mecánica fina, ahora en profundidad](../../parts/05-mecanica-fina.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/19_regularization_dropout_batchnorm/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
