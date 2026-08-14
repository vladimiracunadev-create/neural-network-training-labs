# Aprendizaje federado por participante

<!-- nav-top -->
> 🧭 **Ruta 16 / 31** · [⬅️ ⚗️ Destilación de conocimiento](../../labs/14_knowledge_distillation/README.md) · [🏠 Índice](../../README.md#laboratorios) · [∂ Backpropagation manual ➡️](../../labs/16_backpropagation_manual/README.md)
>
> **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

<!-- ficha -->
## 📋 Ficha del laboratorio

![ruta](https://img.shields.io/badge/ruta-16%20de%2031-7c5cff?style=flat-square) ![nivel](https://img.shields.io/badge/nivel-avanzado-8957e5?style=flat-square) ![categoría](https://img.shields.io/badge/categoría-Central-2e8b57?style=flat-square) ![horas](https://img.shields.io/badge/horas-~8%20h-f0b429?style=flat-square) ![dataset](https://img.shields.io/badge/dataset-uci__har__subjects-1f6feb?style=flat-square) ![selección](https://img.shields.io/badge/selección-macro__f1-8957e5?style=flat-square)

| Campo | Valor |
|---|---|
| 🧭 Posición | Ruta **16 de 31** del recorrido · categoría central |
| 🎚️ Nivel | avanzado |
| ⏱️ Dedicación estimada | 8 horas |
| 🧩 Tarea | `multiclass_classification` |
| 🏗️ Arquitectura | `fedavg_mlp` |
| 🗄️ Dataset | [`uci_har_subjects`](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) — UCI |
| ⚖️ Licencia del dataset | CC BY 4.0 |
| 🎯 Métrica de selección | `macro_f1` sobre `validation` |
| 📏 Línea base a superar | Entrenamiento centralizado |
| 🔒 Política de `test` | se abre una sola vez, tras escribir `experiment.lock.json` |

### 🎯 Qué vas a poder hacer al terminar

- Aplicar FedAvg usando participantes reales como clientes naturales.
- Preparar y auditar el dataset real uci_har_subjects sin fuga de datos.
- Entrenar y evaluar agregación federada de clientes reales.
- Comparar contra la línea base: Entrenamiento centralizado.
- Interpretar intervalos de confianza, errores y limitaciones.

### 🧩 Prerrequisitos

- PyTorch intermedio
- optimización
- lectura de artículos técnicos

> Si alguno te falta, retrocede antes de continuar. Viniendo de [⚗️ Destilación de conocimiento](../../labs/14_knowledge_distillation/README.md).

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

Aplicar FedAvg usando participantes reales como clientes naturales.

## Dataset real

- **Dataset:** `uci_har_subjects`
- **Fuente:** UCI
- **Referencia:** https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
- **Licencia/condiciones:** CC BY 4.0
- **Uso:** los datos se descargan desde la fuente; no hay ejemplos sintéticos ni archivos inventados.

No crea clientes espaciales artificiales; conserva identificadores reales de sujetos.

## Fundamento matemático

w_{t+1}=Σ_k(n_k/n)w_k.

## Protocolo experimental

1. Descargar y verificar la procedencia.
2. Conservar o crear una partición reproducible.
3. Ajustar transformaciones únicamente con `train`.
4. Seleccionar modelo e hiperparámetros usando `validation`.
5. Evaluar `test` una sola vez tras congelar la decisión.
6. Comparar con la línea base: **Entrenamiento centralizado**.
7. Guardar configuración, entorno, métricas, predicciones, gráficos y modelo.

## Ejecución

```bash
python labs/15_federated_learning/train.py --quick
python labs/15_federated_learning/train.py --config improved
```

Preparar únicamente el dataset:

```bash
python -m neural_labs.cli dataset --lab 15_federated_learning
```

Inferencia y exportación:

```bash
neural-labs predict --lab 15_federated_learning --run latest --input sample.json
neural-labs export --lab 15_federated_learning --run latest --format onnx --verify
```

## Métricas

accuracy, macro_f1, client_accuracy_std.

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
neural-labs quality --lab 15_federated_learning --quick
neural-labs benchmark --lab 15_federated_learning --quick --split-seed 42 --training-seeds 41 42 43
neural-labs leaderboard
```

## Sellado del experimento

La partición se controla con `split_seed`; la inicialización y el entrenamiento con `training_seed`. El conjunto `test` se abre solamente después de seleccionar el checkpoint mediante validación y escribir `experiment.lock.json`.

<!-- nav-bottom -->
## 🧭 Navegación del recorrido

| ⬅️ Laboratorio anterior | 🏠 Índice | Laboratorio siguiente ➡️ |
|---|:---:|---|
| [⚗️ Destilación de conocimiento](../../labs/14_knowledge_distillation/README.md) | [Las 31 rutas](../../README.md#laboratorios) | [∂ Backpropagation manual](../../labs/16_backpropagation_manual/README.md) |

**En este laboratorio:** **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

[🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/15_federated_learning/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
