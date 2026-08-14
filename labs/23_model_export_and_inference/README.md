# Exportación e inferencia

<!-- nav-top -->
> 🧭 **Ruta 24 / 31** · ⚫ [Parte 6 — Confiar en el modelo y sacarlo del cuaderno](../../parts/06-confianza-y-despliegue.md)
>
> [⬅️ 🎯 Incertidumbre y calibración](../../labs/22_uncertainty_calibration/README.md) · [🏠 Índice de rutas](../../parts/README.md) · [🏁 Proyecto final: churn de telecomunicaciones ➡️](../../labs/24_capstone_real_project/README.md)
>
> **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

<!-- ficha -->
## 📋 Ficha del laboratorio

![ruta](https://img.shields.io/badge/ruta-24%20de%2031-7c5cff?style=flat-square) ![nivel](https://img.shields.io/badge/nivel-avanzado-8957e5?style=flat-square) ![categoría](https://img.shields.io/badge/categoría-Central-2e8b57?style=flat-square) ![horas](https://img.shields.io/badge/horas-~8%20h-f0b429?style=flat-square) ![dataset](https://img.shields.io/badge/dataset-cifar10-1f6feb?style=flat-square) ![selección](https://img.shields.io/badge/selección-macro__f1-8957e5?style=flat-square)

| Campo | Valor |
|---|---|
| 🧭 Posición | Ruta **24 de 31** del recorrido · categoría central |
| 🎚️ Nivel | avanzado |
| ⏱️ Dedicación estimada | 8 horas |
| 🧩 Tarea | `multiclass_classification` |
| 🏗️ Arquitectura | `cnn_export` |
| 🗄️ Dataset | [`cifar10`](https://www.cs.toronto.edu/~kriz/cifar.html) — Torchvision / University of Toronto |
| ⚖️ Licencia del dataset | Consultar términos CIFAR-10 |
| 🎯 Métrica de selección | `macro_f1` sobre `validation` |
| 📏 Línea base a superar | PyTorch eager |
| 🔒 Política de `test` | se abre una sola vez, tras escribir `experiment.lock.json` |

### 🎯 Qué vas a poder hacer al terminar

- Exportar ONNX, validar paridad y medir latencia por lotes.
- Preparar y auditar el dataset real cifar10 sin fuga de datos.
- Entrenar y evaluar exportación y perfil de inferencia.
- Comparar contra la línea base: PyTorch eager.
- Interpretar intervalos de confianza, errores y limitaciones.

### 🧩 Prerrequisitos

- PyTorch intermedio
- optimización
- lectura de artículos técnicos

> Si alguno te falta, retrocede antes de continuar. Viniendo de [🎯 Incertidumbre y calibración](../../labs/22_uncertainty_calibration/README.md).

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

<!-- nav-bottom -->
## 🧭 Navegación del recorrido

| ⬅️ Laboratorio anterior | 🏠 Índice | Laboratorio siguiente ➡️ |
|---|:---:|---|
| [🎯 Incertidumbre y calibración](../../labs/22_uncertainty_calibration/README.md) | [Las 31 rutas](../../parts/README.md) | [🏁 Proyecto final: churn de telecomunicaciones](../../labs/24_capstone_real_project/README.md) |

**En este laboratorio:** **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

⚫ [Parte 6 — Confiar en el modelo y sacarlo del cuaderno](../../parts/06-confianza-y-despliegue.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/23_model_export_and_inference/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
