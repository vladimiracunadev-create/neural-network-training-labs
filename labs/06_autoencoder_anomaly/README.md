# Autoencoder para fraude

<!-- nav-top -->
> 🧭 **Ruta 7 / 31** · [⬅️ 📈 LSTM para series temporales](../../labs/05_lstm_time_series/README.md) · [🏠 Índice](../../README.md#laboratorios) · [🔭 Transformer para noticias ➡️](../../labs/07_transformer_attention/README.md)
>
> **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

<!-- ficha -->
## 📋 Ficha del laboratorio

![ruta](https://img.shields.io/badge/ruta-7%20de%2031-7c5cff?style=flat-square) ![nivel](https://img.shields.io/badge/nivel-intermedio-1f6feb?style=flat-square) ![categoría](https://img.shields.io/badge/categoría-Central-2e8b57?style=flat-square) ![horas](https://img.shields.io/badge/horas-~6%20h-f0b429?style=flat-square) ![dataset](https://img.shields.io/badge/dataset-credit__card__fraud-1f6feb?style=flat-square) ![selección](https://img.shields.io/badge/selección-f1-8957e5?style=flat-square)

| Campo | Valor |
|---|---|
| 🧭 Posición | Ruta **7 de 31** del recorrido · categoría central |
| 🎚️ Nivel | intermedio |
| ⏱️ Dedicación estimada | 6 horas |
| 🧩 Tarea | `anomaly_detection` |
| 🏗️ Arquitectura | `autoencoder` |
| 🗄️ Dataset | [`credit_card_fraud`](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — Kaggle / ULB |
| ⚖️ Licencia del dataset | Uso sujeto a términos de Kaggle y autor |
| 🎯 Métrica de selección | `f1` sobre `validation` |
| 📏 Línea base a superar | Isolation Forest |
| 🔒 Política de `test` | se abre una sola vez, tras escribir `experiment.lock.json` |

### 🎯 Qué vas a poder hacer al terminar

- Detectar transacciones fraudulentas mediante error de reconstrucción.
- Preparar y auditar el dataset real credit_card_fraud sin fuga de datos.
- Entrenar y evaluar reconstrucción para detección de anomalías.
- Comparar contra la línea base: Isolation Forest.
- Interpretar intervalos de confianza, errores y limitaciones.

### 🧩 Prerrequisitos

- PyTorch básico
- particiones train/validation/test
- métricas de evaluación

> Si alguno te falta, retrocede antes de continuar. Viniendo de [📈 LSTM para series temporales](../../labs/05_lstm_time_series/README.md).

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

Detectar transacciones fraudulentas mediante error de reconstrucción.

## Dataset real

- **Dataset:** `credit_card_fraud`
- **Fuente:** Kaggle / ULB
- **Referencia:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- **Licencia/condiciones:** Uso sujeto a términos de Kaggle y autor
- **Uso:** los datos se descargan desde la fuente; no hay ejemplos sintéticos ni archivos inventados.

284.807 transacciones reales; el laboratorio evita reequilibrar el conjunto de test.

## Fundamento matemático

Minimizar ||x-decoder(encoder(x))||² sobre transacciones normales.

## Protocolo experimental

1. Descargar y verificar la procedencia.
2. Conservar o crear una partición reproducible.
3. Ajustar transformaciones únicamente con `train`.
4. Seleccionar modelo e hiperparámetros usando `validation`.
5. Evaluar `test` una sola vez tras congelar la decisión.
6. Comparar con la línea base: **Isolation Forest**.
7. Guardar configuración, entorno, métricas, predicciones, gráficos y modelo.

## Ejecución

```bash
python labs/06_autoencoder_anomaly/train.py --quick
python labs/06_autoencoder_anomaly/train.py --config improved
```

Preparar únicamente el dataset:

```bash
python -m neural_labs.cli dataset --lab 06_autoencoder_anomaly
```

Inferencia y exportación:

```bash
neural-labs predict --lab 06_autoencoder_anomaly --run latest --input sample.json
neural-labs export --lab 06_autoencoder_anomaly --run latest --format onnx --verify
```

## Métricas

precision, recall, f1, roc_auc, pr_auc.

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
neural-labs quality --lab 06_autoencoder_anomaly --quick
neural-labs benchmark --lab 06_autoencoder_anomaly --quick --split-seed 42 --training-seeds 41 42 43
neural-labs leaderboard
```

## Sellado del experimento

La partición se controla con `split_seed`; la inicialización y el entrenamiento con `training_seed`. El conjunto `test` se abre solamente después de seleccionar el checkpoint mediante validación y escribir `experiment.lock.json`.

<!-- nav-bottom -->
## 🧭 Navegación del recorrido

| ⬅️ Laboratorio anterior | 🏠 Índice | Laboratorio siguiente ➡️ |
|---|:---:|---|
| [📈 LSTM para series temporales](../../labs/05_lstm_time_series/README.md) | [Las 31 rutas](../../README.md#laboratorios) | [🔭 Transformer para noticias](../../labs/07_transformer_attention/README.md) |

**En este laboratorio:** **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

[🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/06_autoencoder_anomaly/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
