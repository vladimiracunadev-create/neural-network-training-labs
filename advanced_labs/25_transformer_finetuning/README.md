# Fine-tuning eficiente de transformer

<!-- nav-top -->
> 🧭 **Ruta 26 / 31** · 🔬 [Parte 7 — Especializaciones avanzadas](../../parts/07-especializaciones-avanzadas.md)
>
> [⬅️ 🏁 Proyecto final: churn de telecomunicaciones](../../labs/24_capstone_real_project/README.md) · [🏠 Índice de rutas](../../parts/README.md) · [🧷 Segmentación semántica con U-Net ➡️](../../advanced_labs/26_segmentation_unet/README.md)
>
> **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

<!-- ficha -->
## 📋 Ficha del laboratorio

![ruta](https://img.shields.io/badge/ruta-26%20de%2031-7c5cff?style=flat-square) ![nivel](https://img.shields.io/badge/nivel-avanzado-8957e5?style=flat-square) ![categoría](https://img.shields.io/badge/categoría-Avanzada-2e8b57?style=flat-square) ![dataset](https://img.shields.io/badge/dataset-ag__news-1f6feb?style=flat-square) ![selección](https://img.shields.io/badge/selección-accuracy-8957e5?style=flat-square)

| Campo | Valor |
|---|---|
| 🧭 Posición | Ruta **26 de 31** del recorrido · categoría avanzada |
| 🎚️ Nivel | avanzado |
| 🗺️ Dominio | `text` |
| 🏗️ Arquitectura | `distilbert-base-uncased` |
| 🗄️ Dataset | `ag_news` — Hugging Face Datasets |
| ⚖️ Licencia del dataset | Consultar ficha AG News |
| 🎯 Métrica de selección | `accuracy` sobre `validation` |
| 🔒 Política de `test` | se abre una sola vez, tras escribir `experiment.lock.json` |

### 🎯 Qué vas a poder hacer al terminar

- Comparar fine-tuning completo y LoRA sin tocar test durante selección.
- Interpretar accuracy, macro_f1
- Aplicar sellado de test y reproducibilidad

### 🧩 Prerrequisitos

- PyTorch
- NLP
- Transformers

> Si alguno te falta, retrocede antes de continuar. Viniendo de [🏁 Proyecto final: churn de telecomunicaciones](../../labs/24_capstone_real_project/README.md).

### 📦 Entregables y criterios de aceptación

**Entregables**

- notebook ejecutado
- reporte experimental
- model card

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

Comparar fine-tuning completo y LoRA sin tocar test durante selección.

## Dataset público real

- **Dataset:** `ag_news`
- **Fuente:** Hugging Face Datasets
- **Licencia/condiciones:** Consultar ficha AG News
- **Entrada:** texto en inglés
- **Datos sintéticos:** no se usan.

Los datos se descargan desde el proveedor oficial mediante los adaptadores del repositorio. Los archivos grandes no se incluyen en Git.

## Modelo y fundamento

- **Modelo:** `distilbert-base-uncased`
- **Teoría:** Tokenización subword, atención preentrenada, fine-tuning completo y adaptación eficiente LoRA.
- **Línea base:** TF-IDF + regresión logística

## Protocolo

1. Crear particiones con `split_seed` o conservar los splits oficiales.
2. Entrenar y seleccionar exclusivamente con `train` y `validation`.
3. Guardar `best_model.pt` y escribir `experiment.lock.json`.
4. Abrir `test` una sola vez después del congelamiento.
5. Registrar métricas, configuración, procedencia y limitaciones.

## Ejecución

```bash
neural-labs train-advanced --track 25_transformer_finetuning --quick
neural-labs train-advanced --track 25_transformer_finetuning --split-seed 42 --training-seed 43
```

## Métricas

accuracy, macro_f1, latency_ms, trainable_parameters.

## Cuadernos

- `notebook.ipynb`: recorrido completo.
- `notebook_student.ipynb`: actividades sin resolver.
- `notebook_solution.ipynb`: referencia docente.

## Limitación principal

El corpus contiene titulares históricos y sesgos editoriales; no representa todo el lenguaje contemporáneo.

<!-- nav-bottom -->
## 🧭 Navegación del recorrido

| ⬅️ Laboratorio anterior | 🏠 Índice | Laboratorio siguiente ➡️ |
|---|:---:|---|
| [🏁 Proyecto final: churn de telecomunicaciones](../../labs/24_capstone_real_project/README.md) | [Las 31 rutas](../../parts/README.md) | [🧷 Segmentación semántica con U-Net](../../advanced_labs/26_segmentation_unet/README.md) |

**En este laboratorio:** **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🔬 [Parte 7 — Especializaciones avanzadas](../../parts/07-especializaciones-avanzadas.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/25_transformer_finetuning/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
