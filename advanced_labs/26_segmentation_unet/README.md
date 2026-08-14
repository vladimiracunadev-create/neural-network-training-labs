# Segmentación semántica con U-Net

<!-- nav-top -->
> 🧭 **Ruta 27 / 31** · [⬅️ 🔧 Fine-tuning eficiente de transformer](../../advanced_labs/25_transformer_finetuning/README.md) · [🏠 Índice](../../README.md#laboratorios) · [🎙️ Clasificación de audio con SpeechCommands ➡️](../../advanced_labs/27_audio_speechcommands/README.md)
>
> **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

<!-- ficha -->
## 📋 Ficha del laboratorio

![ruta](https://img.shields.io/badge/ruta-27%20de%2031-7c5cff?style=flat-square) ![nivel](https://img.shields.io/badge/nivel-avanzado-8957e5?style=flat-square) ![categoría](https://img.shields.io/badge/categoría-Avanzada-2e8b57?style=flat-square) ![dataset](https://img.shields.io/badge/dataset-oxford__iiit__pet__segmentation-1f6feb?style=flat-square) ![selección](https://img.shields.io/badge/selección-mean__iou-8957e5?style=flat-square)

| Campo | Valor |
|---|---|
| 🧭 Posición | Ruta **27 de 31** del recorrido · categoría avanzada |
| 🎚️ Nivel | avanzado |
| 🗺️ Dominio | `vision` |
| 🏗️ Arquitectura | `unet-small` |
| 🗄️ Dataset | `oxford_iiit_pet_segmentation` — Torchvision / University of Oxford |
| ⚖️ Licencia del dataset | Consultar términos Oxford-IIIT Pet |
| 🎯 Métrica de selección | `mean_iou` sobre `validation` |
| 🔒 Política de `test` | se abre una sola vez, tras escribir `experiment.lock.json` |

### 🎯 Qué vas a poder hacer al terminar

- Segmentar mascota, fondo y contorno con IoU por clase.
- Interpretar mean_iou, iou_per_class
- Aplicar sellado de test y reproducibilidad

### 🧩 Prerrequisitos

- CNN
- visión
- métricas por píxel

> Si alguno te falta, retrocede antes de continuar. Viniendo de [🔧 Fine-tuning eficiente de transformer](../../advanced_labs/25_transformer_finetuning/README.md).

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

Segmentar mascota, fondo y contorno con IoU por clase.

## Dataset público real

- **Dataset:** `oxford_iiit_pet_segmentation`
- **Fuente:** Torchvision / University of Oxford
- **Licencia/condiciones:** Consultar términos Oxford-IIIT Pet
- **Entrada:** imagen RGB y máscara trimap
- **Datos sintéticos:** no se usan.

Los datos se descargan desde el proveedor oficial mediante los adaptadores del repositorio. Los archivos grandes no se incluyen en Git.

## Modelo y fundamento

- **Modelo:** `unet-small`
- **Teoría:** Arquitectura encoder-decoder, conexiones skip, pérdida por píxel e intersección sobre unión.
- **Línea base:** Máscara de clase mayoritaria

## Protocolo

1. Crear particiones con `split_seed` o conservar los splits oficiales.
2. Entrenar y seleccionar exclusivamente con `train` y `validation`.
3. Guardar `best_model.pt` y escribir `experiment.lock.json`.
4. Abrir `test` una sola vez después del congelamiento.
5. Registrar métricas, configuración, procedencia y limitaciones.

## Ejecución

```bash
neural-labs train-advanced --track 26_segmentation_unet --quick
neural-labs train-advanced --track 26_segmentation_unet --split-seed 42 --training-seed 43
```

## Métricas

mean_iou, iou_per_class, pixel_accuracy, dice.

## Cuadernos

- `notebook.ipynb`: recorrido completo.
- `notebook_student.ipynb`: actividades sin resolver.
- `notebook_solution.ipynb`: referencia docente.

## Limitación principal

Las imágenes se concentran en mascotas y fondos cotidianos; no generaliza a segmentación médica o industrial.

<!-- nav-bottom -->
## 🧭 Navegación del recorrido

| ⬅️ Laboratorio anterior | 🏠 Índice | Laboratorio siguiente ➡️ |
|---|:---:|---|
| [🔧 Fine-tuning eficiente de transformer](../../advanced_labs/25_transformer_finetuning/README.md) | [Las 31 rutas](../../README.md#laboratorios) | [🎙️ Clasificación de audio con SpeechCommands](../../advanced_labs/27_audio_speechcommands/README.md) |

**En este laboratorio:** **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

[🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/26_segmentation_unet/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
