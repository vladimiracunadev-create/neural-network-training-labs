# Clasificación de audio con SpeechCommands

<!-- nav-top -->
> 🧭 **Ruta 28 / 31** · [⬅️ 🧷 Segmentación semántica con U-Net](../../advanced_labs/26_segmentation_unet/README.md) · [🏠 Índice](../../README.md#laboratorios) · [🖌️ WGAN-GP sobre Fashion-MNIST ➡️](../../advanced_labs/28_wgan_gp/README.md)
>
> **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

<!-- ficha -->
## 📋 Ficha del laboratorio

![ruta](https://img.shields.io/badge/ruta-28%20de%2031-7c5cff?style=flat-square) ![nivel](https://img.shields.io/badge/nivel-avanzado-8957e5?style=flat-square) ![categoría](https://img.shields.io/badge/categoría-Avanzada-2e8b57?style=flat-square) ![dataset](https://img.shields.io/badge/dataset-speechcommands__v0.02-1f6feb?style=flat-square) ![selección](https://img.shields.io/badge/selección-accuracy-8957e5?style=flat-square)

| Campo | Valor |
|---|---|
| 🧭 Posición | Ruta **28 de 31** del recorrido · categoría avanzada |
| 🎚️ Nivel | avanzado |
| 🗺️ Dominio | `audio` |
| 🏗️ Arquitectura | `audio-cnn` |
| 🗄️ Dataset | `speechcommands_v0.02` — Torchaudio / Google Speech Commands |
| ⚖️ Licencia del dataset | Creative Commons BY 4.0 |
| 🎯 Métrica de selección | `accuracy` sobre `validation` |
| 🔒 Política de `test` | se abre una sola vez, tras escribir `experiment.lock.json` |

### 🎯 Qué vas a poder hacer al terminar

- Clasificar comandos hablados desde waveform y log-mel spectrograms.
- Interpretar accuracy, macro_f1
- Aplicar sellado de test y reproducibilidad

### 🧩 Prerrequisitos

- CNN
- señales
- transformada tiempo-frecuencia

> Si alguno te falta, retrocede antes de continuar. Viniendo de [🧷 Segmentación semántica con U-Net](../../advanced_labs/26_segmentation_unet/README.md).

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

Clasificar comandos hablados desde waveform y log-mel spectrograms.

## Dataset público real

- **Dataset:** `speechcommands_v0.02`
- **Fuente:** Torchaudio / Google Speech Commands
- **Licencia/condiciones:** Creative Commons BY 4.0
- **Entrada:** audio mono de un segundo a 16 kHz
- **Datos sintéticos:** no se usan.

Los datos se descargan desde el proveedor oficial mediante los adaptadores del repositorio. Los archivos grandes no se incluyen en Git.

## Modelo y fundamento

- **Modelo:** `audio-cnn`
- **Teoría:** Waveform, espectrograma log-mel, convolución 2D y robustez ante ruido.
- **Línea base:** MFCC + regresión logística

## Protocolo

1. Crear particiones con `split_seed` o conservar los splits oficiales.
2. Entrenar y seleccionar exclusivamente con `train` y `validation`.
3. Guardar `best_model.pt` y escribir `experiment.lock.json`.
4. Abrir `test` una sola vez después del congelamiento.
5. Registrar métricas, configuración, procedencia y limitaciones.

## Ejecución

```bash
neural-labs train-advanced --track 27_audio_speechcommands --quick
neural-labs train-advanced --track 27_audio_speechcommands --split-seed 42 --training-seed 43
```

## Métricas

accuracy, macro_f1, confusion_matrix, noise_robustness.

## Cuadernos

- `notebook.ipynb`: recorrido completo.
- `notebook_student.ipynb`: actividades sin resolver.
- `notebook_solution.ipynb`: referencia docente.

## Limitación principal

Acentos, micrófonos y ambientes no están representados uniformemente.

<!-- nav-bottom -->
## 🧭 Navegación del recorrido

| ⬅️ Laboratorio anterior | 🏠 Índice | Laboratorio siguiente ➡️ |
|---|:---:|---|
| [🧷 Segmentación semántica con U-Net](../../advanced_labs/26_segmentation_unet/README.md) | [Las 31 rutas](../../README.md#laboratorios) | [🖌️ WGAN-GP sobre Fashion-MNIST](../../advanced_labs/28_wgan_gp/README.md) |

**En este laboratorio:** **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

[🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/27_audio_speechcommands/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
