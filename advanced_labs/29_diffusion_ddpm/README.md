# Difusión DDPM sobre Fashion-MNIST

<!-- nav-top -->
> 🧭 **Ruta 30 / 31** · 🔬 [Parte 7 — Especializaciones avanzadas](../../parts/07-especializaciones-avanzadas.md)
>
> [⬅️ 🖌️ WGAN-GP sobre Fashion-MNIST](../../advanced_labs/28_wgan_gp/README.md) · [🏠 Índice de rutas](../../parts/README.md) · [🪞 Aprendizaje autosupervisado SimCLR ➡️](../../advanced_labs/30_self_supervised_simclr/README.md)
>
> **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

<!-- ficha -->
## 📋 Ficha del laboratorio

![ruta](https://img.shields.io/badge/ruta-30%20de%2031-7c5cff?style=flat-square) ![nivel](https://img.shields.io/badge/nivel-avanzado-8957e5?style=flat-square) ![categoría](https://img.shields.io/badge/categoría-Avanzada-2e8b57?style=flat-square) ![dataset](https://img.shields.io/badge/dataset-fashion__mnist-1f6feb?style=flat-square) ![selección](https://img.shields.io/badge/selección-noise__mse-8957e5?style=flat-square)

| Campo | Valor |
|---|---|
| 🧭 Posición | Ruta **30 de 31** del recorrido · categoría avanzada |
| 🎚️ Nivel | avanzado |
| 🗺️ Dominio | `generative` |
| 🏗️ Arquitectura | `tiny-ddpm` |
| 🗄️ Dataset | `fashion_mnist` — Torchvision / Zalando Research |
| ⚖️ Licencia del dataset | MIT para código; consultar dataset |
| 🎯 Métrica de selección | `noise_mse` sobre `validation` |
| 🔒 Política de `test` | se abre una sola vez, tras escribir `experiment.lock.json` |

### 🎯 Qué vas a poder hacer al terminar

- Aprender predicción de ruido y muestreo iterativo sobre imágenes reales.
- Interpretar noise_mse, sample_diversity
- Aplicar sellado de test y reproducibilidad

### 🧩 Prerrequisitos

- CNN
- probabilidad
- modelos generativos

> Si alguno te falta, retrocede antes de continuar. Viniendo de [🖌️ WGAN-GP sobre Fashion-MNIST](../../advanced_labs/28_wgan_gp/README.md).

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

Aprender predicción de ruido y muestreo iterativo sobre imágenes reales.

## Dataset público real

- **Dataset:** `fashion_mnist`
- **Fuente:** Torchvision / Zalando Research
- **Licencia/condiciones:** MIT para código; consultar dataset
- **Entrada:** imágenes Fashion-MNIST normalizadas
- **Datos sintéticos:** no se usan.

Los datos se descargan desde el proveedor oficial mediante los adaptadores del repositorio. Los archivos grandes no se incluyen en Git.

## Modelo y fundamento

- **Modelo:** `tiny-ddpm`
- **Teoría:** Proceso directo de ruido, predicción de epsilon, cronograma beta y muestreo inverso.
- **Línea base:** Autoencoder generativo simple

## Protocolo

1. Crear particiones con `split_seed` o conservar los splits oficiales.
2. Entrenar y seleccionar exclusivamente con `train` y `validation`.
3. Guardar `best_model.pt` y escribir `experiment.lock.json`.
4. Abrir `test` una sola vez después del congelamiento.
5. Registrar métricas, configuración, procedencia y limitaciones.

## Ejecución

```bash
neural-labs train-advanced --track 29_diffusion_ddpm --quick
neural-labs train-advanced --track 29_diffusion_ddpm --split-seed 42 --training-seed 43
```

## Métricas

noise_mse, sample_diversity, sampling_latency, reconstruction_proxy.

## Cuadernos

- `notebook.ipynb`: recorrido completo.
- `notebook_student.ipynb`: actividades sin resolver.
- `notebook_solution.ipynb`: referencia docente.

## Limitación principal

El modelo pequeño sirve para estudio; no debe extrapolarse a generación fotográfica de alta resolución.

<!-- nav-bottom -->
## 🧭 Navegación del recorrido

| ⬅️ Laboratorio anterior | 🏠 Índice | Laboratorio siguiente ➡️ |
|---|:---:|---|
| [🖌️ WGAN-GP sobre Fashion-MNIST](../../advanced_labs/28_wgan_gp/README.md) | [Las 31 rutas](../../parts/README.md) | [🪞 Aprendizaje autosupervisado SimCLR](../../advanced_labs/30_self_supervised_simclr/README.md) |

**En este laboratorio:** **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🔬 [Parte 7 — Especializaciones avanzadas](../../parts/07-especializaciones-avanzadas.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/29_diffusion_ddpm/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
