# Aprendizaje autosupervisado SimCLR

<!-- nav-top -->
> 🧭 **Ruta 31 / 31** · 🔬 [Parte 7 — Especializaciones avanzadas](../../parts/07-especializaciones-avanzadas.md)
>
> [⬅️ 🌫️ Difusión DDPM sobre Fashion-MNIST](../../advanced_labs/29_diffusion_ddpm/README.md) · [🏠 Índice de rutas](../../parts/README.md) · *fin del recorrido* ➡️
>
> **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

<!-- ficha -->
## 📋 Ficha del laboratorio

![ruta](https://img.shields.io/badge/ruta-31%20de%2031-7c5cff?style=flat-square) ![nivel](https://img.shields.io/badge/nivel-avanzado-8957e5?style=flat-square) ![categoría](https://img.shields.io/badge/categoría-Avanzada-2e8b57?style=flat-square) ![dataset](https://img.shields.io/badge/dataset-cifar10-1f6feb?style=flat-square) ![selección](https://img.shields.io/badge/selección-nt__xent-8957e5?style=flat-square)

| Campo | Valor |
|---|---|
| 🧭 Posición | Ruta **31 de 31** del recorrido · categoría avanzada |
| 🎚️ Nivel | avanzado |
| 🗺️ Dominio | `vision` |
| 🏗️ Arquitectura | `resnet18-simclr` |
| 🗄️ Dataset | `cifar10` — Torchvision / University of Toronto |
| ⚖️ Licencia del dataset | Consultar términos CIFAR-10 |
| 🎯 Métrica de selección | `nt_xent` sobre `validation` |
| 🔒 Política de `test` | se abre una sola vez, tras escribir `experiment.lock.json` |

### 🎯 Qué vas a poder hacer al terminar

- Preentrenar representaciones con dos vistas reales y evaluar mediante linear probe.
- Interpretar nt_xent, linear_probe_accuracy
- Aplicar sellado de test y reproducibilidad

### 🧩 Prerrequisitos

- CNN
- embeddings
- aprendizaje contrastivo

> Si alguno te falta, retrocede antes de continuar. Viniendo de [🌫️ Difusión DDPM sobre Fashion-MNIST](../../advanced_labs/29_diffusion_ddpm/README.md).

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

Preentrenar representaciones con dos vistas reales y evaluar mediante linear probe.

## Dataset público real

- **Dataset:** `cifar10`
- **Fuente:** Torchvision / University of Toronto
- **Licencia/condiciones:** Consultar términos CIFAR-10
- **Entrada:** imágenes CIFAR-10
- **Datos sintéticos:** no se usan.

Los datos se descargan desde el proveedor oficial mediante los adaptadores del repositorio. Los archivos grandes no se incluyen en Git.

## Modelo y fundamento

- **Modelo:** `resnet18-simclr`
- **Teoría:** Dos vistas, similitud coseno, pérdida NT-Xent y evaluación linear probe.
- **Línea base:** ResNet18 aleatoria + linear probe

## Protocolo

1. Crear particiones con `split_seed` o conservar los splits oficiales.
2. Entrenar y seleccionar exclusivamente con `train` y `validation`.
3. Guardar `best_model.pt` y escribir `experiment.lock.json`.
4. Abrir `test` una sola vez después del congelamiento.
5. Registrar métricas, configuración, procedencia y limitaciones.

## Ejecución

```bash
neural-labs train-advanced --track 30_self_supervised_simclr --quick
neural-labs train-advanced --track 30_self_supervised_simclr --split-seed 42 --training-seed 43
```

## Métricas

nt_xent, linear_probe_accuracy, knn_accuracy, embedding_uniformity.

## Cuadernos

- `notebook.ipynb`: recorrido completo.
- `notebook_student.ipynb`: actividades sin resolver.
- `notebook_solution.ipynb`: referencia docente.

## Limitación principal

La elección de aumentos define invariancias y puede borrar información relevante para tareas posteriores.

<!-- nav-bottom -->
## 🧭 Navegación del recorrido

| ⬅️ Laboratorio anterior | 🏠 Índice | Laboratorio siguiente ➡️ |
|---|:---:|---|
| [🌫️ Difusión DDPM sobre Fashion-MNIST](../../advanced_labs/29_diffusion_ddpm/README.md) | [Las 31 rutas](../../parts/README.md) | *— fin del recorrido* |

**En este laboratorio:** **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🔬 [Parte 7 — Especializaciones avanzadas](../../parts/07-especializaciones-avanzadas.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/30_self_supervised_simclr/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
