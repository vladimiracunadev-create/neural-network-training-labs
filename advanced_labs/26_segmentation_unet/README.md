# Segmentación semántica con U-Net

<!-- nav-top -->
> 🧭 **Ruta 27 / 31** · 🔬 [Parte 7 — Especializaciones avanzadas](../../parts/07-especializaciones-avanzadas.md)
>
> [⬅️ 🔧 Fine-tuning eficiente de transformer](../../advanced_labs/25_transformer_finetuning/README.md) · [🏠 Índice de rutas](../../parts/README.md) · [🎙️ Clasificación de audio con SpeechCommands ➡️](../../advanced_labs/27_audio_speechcommands/README.md)
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
| 📏 Línea base a superar | Máscara de clase mayoritaria |
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

<!-- guia -->
## 🎯 Qué vas a hacer aquí

Segmentar mascota, fondo y contorno con IoU por clase.

Es la **ruta 27 de 31** y pertenece a 🔬 [la parte 7, Especializaciones avanzadas](../../parts/07-especializaciones-avanzadas.md). Llegas desde [🔧 Fine-tuning eficiente de transformer](../../advanced_labs/25_transformer_finetuning/README.md) y lo que aprendas aquí lo da por supuesto [🎙️ Clasificación de audio con SpeechCommands](../../advanced_labs/27_audio_speechcommands/README.md).

**Entrada del modelo:** imagen RGB y máscara trimap.

## 🧠 La idea que se pone a prueba

Este laboratorio trabaja **Arquitectura encoder-decoder, conexiones skip, pérdida por píxel e intersección sobre unión**.

El desarrollo completo —qué calcula cada parte, de dónde sale la fórmula, qué riesgos tiene interpretarla mal y en qué libros y papers se estudia— está en [`theory.md`](theory.md). Léelo antes de entrenar: los pasos de abajo te dicen *qué* hacer, y la teoría, *por qué* funciona y cuándo deja de funcionar.

**Métricas que se reportan:** `mean_iou`, `iou_per_class`, `pixel_accuracy`, `dice`. La selección del modelo se decide con `mean_iou` sobre `validation`.

## 🪜 Paso a paso

Cada paso dice qué ocurre, por qué se hace así y cómo comprobar que salió bien. El orden no es una convención: es el que ejecuta el código, y cambiarlo rompe la validez del resultado.

### Paso 1 — Estudiar la teoría antes de ejecutar nada

**Qué ocurre.** Leer [`theory.md`](theory.md), que desarrolla Arquitectura encoder-decoder, conexiones skip, pérdida por píxel e intersección sobre unión. y cita las obras y papers de los que procede.

**Por qué.** Estas rutas usan arquitecturas donde un error de comprensión no se manifiesta como un fallo, sino como un número plausible pero equivocado.

**Cómo sabes que salió bien.** Puedes explicar qué mide `mean_iou` y por qué es la métrica de selección aquí.

### Paso 2 — Ejecutar la versión rápida

**Qué ocurre.** Descarga el dataset y los pesos preentrenados desde su proveedor, entrena una versión reducida y escribe la ejecución en `runs-advanced/`.

**Por qué.** Antes de gastar horas de cómputo conviene comprobar que la descarga, el entorno y la ruta completa funcionan de extremo a extremo.

```bash
neural-labs train-advanced --track 26_segmentation_unet --quick
```

**Cómo sabes que salió bien.** Termina sin error y deja `metrics.json`, `history.json` y `best_model.pt` en el directorio de la ejecución.

### Paso 3 — Entrenar en serio y seleccionar con `validation`

**Qué ocurre.** Se entrena el modelo completo conservando el checkpoint con el mejor valor de `mean_iou` en validación, y se sella el experimento antes de evaluar `test`.

**Por qué.** Igual que en las rutas centrales: `validation` decide, `test` solo confirma, y el sello deja por escrito qué se había decidido antes de mirar.

```bash
neural-labs train-advanced --track 26_segmentation_unet --split-seed 42 --training-seed 43
```

**Cómo sabes que salió bien.** Existe `experiment.lock.json` y `metrics.json` incluye tanto el valor de validación como el de test.

### Paso 4 — Repetir con otra semilla de entrenamiento

**Qué ocurre.** Se repite el entrenamiento con la misma partición y distinta semilla de entrenamiento.

**Por qué.** Estas arquitecturas —adversariales, contrastivas, de difusión— son especialmente sensibles a la inicialización: una sola ejecución no permite distinguir una mejora de una casualidad.

```bash
neural-labs train-advanced --track 26_segmentation_unet --split-seed 42 --training-seed 44
```

**Cómo sabes que salió bien.** Puedes reportar el rango entre ejecuciones, no un único número.

### Paso 5 — Documentar los límites

**Qué ocurre.** Registrar el resultado junto con la limitación declarada de la ruta y responder [`assessment.md`](assessment.md).

**Por qué.** En generación y aprendizaje autosupervisado las métricas son aproximaciones: sin declarar qué NO demuestran, invitan a conclusiones que los números no sostienen.

**Cómo sabes que salió bien.** Tu reporte dice qué mejoró, cuánto costó y en qué condiciones no esperarías el mismo resultado.

## 🔍 Cómo leer lo que produce la ejecución

Cada ejecución escribe su propio directorio. Estos son los archivos que encontrarás y para qué sirve cada uno:

| Archivo | Qué contiene y qué mirar |
|---|---|
| `config.json` | Track, semillas, dispositivo y opciones con las que se lanzó. |
| `dataset_manifest.json` | Fuente, licencia y número de ejemplos por partición. |
| `best_model.pt` | El checkpoint seleccionado por validación. |
| `experiment.lock.json` | El sello: qué se decidió antes de abrir `test`. |
| `history.json` | La métrica de validación época a época. |
| `metrics.json` | Resultado de validación y de test, ya con el modelo congelado. |

## ⚠️ Dónde suele perderse la gente

- **Cambiar algo después de ver `test` invalida la comparación.** Si al mirar el resultado final se te ocurre una mejora, la ruta correcta es volver a `validation`, decidir allí, y sellar de nuevo.
- **Las dos semillas no son intercambiables.** `--split-seed` cambia *qué datos* caen en cada partición; `--training-seed` cambia *cómo se inicializa y baraja* el entrenamiento. Para comparar modelos se fija la primera y se varía la segunda.
- **Límite declarado de este dataset.** Las imágenes se concentran en mascotas y fondos cotidianos; no generaliza a segmentación médica o industrial.

## ✅ Antes de darlo por terminado

Y cuando tienes estos entregables:

- [ ] notebook ejecutado
- [ ] reporte experimental
- [ ] model card

Las preguntas y la rúbrica con la que se corrige están en [`assessment.md`](assessment.md); el plan de experimentos y la tabla multi-semilla que hay que completar, en [`experiments.md`](experiments.md).

## 🧪 Para ir más lejos

- Cambia una decisión experimental y justifícala con el resultado en `validation`, no con la intuición.
- Analiza los errores por clase o por segmento: casi siempre se concentran en un subconjunto reconocible.
- Compara costo, precisión y latencia; el mejor modelo no siempre es el que gana por décimas.
- Documenta sesgos, limitaciones y usos para los que **no** recomendarías este modelo.

## 📚 De dónde sale cada cosa de esta guía

Nada de lo anterior está escrito de memoria. Cada afirmación se puede comprobar en un archivo concreto del repositorio:

| Lo que dice la guía | Dónde comprobarlo |
|---|---|
| Objetivo, línea base, métricas y arquitectura | [`configs/advanced_tracks.yaml`](../../configs/advanced_tracks.yaml) |
| Fuente, licencia, procedencia y límites del dataset | [`data/dataset.yaml`](data/dataset.yaml) |
| Épocas, tamaño de lote, tasa de aprendizaje y recorte de `--quick` | [`configs/baseline.yaml`](configs/baseline.yaml) · [`configs/improved.yaml`](configs/improved.yaml) |
| Nivel, prerrequisitos, resultados de aprendizaje y criterios | [`lesson.yaml`](lesson.yaml) |
| El orden de los pasos y los archivos que escribe cada ejecución | [`src/neural_labs/advanced/training.py`](../../src/neural_labs/advanced/training.py) |
| La teoría, los papers y los libros de referencia | [`theory.md`](theory.md), sección 🔗 Referencias |
| La regla general del protocolo | [`docs/experiment-protocol.md`](../../docs/experiment-protocol.md) |

Los datasets se descargan de su proveedor original y conservan su propia licencia; este repositorio no los redistribuye ni sustituye una descarga fallida por datos generados.
<!-- /guia -->

<!-- nav-bottom -->
## 🧭 Navegación del recorrido

| ⬅️ Laboratorio anterior | 🏠 Índice | Laboratorio siguiente ➡️ |
|---|:---:|---|
| [🔧 Fine-tuning eficiente de transformer](../../advanced_labs/25_transformer_finetuning/README.md) | [Las 31 rutas](../../parts/README.md) | [🎙️ Clasificación de audio con SpeechCommands](../../advanced_labs/27_audio_speechcommands/README.md) |

**En este laboratorio:** **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🔬 [Parte 7 — Especializaciones avanzadas](../../parts/07-especializaciones-avanzadas.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/26_segmentation_unet/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
