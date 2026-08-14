# Optimizadores y schedulers

<!-- nav-top -->
> 🧭 **Ruta 19 / 31** · 🔴 [Parte 5 — La mecánica fina, ahora en profundidad](../../parts/05-mecanica-fina.md)
>
> [⬅️ 📐 Activaciones y funciones de pérdida](../../labs/17_activations_and_losses/README.md) · [🏠 Índice de rutas](../../parts/README.md) · [🛡️ Regularización ➡️](../../labs/19_regularization_dropout_batchnorm/README.md)
>
> **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

<!-- ficha -->
## 📋 Ficha del laboratorio

![ruta](https://img.shields.io/badge/ruta-19%20de%2031-7c5cff?style=flat-square) ![nivel](https://img.shields.io/badge/nivel-fundamentos-3fb950?style=flat-square) ![categoría](https://img.shields.io/badge/categoría-Central-2e8b57?style=flat-square) ![horas](https://img.shields.io/badge/horas-~4%20h-f0b429?style=flat-square) ![dataset](https://img.shields.io/badge/dataset-california__housing-1f6feb?style=flat-square) ![selección](https://img.shields.io/badge/selección-rmse-8957e5?style=flat-square)

| Campo | Valor |
|---|---|
| 🧭 Posición | Ruta **19 de 31** del recorrido · categoría central |
| 🎚️ Nivel | fundamentos |
| ⏱️ Dedicación estimada | 4 horas |
| 🧩 Tarea | `regression` |
| 🏗️ Arquitectura | `optimizer_comparison` |
| 🗄️ Dataset | [`california_housing`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) — scikit-learn / StatLib |
| ⚖️ Licencia del dataset | Consultar fuente StatLib |
| 🎯 Métrica de selección | `rmse` sobre `validation` |
| 📏 Línea base a superar | Media y Ridge |
| 🔒 Política de `test` | se abre una sola vez, tras escribir `experiment.lock.json` |

### 🎯 Qué vas a poder hacer al terminar

- Comparar SGD, Momentum, Adam y reducción de tasa de aprendizaje.
- Preparar y auditar el dataset real california_housing sin fuga de datos.
- Entrenar y evaluar comparación controlada de optimizadores y schedulers.
- Comparar contra la línea base: Media y Ridge.
- Interpretar intervalos de confianza, errores y limitaciones.

### 🧩 Prerrequisitos

- Python básico
- NumPy
- álgebra lineal elemental

> Si alguno te falta, retrocede antes de continuar. Viniendo de [📐 Activaciones y funciones de pérdida](../../labs/17_activations_and_losses/README.md).

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

<!-- guia -->
## 🎯 Qué vas a hacer aquí

Comparar SGD, Momentum, Adam y reducción de tasa de aprendizaje.

Es la **ruta 19 de 31** y pertenece a 🔴 [la parte 5, La mecánica fina, ahora en profundidad](../../parts/05-mecanica-fina.md). Llegas desde [📐 Activaciones y funciones de pérdida](../../labs/17_activations_and_losses/README.md) y lo que aprendas aquí lo da por supuesto [🛡️ Regularización](../../labs/19_regularization_dropout_batchnorm/README.md).

## 🧠 La idea que se pone a prueba

Este laboratorio trabaja **Actualizaciones de parámetros y programación de learning rate**.

El desarrollo completo —qué calcula cada parte, de dónde sale la fórmula, qué riesgos tiene interpretarla mal y en qué libros y papers se estudia— está en [`theory.md`](theory.md). Léelo antes de entrenar: los pasos de abajo te dicen *qué* hacer, y la teoría, *por qué* funciona y cuándo deja de funcionar.

> **La pregunta que deberías poder responder al final:** ¿Cuál mejora más rápido y cuál generaliza mejor?

**Métricas que se reportan:** `mae`, `rmse`, `r2`. La selección del modelo se decide con `rmse` sobre `validation`.

## 🪜 Paso a paso

Cada paso dice qué ocurre, por qué se hace así y cómo comprobar que salió bien. El orden no es una convención: es el que ejecuta el código, y cambiarlo rompe la validez del resultado.

### Paso 1 — Traer el dataset real y partirlo

**Qué ocurre.** Descarga `california_housing` desde su proveedor y construye las tres particiones —`train`, `validation` y `test`— con la semilla de partición que le pases.

**Por qué.** La partición se fija **antes** de entrenar y su semilla (`--split-seed`) es independiente de la del entrenamiento (`--training-seed`). Si ambas fueran la misma, no podrías distinguir si un cambio en el resultado viene de haber repartido los datos de otro modo o de haber inicializado los pesos de otro modo.

```bash
neural-labs dataset --lab 18_optimizers_and_schedulers --quick --split-seed 42
```

**Cómo sabes que salió bien.** El comando termina sin error y deja el dataset en caché. Si no hay red, **falla**: no se rellena con datos inventados (`data/dataset.yaml` declara `fallback_to_generated_data: false`).

### Paso 2 — Comprobar que las particiones no se tocan

**Qué ocurre.** Recorre los identificadores de las tres particiones y verifica que ninguno aparece en dos de ellas.

**Por qué.** Una sola fila compartida entre `train` y `test` basta para que la métrica final quede inflada. Es el error más caro del oficio porque no produce ningún síntoma visible: el modelo simplemente «parece» mejor de lo que es.

```bash
neural-labs audit --lab 18_optimizers_and_schedulers --quick --split-seed 42
```

**Cómo sabes que salió bien.** La auditoría reporta cero solapamientos. Si reporta alguno, no sigas: el resto del laboratorio no significaría nada.

### Paso 3 — Mirar los datos antes de modelarlos

**Qué ocurre.** Genera el informe de calidad (valores faltantes, balance de clases, rangos) y el de deriva entre particiones.

**Por qué.** Un desbalance fuerte o una diferencia de distribución entre `train` y `test` cambia qué métrica tiene sentido leer. Descubrirlo después de entrenar obliga a repetirlo todo.

```bash
neural-labs quality --lab 18_optimizers_and_schedulers --quick --split-seed 42
```

**Cómo sabes que salió bien.** Obtienes `data_quality.json` y `drift_report.json`; ábrelos antes de decidir la configuración.

### Paso 4 — Estudiar la teoría del laboratorio

**Qué ocurre.** Leer [`theory.md`](theory.md): la idea central, el desarrollo matemático, los riesgos de interpretación y la bibliografía de la que sale todo eso.

**Por qué.** Sin esto, el entrenamiento es una caja que devuelve números. La teoría es lo que te permite decidir qué mirar y reconocer cuándo un resultado es sospechoso.

**Cómo sabes que salió bien.** Puedes responder, con tus palabras, qué calcula el modelo y por qué esa arquitectura encaja con la tarea `regression`.

### Paso 5 — Entrenar y seleccionar con `validation`

**Qué ocurre.** El entrenamiento recorre las épocas midiendo en `validation` después de cada una, y conserva el checkpoint con el mejor valor de `rmse`.

**Por qué.** El conjunto de validación existe para tomar decisiones —arquitectura, hiperparámetros, cuándo parar—. Si esas decisiones se tomaran mirando `test`, `test` dejaría de ser una estimación de lo que pasará con datos nuevos y pasaría a ser parte del entrenamiento.

```bash
python labs/18_optimizers_and_schedulers/train.py --quick
# o, con control explícito de las dos semillas:
neural-labs train --lab 18_optimizers_and_schedulers --config baseline --split-seed 42 --training-seed 43
```

**Cómo sabes que salió bien.** En `runs/18_optimizers_and_schedulers/<ejecución>/` aparecen `history.csv` y `best_model.pt`; la métrica de validación mejora respecto de la primera época.

### Paso 6 — Compararte con la línea base

**Qué ocurre.** El repositorio entrena por su cuenta **Media y Ridge** y guarda su resultado, primero sobre `validation` y —solo al final— sobre `test`.

**Por qué.** Una métrica sola no dice si el modelo aporta algo. Puede que un método mucho más simple llegue igual de lejos, y entonces la complejidad añadida no está justificada. Esta comparación es la que convierte un número en un argumento.

**Cómo sabes que salió bien.** Comparas `metrics.json` con `baseline_metrics.json`. Si tu modelo no supera la línea base, el resultado del laboratorio es exactamente ese, y hay que reportarlo.

### Paso 7 — El sellado: `experiment.lock.json`

**Qué ocurre.** Antes de tocar `test`, el código escribe un archivo que fija el laboratorio, las dos semillas, la configuración, la métrica de selección, el checkpoint elegido y el hash del dataset.

**Por qué.** Es la frontera del experimento. A partir de ahí, cualquier ajuste que hagas mirando `test` queda a la vista: el sello dice qué habías decidido *antes* de ver el resultado final. Sin ese archivo, nadie —incluido tú dentro de un mes— puede distinguir una predicción de una racionalización.

**Cómo sabes que salió bien.** El archivo existe y su contenido coincide con lo que creías haber ejecutado.

### Paso 8 — Evaluar `test` una sola vez y medir la incertidumbre

**Qué ocurre.** Con el checkpoint congelado se evalúa `test`, se calculan intervalos de confianza por bootstrap y se desglosan las métricas por subgrupo.

**Por qué.** Un número puntual esconde cuánto podría moverse. Los intervalos dicen si la diferencia con la línea base es real o cabe dentro del ruido; el desglose por subgrupo revela si el promedio está tapando un grupo donde el modelo funciona mucho peor.

**Cómo sabes que salió bien.** Tienes `metrics.json`, `confidence_intervals.json` y `subgroup_metrics.json`, y puedes decir la magnitud de la mejora **y** su incertidumbre.

### Paso 9 — Repetir con varias semillas de entrenamiento

**Qué ocurre.** Se repite el entrenamiento manteniendo **fija** la partición y cambiando solo la semilla de entrenamiento.

**Por qué.** Dos ejecuciones idénticas salvo por la inicialización pueden diferir bastante. Si no mides esa dispersión, corres el riesgo de celebrar una mejora que era una semilla afortunada.

```bash
neural-labs benchmark --lab 18_optimizers_and_schedulers --quick --split-seed 42 --training-seeds 41 42 43
```

**Cómo sabes que salió bien.** Obtienes media y dispersión entre semillas, no un único número.

### Paso 10 — Documentar y cerrar

**Qué ocurre.** Cada ejecución deja `model_card.md` y `report.md`; el plan de experimentos vive en [`experiments.md`](experiments.md) y la rúbrica en [`assessment.md`](assessment.md).

**Por qué.** Un resultado sin su contexto —qué datos, qué decisiones, qué límites— no es reutilizable. La model card es lo que permite que otra persona sepa cuándo *no* debería usar tu modelo.

**Cómo sabes que salió bien.** Completaste la tabla multi-semilla de `experiments.md` y respondiste las preguntas de `assessment.md`.

## 🔍 Cómo leer lo que produce la ejecución

Cada ejecución escribe su propio directorio. Estos son los archivos que encontrarás y para qué sirve cada uno:

| Archivo | Qué contiene y qué mirar |
|---|---|
| `config.yaml` · `environment.json` | La configuración exacta y el entorno (versiones, dispositivo) de la ejecución. |
| `dataset_manifest.json` · `dataset_card.md` | Procedencia, licencia, hash y tamaño de cada partición. |
| `data_quality.json` · `drift_report.json` | Calidad de los datos y diferencias de distribución entre particiones. |
| `baseline_validation_metrics.json` | La línea base medida en `validation`, **antes** de entrenar. |
| `history.csv` · `history.png` | Pérdida y métricas época a época: aquí se ve si el entrenamiento converge o sobreajusta. |
| `experiment.lock.json` | El sello del experimento, escrito antes de tocar `test`. |
| `metrics.json` | El resultado final en `test`, más tiempo, dispositivo y número de parámetros. |
| `baseline_metrics.json` | La línea base en `test`, calculada después de tu evaluación final. |
| `best_model.pt` · `last_model.pt` | El checkpoint elegido por validación y el último, para poder compararlos. |
| `confidence_intervals.json` | Intervalos por bootstrap: cuánto podría moverse cada métrica. |
| `subgroup_metrics.json` | El mismo resultado desglosado por subgrupo, para ver qué esconde el promedio. |
| `predictions.csv` | Predicción por ejemplo, para analizar los errores uno a uno. |
| `model_spec.json` · `inference_contract.json` | Qué entrada espera el modelo y qué devuelve: lo que necesita quien lo despliegue. |
| `model_card.md` · `report.md` | La ficha del modelo y el informe legible de la ejecución. |
| `variant_comparison.json` | **Propio de esta ruta.** Una fila por variante comparada, con su métrica de validación. |

## ⚠️ Dónde suele perderse la gente

- **`--quick` no es una versión pequeña del resultado, es una prueba de que todo corre.** En esta ruta recorta a 1024 ejemplos de entrenamiento · 256 de validación · 256 de test · 2 épocas. Sirve para comprobar la instalación y la descarga; cualquier conclusión sobre el modelo exige la ejecución completa.
- **Cambiar algo después de ver `test` invalida la comparación.** Si al mirar el resultado final se te ocurre una mejora, la ruta correcta es volver a `validation`, decidir allí, y sellar de nuevo.
- **Las dos semillas no son intercambiables.** `--split-seed` cambia *qué datos* caen en cada partición; `--training-seed` cambia *cómo se inicializa y baraja* el entrenamiento. Para comparar modelos se fija la primera y se varía la segunda.
- **No hay `confusion_matrix.png`, y no es un error.** Es una tarea de regresión: no existen clases que confundir.
- **Límite declarado de este dataset.** Datos reales del censo de California de 1990.

## ✅ Antes de darlo por terminado

El laboratorio está aprobado cuando se cumplen estos criterios:

- [ ] cero solapamiento entre train, validation y test
- [ ] selección basada únicamente en validation
- [ ] métricas finales acompañadas por incertidumbre
- [ ] conclusiones que distinguen evidencia de suposición

Y cuando tienes estos entregables:

- [ ] notebook ejecutado
- [ ] reporte experimental
- [ ] model card
- [ ] comparación con línea base
- [ ] respuesta a preguntas críticas

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
| Objetivo, línea base, métricas y arquitectura | [`configs/labs.yaml`](../../configs/labs.yaml) |
| Fuente, licencia, procedencia y límites del dataset | [`data/dataset.yaml`](data/dataset.yaml) |
| Épocas, tamaño de lote, tasa de aprendizaje y recorte de `--quick` | [`configs/baseline.yaml`](configs/baseline.yaml) · [`configs/improved.yaml`](configs/improved.yaml) |
| Nivel, prerrequisitos, resultados de aprendizaje y criterios | [`lesson.yaml`](lesson.yaml) |
| El orden de los pasos y los archivos que escribe cada ejecución | [`src/neural_labs/experiments.py`](../../src/neural_labs/experiments.py) |
| La teoría, los papers y los libros de referencia | [`theory.md`](theory.md), sección 🔗 Referencias |
| La regla general del protocolo | [`docs/experiment-protocol.md`](../../docs/experiment-protocol.md) |

Los datasets se descargan de su proveedor original y conservan su propia licencia; este repositorio no los redistribuye ni sustituye una descarga fallida por datos generados.
<!-- /guia -->

<!-- nav-bottom -->
## 🧭 Navegación del recorrido

| ⬅️ Laboratorio anterior | 🏠 Índice | Laboratorio siguiente ➡️ |
|---|:---:|---|
| [📐 Activaciones y funciones de pérdida](../../labs/17_activations_and_losses/README.md) | [Las 31 rutas](../../parts/README.md) | [🛡️ Regularización](../../labs/19_regularization_dropout_batchnorm/README.md) |

**En este laboratorio:** **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🔴 [Parte 5 — La mecánica fina, ahora en profundidad](../../parts/05-mecanica-fina.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/18_optimizers_and_schedulers/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
