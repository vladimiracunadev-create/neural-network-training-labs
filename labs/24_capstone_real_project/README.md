# Proyecto final: churn de telecomunicaciones

<!-- nav-top -->
> 🧭 **Ruta 25 / 31** · ⚫ [Parte 6 — Confiar en el modelo y sacarlo del cuaderno](../../parts/06-confianza-y-despliegue.md)
>
> [⬅️ 📦 Exportación e inferencia](../../labs/23_model_export_and_inference/README.md) · [🏠 Índice de rutas](../../parts/README.md) · [🔧 Fine-tuning eficiente de transformer ➡️](../../advanced_labs/25_transformer_finetuning/README.md)
>
> **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

## 🎯 Qué vas a hacer aquí

Resolver de extremo a extremo un problema real de abandono de clientes con documentación, evaluación y despliegue.

Es la **ruta 25 de 31** del recorrido y pertenece a ⚫ la parte 6, *Confiar en el modelo y sacarlo del cuaderno*. Llegas desde **Exportación e inferencia** y lo que hagas aquí lo da por supuesto **Fine-tuning eficiente de transformer**.

Trabajarás con el dataset **`iranian_churn`** (UCI, licencia: CC BY 4.0), y tendrás que superar la línea base **Regresión logística y Gradient Boosting**, decidiendo con la métrica `f1` medida sobre `validation`. Nivel proyecto, unas **10 horas** de dedicación.

**Al terminar deberías ser capaz de:**

- Resolver de extremo a extremo un problema real de abandono de clientes con documentación, evaluación y despliegue.
- Preparar y auditar el dataset real iranian_churn sin fuga de datos.
- Entrenar y evaluar proyecto integral de churn.
- Comparar contra la línea base: Regresión logística y Gradient Boosting.
- Interpretar intervalos de confianza, errores y limitaciones.

## 🧠 La teoría de este laboratorio

Esta sección es la explicación completa del tema. No hace falta abrir otro archivo para entender lo que viene después: aquí está la idea, la matemática que la sostiene y sus límites. (El mismo texto vive en `theory.md`, que es la fuente desde la que se genera esta guía, junto con la bibliografía del final.)

### De qué trata

Este laboratorio estudia **proyecto integral de churn** usando `iranian_churn`, un dataset público real procedente de UCI.

El *churn* —abandono de clientes— es un problema de negocio antes que un problema de aprendizaje: una operadora quiere anticipar qué clientes dejarán el servicio para intervenir con retención. Traducirlo a un modelo obliga a recorrer todo el ciclo de vida de un proyecto de ML: entender los datos y su procedencia, definir la métrica que importa, construir líneas base honestas, entrenar y calibrar, elegir un umbral de decisión ligado al *costo* de los errores, y documentar el resultado para que sea auditable y desplegable. Este capstone integra todo lo aprendido en los laboratorios anteriores sobre un caso real: 3.150 clientes de una empresa iraní de telecomunicaciones seguidos durante 12 meses.

Lo distintivo de un proyecto end-to-end es que la exactitud bruta rara vez es la meta. El churn es un problema **desbalanceado** (los que se van son minoría) y con **costos asimétricos**: no cuesta lo mismo dejar escapar a un cliente que se iba (falso negativo, se pierde su valor) que ofrecer una promoción a alguien que se quedaba igual (falso positivo, gasto innecesario). Por eso el laboratorio insiste en métricas sensibles al desbalance, en calibración de probabilidades y en la selección de umbral como decisión de negocio, comparando siempre contra líneas base sólidas: regresión logística y gradient boosting.

### La matemática, paso a paso

Clasificación, calibración, selección de umbral y costo de errores.

El modelo produce una probabilidad de abandono p = P(churn | x) para cada cliente, pero la *decisión* de actuar requiere un **umbral** τ: se interviene si p ≥ τ. La elección de τ no es un detalle técnico, sino donde entra la economía del problema. Cada resultado tiene un costo: un verdadero positivo detectado permite una acción de retención; un falso negativo (τ demasiado alto) deja escapar clientes; un falso positivo (τ demasiado bajo) malgasta recursos. Si asignamos costos c_FN y c_FP a cada tipo de error, el umbral óptimo minimiza el costo esperado y, bajo el análisis clásico, satisface una relación de la forma τ\* = c_FP / (c_FP + c_FN): cuanto más caro es dejar escapar a un cliente (c_FN grande), más bajo conviene poner el umbral para capturar a más candidatos. Este umbral se elige en **validación**, nunca en test.

Como el umbral depende de la probabilidad, esa probabilidad debe ser **confiable**, y aquí reaparece la calibración del laboratorio 22: un modelo sobreconfiado desplaza el punto de operación y distorsiona el análisis de costos. Por eso, antes de fijar τ, conviene recalibrar (p. ej. con temperature scaling o Platt scaling) para que p ≈ frecuencia real de churn. Para evaluar el modelo con independencia del umbral se usan métricas basadas en el *ranking*: el **ROC-AUC** mide la probabilidad de ordenar correctamente un par (cliente que abandona, cliente que se queda), mientras que el **PR-AUC** (precisión–recall) es más informativo en datos desbalanceados porque se centra en la clase positiva minoritaria y no se deja "inflar" por la abundancia de negativos. La **balanced accuracy** —media de la sensibilidad y la especificidad— corrige el sesgo de la exactitud simple cuando las clases están desequilibradas.

La cadena de razonamiento del capstone es, entonces: entrenar un clasificador → recalibrar sus probabilidades → medir su capacidad de ordenamiento con AUC/PR-AUC de forma independiente del umbral → traducir esa capacidad en una política de decisión eligiendo τ según los costos del negocio en validación → y solo entonces reportar el desempeño una única vez en test. La contribución matemática no está en un algoritmo nuevo, sino en *encadenar correctamente* clasificación, calibración, coste y umbral para que el número final sea una decisión responsable y no una métrica aislada.

### El problema de negocio no es el problema de aprendizaje

El paso que este proyecto final exige y que ningún laboratorio anterior pedía es la **traducción**: convertir «reducir el abandono de clientes» en una tarea que un modelo pueda optimizar. Esa traducción implica decisiones que no son técnicas y que determinan el resultado más que la arquitectura.

Hay que fijar tres cosas antes de tocar los datos. **Qué se predice**: definir abandono exige un criterio —¿cuántos días sin actividad?, ¿la baja formal?— y criterios distintos producen conjuntos de etiquetas distintos. **Cuándo se predice**: la ventana de observación y el horizonte, es decir, con qué información se cuenta en el momento de decidir y con cuánta antelación hay que avisar. Y **para qué**: si la salida alimenta una campaña de retención con presupuesto para N clientes, el modelo no necesita clasificar bien a todos, necesita **ordenar bien los N primeros**, que es un objetivo distinto.

De ahí sale la métrica correcta, y no al revés. Con un presupuesto de retención fijo, lo que importa es la **precisión en los k primeros** y la ganancia acumulada de los primeros deciles, no la exactitud global. Elegir la métrica después de ver los resultados es la forma más común de autoengaño en un proyecto aplicado; declararla antes, junto con el umbral y su justificación económica, es lo que el sellado del experimento hace explícito.

### La fuga temporal en datos de clientes

En un dataset tabular de clientes, la fuga no viene de mezclar particiones sino del **contenido de las variables**, y es mucho más difícil de detectar.

Una variable produce fuga cuando su valor se conoce **después** o **a causa** del hecho que se quiere predecir. En abandono, los ejemplos clásicos son campos de baja, motivos de cancelación, o el consumo del último mes cuando ese mes ya es posterior al momento de decisión. También cuentan las variables agregadas calculadas sobre todo el histórico —un promedio que incluye el periodo objetivo—, y los identificadores que correlacionan con la etiqueta por el orden en que se cargaron los datos.

El síntoma es siempre el mismo y hay que aprender a desconfiar de él: una métrica **sospechosamente alta**. Un modelo de abandono con AUC de 0,99 casi nunca es un gran modelo; casi siempre es una fuga. El diagnóstico consiste en mirar la importancia de las variables, encontrar la que domina, y preguntarse si estaría disponible en el momento real de la predicción. Es la razón de que este proyecto exija la ruta 21 como herramienta de auditoría y no solo como capítulo de interpretabilidad.

La regla operativa que resume todo: para cada variable, responder **en qué instante se conoce su valor**. Si la respuesta es «después del corte de decisión», la variable no puede usarse, por informativa que sea.

### De la probabilidad a la decisión

El modelo entrega una probabilidad; el negocio necesita una acción. El puente es el umbral, y fijarlo es la decisión con más impacto económico de todo el proyecto.

Si retener a un cliente cuesta c_int y perderlo cuesta c_perd, y la intervención tiene una eficacia e —la fracción de clientes en riesgo que efectivamente se retienen—, el valor esperado de intervenir sobre un cliente con probabilidad p̂ es e·p̂·c_perd − c_int, de modo que conviene actuar cuando

p̂ > c_int / (e · c_perd).

La fórmula tiene tres consecuencias que conviene declarar. Primero, **el umbral no es 0,5** salvo por coincidencia. Segundo, exige que p̂ sea una probabilidad de verdad, lo que enlaza directamente con la calibración de la ruta 22. Y tercero, cuando el presupuesto es limitado, la restricción no es un umbral sino una capacidad: se interviene sobre los k clientes de mayor p̂, y lo que hay que medir es cuántos de ellos habrían abandonado realmente.

Al reportar el impacto conviene separar dos cifras que suelen mezclarse. El desempeño del **modelo** —discriminación, calibración, estabilidad entre semillas— se mide con datos históricos. El impacto de la **intervención** —cuántas bajas se evitaron— no se puede estimar con datos observacionales, porque requiere saber qué habría pasado sin actuar: eso exige un experimento con grupo de control. Presentar el segundo como si se dedujera del primero es un error que este proyecto pide evitar explícitamente en su reporte.

### Lo que hay que dejar escrito para que el trabajo sirva

Un proyecto de extremo a extremo termina cuando otra persona puede tomarlo, entenderlo y decidir sobre él. Eso exige, además de las métricas, cinco cosas que la model card debe contener.

**Uso previsto y usos desaconsejados**, en una frase cada uno. **Población de entrenamiento**: de dónde salen los datos, de qué periodo, y en qué se diferencia de la población donde se aplicará. **Desempeño por subgrupo**, no solo agregado: un modelo que funciona bien en promedio y mal en un segmento concreto es un problema operativo y, según el dominio, también legal. **Condiciones de revisión**: cada cuánto se vigila la deriva, qué degradación dispara un reentrenamiento, quién decide. Y **límites conocidos**: qué no se validó, qué supuestos podrían romperse.

Esa última parte es la que distingue un trabajo terminado de uno abandonado. Un modelo de abandono entrenado con un histórico concreto asume que el comportamiento de los clientes, la competencia y la oferta comercial siguen siendo los de entonces. Cuando cambian —y cambian— el modelo se degrada sin avisar. Declararlo por escrito, con el mecanismo de vigilancia al lado, es lo que convierte un experimento en algo desplegable con responsabilidad.

> **La pregunta que deberías poder responder al terminar:** ¿Cómo convertir resultados en una decisión responsable?

### Qué se mide y con qué se decide

El laboratorio reporta `accuracy`, `balanced_accuracy`, `precision`, `recall`, `f1`, `roc_auc`, `pr_auc`. De todas ellas, la que **decide** qué modelo se conserva es `f1`, y se mide siempre sobre `validation`: es la única forma de que `test` siga siendo una estimación honesta de lo que pasará con datos nuevos.

## 📓 Los tres cuadernos

El laboratorio incluye tres cuadernos Jupyter de **26 celdas** cada uno, de las cuales **13 son de código ejecutable**. Los tres recorren el mismo camino —descargar el dataset real, auditar la partición, entrenar, sellar el experimento y evaluar `test` una vez— y se diferencian en cuánto viene resuelto:

| Cuaderno | Qué trae | Cuándo usarlo |
|---|---|---|
| [📓 `notebook.ipynb`](notebook.ipynb) | El recorrido completo con **todo el código escrito y ejecutable**, celda a celda, intercalado con las explicaciones. | Para leer y ejecutar de principio a fin, entendiendo qué hace cada paso. |
| [✏️ `notebook_student.ipynb`](notebook_student.ipynb) | El mismo recorrido con **2 celdas vaciadas**, marcadas con `# YOUR CODE HERE`, que hay que completar. | Para practicar: se ejecuta igual, pero falla hasta que completas los huecos. |
| [✅ `notebook_solution.ipynb`](notebook_solution.ipynb) | Las celdas anteriores ya resueltas, marcadas con `# SOLUCIÓN DE REFERENCIA`. | Para contrastar tu respuesta después de intentarlo. |

> **Aviso honesto sobre el estado actual.** Hoy `notebook.ipynb` y `notebook_solution.ipynb` tienen **el mismo contenido**, y los ejercicios que los separan del cuaderno de estudiante son **2**. Es decir: el código del laboratorio está completo y es ejecutable en los tres, pero la versión de estudiante todavía no propone una práctica extensa. Está anotado en el [roadmap](../../ROADMAP.md) y se dice aquí para que nadie descubra el límite después de abrir el archivo.

### Cómo abrirlos

Los cuadernos necesitan el extra `notebooks`, que instala Jupyter junto con el paquete:

```bash
pip install -e ".[dev,notebooks]"
jupyter lab labs/24_capstone_real_project/notebook.ipynb
```

También se abren desde VS Code —con la extensión de Jupyter— haciendo doble clic en el archivo, o desde la interfaz clásica con `jupyter notebook`. El primer arranque descarga el dataset real desde su proveedor, así que la primera ejecución tarda más y **requiere conexión**.

Si prefieres ejecutar sin abrir un cuaderno, `train.py` hace exactamente lo mismo desde la terminal, y la sección de comandos de arriba explica cada opción.

## 🖥️ Los comandos, explicados

Todo el laboratorio se maneja con una sola herramienta de terminal, `neural-labs`, que se instala junto con el paquete (`pip install -e ".[dev,notebooks]"`). Cada subcomando hace **una** cosa del protocolo, y por eso se pueden ejecutar por separado: preparar datos, auditar la partición, entrenar, repetir con varias semillas.

La forma general es siempre la misma:

```bash
neural-labs <subcomando> --lab <identificador> [opciones]
```

| Opción | Valor por defecto | Valores | Qué hace y cuándo cambiarla |
|---|---|---|---|
| `--lab` | `24_capstone_real_project` | obligatorio | Qué laboratorio se ejecuta. Solo acepta los identificadores del catálogo. |
| `--quick` | desactivado | — | Usa una fracción real del dataset y pocas épocas. Sirve para comprobar la instalación, no para concluir nada sobre el modelo. |
| `--split-seed N` | `42` | entero | Semilla que decide **qué ejemplo cae en qué partición**. Se mantiene fija al comparar modelos. |
| `--training-seed N` | `42` | entero | Semilla de la inicialización de pesos y del barajado de lotes. Es la que se varía para medir cuánta diferencia es simple azar. |
| `--config` | `baseline` | `baseline` · `improved` | Cuál de las dos configuraciones del laboratorio se usa. |
| `--device` | `auto` | `auto` · `cpu` · `cuda` · `mps` | Dónde entrenar. `auto` elige GPU si está disponible y cae a CPU si no. |
| `--training-seeds A B C` | `41 42 43` | enteros | Solo en `benchmark`: la lista de semillas de entrenamiento que se repiten. |
| `--output-dir` | `runs` | ruta | Dónde se escribe el directorio de la ejecución. |

### El script del laboratorio

`labs/24_capstone_real_project/train.py` no es un programa distinto: fija el `--lab` y delega en la misma herramienta, de modo que estas dos líneas hacen exactamente lo mismo.

```bash
python labs/24_capstone_real_project/train.py --quick
neural-labs train --lab 24_capstone_real_project --quick
```

### Lo mismo desde Python

Si prefieres trabajar en un cuaderno o llamar al laboratorio desde tu propio código, la misma ejecución se lanza así. La función devuelve un objeto con el directorio de la ejecución, las métricas y el historial ya cargados:

```python
from neural_labs.experiments import run_lab

resultado = run_lab(
    "24_capstone_real_project",
    quick=True,          # False para la ejecución completa
    config_name="baseline",
    split_seed=42,       # fija la partición
    training_seed=43,    # varía la inicialización
)

print(resultado.run_dir)   # dónde quedaron los archivos
print(resultado.metrics)   # el diccionario de métricas finales
```

Y para preparar el dataset sin entrenar —útil para inspeccionarlo antes—:

```python
from neural_labs.datasets import prepare_dataset

datos = prepare_dataset("24_capstone_real_project", quick=True, seed=42)
print(datos.summary)       # tamaño de cada partición y metadatos de la fuente
```

## 🪜 Paso a paso

Cada paso dice qué ocurre por dentro, por qué se hace en ese orden y cómo comprobar que salió bien. El orden no es una convención de estilo: es el que ejecuta el código, y alterarlo invalida el resultado.

### Paso 1 — Traer el dataset real y partirlo

**Qué ocurre.** Descarga `iranian_churn` desde su proveedor y construye las tres particiones —`train`, `validation` y `test`— con la semilla de partición que le pases.

**Por qué.** La partición se fija **antes** de entrenar y su semilla (`--split-seed`) es independiente de la del entrenamiento (`--training-seed`). Si ambas fueran la misma, no podrías distinguir si un cambio en el resultado viene de haber repartido los datos de otro modo o de haber inicializado los pesos de otro modo.

```bash
neural-labs dataset --lab 24_capstone_real_project --quick --split-seed 42
```

**Cómo sabes que salió bien.** El comando termina sin error y deja el dataset en caché. Si no hay red, **falla**: no se rellena con datos inventados (`data/dataset.yaml` declara `fallback_to_generated_data: false`).

### Paso 2 — Comprobar que las particiones no se tocan

**Qué ocurre.** Recorre los identificadores de las tres particiones y verifica que ninguno aparece en dos de ellas.

**Por qué.** Una sola fila compartida entre `train` y `test` basta para que la métrica final quede inflada. Es el error más caro del oficio porque no produce ningún síntoma visible: el modelo simplemente «parece» mejor de lo que es.

```bash
neural-labs audit --lab 24_capstone_real_project --quick --split-seed 42
```

**Cómo sabes que salió bien.** La auditoría reporta cero solapamientos. Si reporta alguno, no sigas: el resto del laboratorio no significaría nada.

### Paso 3 — Mirar los datos antes de modelarlos

**Qué ocurre.** Genera el informe de calidad (valores faltantes, balance de clases, rangos) y el de deriva entre particiones.

**Por qué.** Un desbalance fuerte o una diferencia de distribución entre `train` y `test` cambia qué métrica tiene sentido leer. Descubrirlo después de entrenar obliga a repetirlo todo.

```bash
neural-labs quality --lab 24_capstone_real_project --quick --split-seed 42
```

**Cómo sabes que salió bien.** Obtienes `data_quality.json` y `drift_report.json`; ábrelos antes de decidir la configuración.

### Paso 4 — Estudiar la teoría del laboratorio

**Qué ocurre.** Leer [`theory.md`](theory.md): la idea central, el desarrollo matemático, los riesgos de interpretación y la bibliografía de la que sale todo eso.

**Por qué.** Sin esto, el entrenamiento es una caja que devuelve números. La teoría es lo que te permite decidir qué mirar y reconocer cuándo un resultado es sospechoso.

**Cómo sabes que salió bien.** Puedes responder, con tus palabras, qué calcula el modelo y por qué esa arquitectura encaja con la tarea `binary_classification`.

### Paso 5 — Entrenar y seleccionar con `validation`

**Qué ocurre.** El entrenamiento recorre las épocas midiendo en `validation` después de cada una, y conserva el checkpoint con el mejor valor de `f1`.

**Por qué.** El conjunto de validación existe para tomar decisiones —arquitectura, hiperparámetros, cuándo parar—. Si esas decisiones se tomaran mirando `test`, `test` dejaría de ser una estimación de lo que pasará con datos nuevos y pasaría a ser parte del entrenamiento.

```bash
python labs/24_capstone_real_project/train.py --quick
# o, con control explícito de las dos semillas:
neural-labs train --lab 24_capstone_real_project --config baseline --split-seed 42 --training-seed 43
```

**Cómo sabes que salió bien.** En `runs/24_capstone_real_project/<ejecución>/` aparecen `history.csv` y `best_model.pt`; la métrica de validación mejora respecto de la primera época.

### Paso 6 — Compararte con la línea base

**Qué ocurre.** El repositorio entrena por su cuenta **Regresión logística y Gradient Boosting** y guarda su resultado, primero sobre `validation` y —solo al final— sobre `test`.

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
neural-labs benchmark --lab 24_capstone_real_project --quick --split-seed 42 --training-seeds 41 42 43
```

**Cómo sabes que salió bien.** Obtienes media y dispersión entre semillas, no un único número.

### Paso 10 — Documentar y cerrar

**Qué ocurre.** Cada ejecución deja `model_card.md` y `report.md`; el plan de experimentos vive en [`experiments.md`](experiments.md) y la rúbrica en [`assessment.md`](assessment.md).

**Por qué.** Un resultado sin su contexto —qué datos, qué decisiones, qué límites— no es reutilizable. La model card es lo que permite que otra persona sepa cuándo *no* debería usar tu modelo.

**Cómo sabes que salió bien.** Completaste la tabla multi-semilla de `experiments.md` y respondiste las preguntas de `assessment.md`.

## 🔍 Cómo leer lo que produce la ejecución

Cada ejecución escribe su propio directorio con nombre único, de modo que dos corridas nunca se pisan. Esto es lo que encontrarás dentro:

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
| `confusion_matrix.png` | Qué clases se confunden entre sí. |
| `model_spec.json` · `inference_contract.json` | Qué entrada espera el modelo y qué devuelve: lo que necesita quien lo despliegue. |
| `model_card.md` · `report.md` | La ficha del modelo y el informe legible de la ejecución. |

## ⚠️ Dónde suele perderse la gente

- **`--quick` no es una versión pequeña del resultado, es una prueba de que todo corre.** En esta ruta recorta a 1024 ejemplos de entrenamiento · 256 de validación · 256 de test · 2 épocas. Sirve para comprobar la instalación y la descarga; cualquier conclusión sobre el modelo exige la ejecución completa.
- **Cambiar algo después de ver `test` invalida la comparación.** Si al mirar el resultado final se te ocurre una mejora, la ruta correcta es volver a `validation`, decidir allí, y sellar de nuevo.
- **Las dos semillas no son intercambiables.** `--split-seed` cambia *qué datos* caen en cada partición; `--training-seed` cambia *cómo se inicializa y baraja* el entrenamiento. Para comparar modelos se fija la primera y se varía la segunda.
- **Límite declarado de este dataset.** 3.150 clientes recolectados aleatoriamente de la base de una empresa iraní de telecomunicaciones durante 12 meses.

### Riesgos al interpretar los resultados

3.150 clientes recolectados aleatoriamente de la base de una empresa iraní de telecomunicaciones durante 12 meses.

El dataset refleja su proceso de recolección y no representa automáticamente otros períodos, países o poblaciones. Una asociación predictiva no demuestra causalidad.

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

El plan experimental con la tabla que hay que completar está en `experiments.md`, y las preguntas con su rúbrica, en `assessment.md`. Ambos documentos se abren desde la barra de navegación de arriba.

### Para ir más lejos

- Cambia una decisión experimental y justifícala con el resultado en `validation`, no con la intuición.
- Analiza los errores por clase o por segmento: casi siempre se concentran en un subconjunto reconocible.
- Compara costo, precisión y latencia; el mejor modelo no siempre es el que gana por décimas.
- Documenta sesgos, limitaciones y usos para los que **no** recomendarías este modelo.

## 📚 Fuentes

La teoría de arriba no es original de este repositorio: se apoya en la literatura de referencia del área y en los papers originales de cada arquitectura. Estas son las obras concretas, y lo que aporta cada una:

> Las referencias apuntan a las obras; no se reproduce su contenido, la redacción es original.

- Géron — *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* (3.ª ed., O'Reilly, 2022), cap. 2 — recorrido completo de un proyecto de ML de punta a punta, del marco del problema al despliegue.
- Huyen — *Designing Machine Learning Systems* (O'Reilly, 2022) — diseño de sistemas de ML en producción: métricas de negocio, monitorización y despliegue responsable.
- Kuhn y Johnson — *Applied Predictive Modeling* (Springer, 2013) — modelado predictivo aplicado: preprocesamiento, evaluación con clases desbalanceadas y selección de umbral.
- Fuente del dataset: https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset
- Consulte `docs/experiment-protocol.md`, `docs/reproducibility.md` y `docs/ethics-and-licenses.md`.

### Los archivos de este laboratorio

Todo lo que necesitas está en esta carpeta. Cada enlace abre el archivo directamente:

| Archivo | Qué es |
|---|---|
| [📄 `README.md`](README.md) | Esta guía. |
| [🧠 `theory.md`](theory.md) | La teoría completa con su bibliografía; es la fuente del apartado teórico de arriba. |
| [🔬 `experiments.md`](experiments.md) | El plan experimental y la tabla multi-semilla que hay que completar. |
| [📝 `assessment.md`](assessment.md) | Las preguntas de evaluación y la rúbrica con la que se corrigen. |
| [📓 `notebook.ipynb`](notebook.ipynb) | El recorrido completo con todo el código escrito y ejecutable. |
| [✏️ `notebook_student.ipynb`](notebook_student.ipynb) | El mismo recorrido con las celdas de ejercicio vacías. |
| [✅ `notebook_solution.ipynb`](notebook_solution.ipynb) | Los ejercicios resueltos, para contrastar. |
| [🖥️ `train.py`](train.py) | El mismo entrenamiento desde la terminal, sin abrir un cuaderno. |
| [🎛️ `configs/baseline.yaml`](configs/baseline.yaml) | Épocas, lote, tasa de aprendizaje y qué recorta `--quick`. |
| [🎚️ `configs/improved.yaml`](configs/improved.yaml) | La configuración ampliada que se compara contra la base. |
| [🗄️ `data/dataset.yaml`](data/dataset.yaml) | Fuente, licencia, política de partición y límites del dataset. |
| [🧾 `lesson.yaml`](lesson.yaml) | Nivel, prerrequisitos, resultados de aprendizaje y criterios. |
| [🖥️ `index.html`](index.html) | Esta misma clase como página autocontenida, para leerla sin conexión. |

Y fuera de la carpeta, tres referencias que esta guía usa: el catálogo `configs/labs.yaml` —de donde salen el objetivo, la línea base y las métricas—, el código `src/neural_labs/experiments.py` —que define el orden de los pasos y los archivos que escribe cada ejecución— y `docs/experiment-protocol.md`, con la regla general del protocolo.

Los datasets se descargan de su proveedor original y conservan su licencia; este repositorio no los redistribuye ni sustituye una descarga fallida por datos generados.

<!-- nav-bottom -->
## 🧭 Navegación del recorrido

| ⬅️ Laboratorio anterior | 🏠 Índice | Laboratorio siguiente ➡️ |
|---|:---:|---|
| [📦 Exportación e inferencia](../../labs/23_model_export_and_inference/README.md) | [Las 31 rutas](../../parts/README.md) | [🔧 Fine-tuning eficiente de transformer](../../advanced_labs/25_transformer_finetuning/README.md) |

**En este laboratorio:** **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

⚫ [Parte 6 — Confiar en el modelo y sacarlo del cuaderno](../../parts/06-confianza-y-despliegue.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/24_capstone_real_project/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
