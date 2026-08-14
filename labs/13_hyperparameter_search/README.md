# Búsqueda de hiperparámetros

<!-- nav-top -->
> 🧭 **Ruta 14 / 31** · 🟠 [Parte 4 — Entrenar mejor, más barato y sin centralizar datos](../../parts/04-entrenamiento-eficiente.md)
>
> [⬅️ 🔀 Fusión de sensores](../../labs/12_multimodal_fusion/README.md) · [🏠 Índice de rutas](../../parts/README.md) · [⚗️ Destilación de conocimiento ➡️](../../labs/14_knowledge_distillation/README.md)
>
> **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

## 🎯 Qué vas a hacer aquí

Optimizar profundidad, ancho, dropout y learning rate sin tocar test.

Es la **ruta 14 de 31** del recorrido y pertenece a 🟠 la parte 4, *Entrenar mejor, más barato y sin centralizar datos*. Llegas desde **Fusión de sensores** y lo que hagas aquí lo da por supuesto **Destilación de conocimiento**.

Trabajarás con el dataset **`adult_census`** (UCI, licencia: CC BY 4.0), y tendrás que superar la línea base **Regresión logística**, decidiendo con la métrica `f1` medida sobre `validation`. Nivel avanzado, unas **8 horas** de dedicación.

**Lo que conviene traer resuelto de las rutas anteriores:** PyTorch intermedio, optimización, lectura de artículos técnicos.

**Al terminar deberías ser capaz de:**

- Optimizar profundidad, ancho, dropout y learning rate sin tocar test.
- Preparar y auditar el dataset real adult_census sin fuga de datos.
- Entrenar y evaluar búsqueda de hiperparámetros sin tocar test.
- Comparar contra la línea base: Regresión logística.
- Interpretar intervalos de confianza, errores y limitaciones.

## 🧠 La teoría de este laboratorio

Esta sección es la explicación completa del tema. No hace falta abrir otro archivo para entender lo que viene después: aquí está la idea, la matemática que la sostiene y sus límites. (El mismo texto vive en `theory.md`, que es la fuente desde la que se genera esta guía, junto con la bibliografía del final.)

### De qué trata

Este laboratorio estudia **búsqueda de hiperparámetros sin tocar test** usando `adult_census`, un dataset público real procedente de UCI. Los hiperparámetros no se aprenden por descenso de gradiente: son decisiones de diseño (número de capas, neuronas por capa, tasa de dropout, learning rate) que gobiernan *cómo* se aprenden los parámetros. Ajustarlos bien es lo que separa un modelo que memoriza de uno que generaliza.

La idea central del protocolo es tratar la búsqueda como un experimento con tres particiones estrictamente separadas. Se prueban muchas configuraciones, cada una se entrena con `train` y se puntúa con `validation`; el conjunto `test` permanece sellado hasta el final. Esto evita el **sesgo de selección optimista**: si eligiéramos la mejor configuración mirando `test`, esa métrica dejaría de ser una estimación honesta del desempeño futuro, porque habríamos ajustado nuestras decisiones al ruido específico de ese conjunto.

Sobre la estrategia de búsqueda, el laboratorio contrasta la intuición ingenua (probar en malla, grid search) con hallazgos empíricos más eficientes. La **búsqueda aleatoria** suele encontrar buenas configuraciones con menos evaluaciones porque, cuando pocos hiperparámetros dominan el desempeño, muestrear al azar explora más valores distintos de esos hiperparámetros importantes que una malla rígida. Frameworks modernos añaden búsqueda guiada (por ejemplo, muestreo bayesiano) y poda temprana de pruebas poco prometedoras.

### La matemática, paso a paso

Sea λ un vector de hiperparámetros en un espacio de búsqueda Λ (profundidad, ancho, dropout p, learning rate η, …). Para cada λ se entrena un modelo obteniendo parámetros óptimos sobre entrenamiento:

  θ*(λ) = argmin_θ ℒ_train(θ; λ)

y se evalúa su calidad en validación. La búsqueda de hiperparámetros es el problema anidado (bilevel):

  λ* = argmin_{λ ∈ Λ} ℒ_val( θ*(λ) )

El punto crítico es que **λ se elige mirando `validation`, nunca `test`**. El error de test solo se mide una vez, con λ* ya congelado, para estimar la generalización sin sesgo.

En la **búsqueda en malla** se discretiza cada dimensión y se prueban todas las combinaciones: el costo crece como el producto de los tamaños por dimensión (maldición de la dimensionalidad). En la **búsqueda aleatoria** se muestrean T configuraciones λ⁽¹⁾, …, λ⁽ᵀ⁾ de una distribución sobre Λ y se conserva el mejor. La intuición de por qué gana: si solo d_eff de las d dimensiones influyen de verdad, la malla desperdicia evaluaciones repitiendo los mismos valores de las dimensiones importantes, mientras que el muestreo aleatorio prueba T valores distintos de cada una.

Como cada θ*(λ) depende de la inicialización y del orden de los minibatches, la métrica de validación es una variable aleatoria. Por eso se reporta ℒ_val como media ± desviación sobre varias semillas: comparar dos configuraciones con un único número puede confundir una mejora real con ruido de entrenamiento. La formulación conecta cuatro elementos: representación de entrada, función del modelo, función de pérdida y regla de actualización. El notebook muestra las dimensiones de los tensores y conserva la misma implementación que el script de terminal.

### Por qué la búsqueda aleatoria gana a la malla

El resultado de Bergstra y Bengio suele citarse como una preferencia práctica, y en realidad es un argumento geométrico que se puede seguir con lápiz.

Supóngase un presupuesto de 25 evaluaciones sobre dos hiperparámetros. La **malla** los reparte en 5×5, así que explora exactamente **5 valores distintos** de cada uno: los otros 20 puntos repiten valores ya probados en la otra dimensión. La búsqueda **aleatoria** con las mismas 25 evaluaciones prueba **25 valores distintos** de cada hiperparámetro, porque cada muestra aporta una coordenada nueva en todas las dimensiones a la vez.

Esa diferencia sería irrelevante si todos los hiperparámetros importaran por igual, pero no es el caso: en la práctica el desempeño depende fuertemente de unos pocos —la tasa de aprendizaje casi siempre— y es casi plano en los demás. Con una malla, la resolución sobre el hiperparámetro que sí importa queda limitada a 5 valores, y las otras 20 evaluaciones se gastan variando cosas que no cambian nada. Con muestreo aleatorio, las 25 evaluaciones aportan 25 puntos distintos sobre la dimensión relevante, sea cual sea.

El problema empeora con la dimensión. Una malla de k valores en d hiperparámetros exige k^d evaluaciones: pasar de 2 a 5 hiperparámetros con k = 5 lleva de 25 a 3 125 entrenamientos. Es la **maldición de la dimensionalidad** en su forma más concreta, y la razón de que la malla solo sea razonable con dos o tres parámetros y muy pocos valores.

Un corolario que también importa: la probabilidad de que **ninguna** de n muestras aleatorias caiga en el mejor 5 % del espacio es 0,95ⁿ. Con n = 60 eso es 0,046, es decir, hay un 95 % de probabilidad de encontrar una configuración dentro del mejor 5 % con solo **60 pruebas**, y ese número no depende de cuántas dimensiones haya. Es la garantía que hace defendible la búsqueda aleatoria como opción por defecto.

### El espacio de búsqueda importa tanto como el algoritmo

Un buscador solo puede encontrar lo que hay dentro del rango que se le da, así que definir el espacio es parte del experimento y debe declararse.

La regla más útil es que los hiperparámetros **multiplicativos se muestrean en escala logarítmica**. La tasa de aprendizaje es el caso claro: entre 10⁻⁵ y 10⁻¹ hay cuatro órdenes de magnitud, y muestrear uniformemente en ese intervalo pondría el 90 % de las muestras por encima de 10⁻², dejando prácticamente inexplorada la zona pequeña. Lo correcto es muestrear log₁₀(η) ~ U(−5, −1), que reparte por igual entre órdenes de magnitud. Lo mismo vale para el weight decay y para el tamaño de capa. Los parámetros **aditivos o acotados** —dropout entre 0 y 0,5, número de capas entre 1 y 4— sí se muestrean de forma uniforme.

Hay un fenómeno adicional que conviene anticipar: los hiperparámetros **interactúan**. Tasa de aprendizaje y tamaño de lote están acopladas —al aumentar el lote, el gradiente es menos ruidoso y admite pasos mayores—, y la tasa óptima suele escalar con el lote. Buscar cada uno por separado, fijando el otro, puede dejar fuera el óptimo conjunto. Es la razón de que la búsqueda se haga sobre el espacio completo y no parámetro a parámetro.

### El riesgo real de este laboratorio: sobreajustar la validación

Este es el punto que distingue una búsqueda rigurosa de una que se engaña a sí misma, y es la razón de que el protocolo del repositorio sea especialmente estricto aquí.

Cada configuración probada se evalúa en `validation`, y al final se elige la mejor. Pero elegir el máximo de n estimaciones ruidosas produce un valor **optimista**: es el mismo sesgo de selección de la ruta 10, 𝔼[max] ≥ max 𝔼. Cuanto mayor es n, mayor es el sesgo. Con cien configuraciones probadas, la métrica de validación de la ganadora incluye una porción apreciable de suerte, y **no es una estimación insesgada** de lo que rendirá con datos nuevos.

De ahí se sigue lo que hay que hacer y lo que no. La cifra que se reporta como resultado del laboratorio es la de `test`, medida **una sola vez** con la configuración ya elegida y sellada; el valor de validación de la ganadora se reporta como lo que es, un criterio de selección y no una estimación de desempeño. Y si se quisiera además estimar honestamente el error del procedimiento completo de búsqueda, haría falta una **validación cruzada anidada**: un bucle externo para estimar y uno interno para buscar, con un costo multiplicativo que este laboratorio no asume, pero que conviene saber que existe.

Un último detalle sobre el presupuesto: comparar dos estrategias de búsqueda solo tiene sentido **a igual número de evaluaciones**. Decir que la búsqueda bayesiana batió a la aleatoria es vacío si la primera probó 200 configuraciones y la segunda 20. Y el costo total —tiempo de cómputo acumulado, no solo el del modelo ganador— forma parte del resultado, porque una mejora de dos décimas que costó cien entrenamientos rara vez se justifica.

> **La pregunta que deberías poder responder al terminar:** ¿El mejor trial generaliza a semillas nuevas?

### Qué se mide y con qué se decide

El laboratorio reporta `accuracy`, `balanced_accuracy`, `f1`, `roc_auc`, `pr_auc`. De todas ellas, la que **decide** qué modelo se conserva es `f1`, y se mide siempre sobre `validation`: es la única forma de que `test` siga siendo una estimación honesta de lo que pasará con datos nuevos.

## 🖥️ Los comandos, explicados

Todo el laboratorio se maneja con una sola herramienta de terminal, `neural-labs`, que se instala junto con el paquete (`pip install -e ".[dev,notebooks]"`). Cada subcomando hace **una** cosa del protocolo, y por eso se pueden ejecutar por separado: preparar datos, auditar la partición, entrenar, repetir con varias semillas.

La forma general es siempre la misma:

```bash
neural-labs <subcomando> --lab <identificador> [opciones]
```

| Opción | Valor por defecto | Valores | Qué hace y cuándo cambiarla |
|---|---|---|---|
| `--lab` | `13_hyperparameter_search` | obligatorio | Qué laboratorio se ejecuta. Solo acepta los identificadores del catálogo. |
| `--quick` | desactivado | — | Usa una fracción real del dataset y pocas épocas. Sirve para comprobar la instalación, no para concluir nada sobre el modelo. |
| `--split-seed N` | `42` | entero | Semilla que decide **qué ejemplo cae en qué partición**. Se mantiene fija al comparar modelos. |
| `--training-seed N` | `42` | entero | Semilla de la inicialización de pesos y del barajado de lotes. Es la que se varía para medir cuánta diferencia es simple azar. |
| `--config` | `baseline` | `baseline` · `improved` | Cuál de las dos configuraciones del laboratorio se usa. |
| `--device` | `auto` | `auto` · `cpu` · `cuda` · `mps` | Dónde entrenar. `auto` elige GPU si está disponible y cae a CPU si no. |
| `--training-seeds A B C` | `41 42 43` | enteros | Solo en `benchmark`: la lista de semillas de entrenamiento que se repiten. |
| `--output-dir` | `runs` | ruta | Dónde se escribe el directorio de la ejecución. |

### El script del laboratorio

`labs/13_hyperparameter_search/train.py` no es un programa distinto: fija el `--lab` y delega en la misma herramienta, de modo que estas dos líneas hacen exactamente lo mismo.

```bash
python labs/13_hyperparameter_search/train.py --quick
neural-labs train --lab 13_hyperparameter_search --quick
```

### Lo mismo desde Python

Si prefieres trabajar en un cuaderno o llamar al laboratorio desde tu propio código, la misma ejecución se lanza así. La función devuelve un objeto con el directorio de la ejecución, las métricas y el historial ya cargados:

```python
from neural_labs.experiments import run_lab

resultado = run_lab(
    "13_hyperparameter_search",
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

datos = prepare_dataset("13_hyperparameter_search", quick=True, seed=42)
print(datos.summary)       # tamaño de cada partición y metadatos de la fuente
```

## 🪜 Paso a paso

Cada paso dice qué ocurre por dentro, por qué se hace en ese orden y cómo comprobar que salió bien. El orden no es una convención de estilo: es el que ejecuta el código, y alterarlo invalida el resultado.

### Paso 1 — Traer el dataset real y partirlo

**Qué ocurre.** Descarga `adult_census` desde su proveedor y construye las tres particiones —`train`, `validation` y `test`— con la semilla de partición que le pases.

**Por qué.** La partición se fija **antes** de entrenar y su semilla (`--split-seed`) es independiente de la del entrenamiento (`--training-seed`). Si ambas fueran la misma, no podrías distinguir si un cambio en el resultado viene de haber repartido los datos de otro modo o de haber inicializado los pesos de otro modo.

```bash
neural-labs dataset --lab 13_hyperparameter_search --quick --split-seed 42
```

**Cómo sabes que salió bien.** El comando termina sin error y deja el dataset en caché. Si no hay red, **falla**: no se rellena con datos inventados (`data/dataset.yaml` declara `fallback_to_generated_data: false`).

### Paso 2 — Comprobar que las particiones no se tocan

**Qué ocurre.** Recorre los identificadores de las tres particiones y verifica que ninguno aparece en dos de ellas.

**Por qué.** Una sola fila compartida entre `train` y `test` basta para que la métrica final quede inflada. Es el error más caro del oficio porque no produce ningún síntoma visible: el modelo simplemente «parece» mejor de lo que es.

```bash
neural-labs audit --lab 13_hyperparameter_search --quick --split-seed 42
```

**Cómo sabes que salió bien.** La auditoría reporta cero solapamientos. Si reporta alguno, no sigas: el resto del laboratorio no significaría nada.

### Paso 3 — Mirar los datos antes de modelarlos

**Qué ocurre.** Genera el informe de calidad (valores faltantes, balance de clases, rangos) y el de deriva entre particiones.

**Por qué.** Un desbalance fuerte o una diferencia de distribución entre `train` y `test` cambia qué métrica tiene sentido leer. Descubrirlo después de entrenar obliga a repetirlo todo.

```bash
neural-labs quality --lab 13_hyperparameter_search --quick --split-seed 42
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
python labs/13_hyperparameter_search/train.py --quick
# o, con control explícito de las dos semillas:
neural-labs train --lab 13_hyperparameter_search --config baseline --split-seed 42 --training-seed 43
```

**Cómo sabes que salió bien.** En `runs/13_hyperparameter_search/<ejecución>/` aparecen `history.csv` y `best_model.pt`; la métrica de validación mejora respecto de la primera época.

### Paso 6 — Compararte con la línea base

**Qué ocurre.** El repositorio entrena por su cuenta **Regresión logística** y guarda su resultado, primero sobre `validation` y —solo al final— sobre `test`.

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
neural-labs benchmark --lab 13_hyperparameter_search --quick --split-seed 42 --training-seeds 41 42 43
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
| `hyperparameter_trials.json` | **Propio de esta ruta.** Cada combinación probada, su puntaje y la ganadora. |

## ⚠️ Dónde suele perderse la gente

- **`--quick` no es una versión pequeña del resultado, es una prueba de que todo corre.** En esta ruta recorta a 1024 ejemplos de entrenamiento · 256 de validación · 256 de test · 2 épocas. Sirve para comprobar la instalación y la descarga; cualquier conclusión sobre el modelo exige la ejecución completa.
- **Cambiar algo después de ver `test` invalida la comparación.** Si al mirar el resultado final se te ocurre una mejora, la ruta correcta es volver a `validation`, decidir allí, y sellar de nuevo.
- **Las dos semillas no son intercambiables.** `--split-seed` cambia *qué datos* caen en cada partición; `--training-seed` cambia *cómo se inicializa y baraja* el entrenamiento. Para comparar modelos se fija la primera y se varía la segunda.
- **Límite declarado de este dataset.** 48.842 registros reales del censo de 1994.

### Riesgos al interpretar los resultados

48.842 registros reales del censo de 1994.

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

- Géron — *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* (3.ª ed., O'Reilly 2022), cap. 10 — introducción práctica a redes densas y al ajuste de sus hiperparámetros.
- Goodfellow, Bengio & Courville — *Deep Learning* (MIT Press, 2016), cap. 11 — metodología práctica para seleccionar hiperparámetros y depurar experimentos.
- Bergstra & Bengio (2012), *Random Search for Hyper-Parameter Optimization*, JMLR — evidencia de por qué la búsqueda aleatoria supera a la malla cuando pocos hiperparámetros dominan.
- Akiba et al. (2019), *Optuna: A Next-generation Hyperparameter Optimization Framework*, KDD — framework de búsqueda guiada con muestreo eficiente y poda temprana.
- Fuente del dataset: https://archive.ics.uci.edu/dataset/2/adult
- Consulte `docs/experiment-protocol.md`, `docs/reproducibility.md` y `docs/ethics-and-licenses.md`.

### Cómo comprobar lo que dice esta guía

Ninguna cifra ni afirmación de esta página está escrita de memoria. Cada una se puede verificar en un archivo del repositorio:

| Lo que dice la guía | Dónde comprobarlo |
|---|---|
| Objetivo, línea base, métricas y arquitectura | `configs/labs.yaml` |
| Fuente, licencia, procedencia y límites del dataset | `data/dataset.yaml` |
| Épocas, tamaño de lote, tasa de aprendizaje y recorte de `--quick` | `configs/baseline.yaml` y `configs/improved.yaml` |
| Nivel, prerrequisitos, resultados de aprendizaje y criterios | `lesson.yaml` |
| Opciones de los comandos y sus valores por defecto | `src/neural_labs/cli.py` |
| El orden de los pasos y los archivos que escribe cada ejecución | `src/neural_labs/experiments.py` |
| La teoría y su bibliografía | `theory.md` |
| La regla general del protocolo | `docs/experiment-protocol.md` |

Los datasets se descargan de su proveedor original y conservan su licencia; este repositorio no los redistribuye ni sustituye una descarga fallida por datos generados.

<!-- nav-bottom -->
## 🧭 Navegación del recorrido

| ⬅️ Laboratorio anterior | 🏠 Índice | Laboratorio siguiente ➡️ |
|---|:---:|---|
| [🔀 Fusión de sensores](../../labs/12_multimodal_fusion/README.md) | [Las 31 rutas](../../parts/README.md) | [⚗️ Destilación de conocimiento](../../labs/14_knowledge_distillation/README.md) |

**En este laboratorio:** **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🟠 [Parte 4 — Entrenar mejor, más barato y sin centralizar datos](../../parts/04-entrenamiento-eficiente.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/13_hyperparameter_search/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
