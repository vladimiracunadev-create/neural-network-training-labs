# LSTM para series temporales

<!-- nav-top -->
> 🧭 **Ruta 6 / 31** · 🔵 [Parte 2 — Arquitecturas según la forma del dato](../../parts/02-arquitecturas.md)
>
> [⬅️ 🔁 RNN para texto](../../labs/04_rnn_sequences/README.md) · [🏠 Índice de rutas](../../parts/README.md) · [🧬 Autoencoder para fraude ➡️](../../labs/06_autoencoder_anomaly/README.md)
>
> **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

## 🎯 Qué vas a hacer aquí

Pronosticar demanda horaria respetando el orden temporal.

Es la **ruta 6 de 31** del recorrido y pertenece a 🔵 la parte 2, *Arquitecturas según la forma del dato*. Llegas desde **RNN para texto** y lo que hagas aquí lo da por supuesto **Autoencoder para fraude**.

Trabajarás con el dataset **`seoul_bike`** (UCI, licencia: CC BY 4.0), y tendrás que superar la línea base **Persistencia, media móvil y Ridge**, decidiendo con la métrica `rmse` medida sobre `validation`. Nivel intermedio, unas **6 horas** de dedicación.

**Lo que conviene traer resuelto de las rutas anteriores:** PyTorch básico, particiones train/validation/test, métricas de evaluación.

**Al terminar deberías ser capaz de:**

- Pronosticar demanda horaria respetando el orden temporal.
- Preparar y auditar el dataset real seoul_bike sin fuga de datos.
- Entrenar y evaluar memoria recurrente para pronóstico temporal.
- Comparar contra la línea base: Persistencia, media móvil y Ridge.
- Interpretar intervalos de confianza, errores y limitaciones.

## 🧠 La teoría de este laboratorio

Esta sección es la explicación completa del tema. No hace falta abrir otro archivo para entender lo que viene después: aquí está la idea, la matemática que la sostiene y sus límites. (El mismo texto vive en `theory.md`, que es la fuente desde la que se genera esta guía, junto con la bibliografía del final.)

### De qué trata

Este laboratorio estudia **memoria recurrente para pronóstico temporal** usando `seoul_bike`, un dataset público real procedente de UCI.

Este laboratorio ataca directamente la limitación descubierta con las RNN simples: su incapacidad para retener información a lo largo de muchos pasos por el desvanecimiento del gradiente. La **LSTM** (Long Short-Term Memory) introduce un canal de memoria protegido —el estado de celda cₜ— y un sistema de **puertas** que deciden, de forma aprendida, qué información conservar, qué olvidar y qué exponer en cada instante. El resultado es una memoria capaz de sostener patrones a largo plazo (ciclos diarios y semanales de demanda) sin que el gradiente se disipe.

El problema es un pronóstico de series temporales genuino: predecir la demanda horaria de bicicletas compartidas en Seúl a partir de su historia reciente y variables climáticas, sobre 8.760 observaciones reales. A diferencia de la clasificación, aquí el **orden temporal es sagrado**: la partición no puede mezclar futuro con pasado, y las líneas base (persistencia, media móvil, Ridge) son duras de batir. La pregunta crítica —si el modelo supera a la persistencia en períodos de cambio— pone el foco donde un pronosticador realmente demuestra su valor.

### La matemática, paso a paso

La LSTM mantiene dos estados que viajan en el tiempo: el estado oculto hₜ (la salida en cada paso) y el **estado de celda** cₜ (la memoria a largo plazo). En cada instante, tres puertas —vectores con valores en (0, 1) producidos por sigmoides σ— regulan el flujo de información. Con la concatenación de la entrada xₜ y el estado previo hₜ₋₁:

Puerta de olvido:  fₜ = σ(W_f·[hₜ₋₁, xₜ] + b_f)

Puerta de entrada:  iₜ = σ(W_i·[hₜ₋₁, xₜ] + b_i)

Candidato de memoria:  c̃ₜ = tanh(W_c·[hₜ₋₁, xₜ] + b_c)

Puerta de salida:  oₜ = σ(W_o·[hₜ₋₁, xₜ] + b_o)

La actualización del estado de celda es el corazón del mecanismo y combina las puertas mediante el **producto elemento a elemento** ⊙:

cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ c̃ₜ

hₜ = oₜ ⊙ tanh(cₜ)

La lectura es intuitiva: la puerta de olvido fₜ decide qué fracción de la memoria vieja cₜ₋₁ se conserva (fₜ ≈ 1 recuerda, fₜ ≈ 0 borra); la puerta de entrada iₜ decide cuánto del nuevo candidato c̃ₜ se escribe; y la de salida oₜ decide qué parte de la memoria se expone como estado oculto. Cuando la red aprende fₜ ≈ 1, el estado de celda actúa como una **cinta transportadora** por la que la información —y el gradiente— fluye a través de muchos pasos casi sin atenuarse. Esa suma cₜ = fₜ ⊙ cₜ₋₁ + … es precisamente lo que evita el producto de jacobianos que desvanecía el gradiente en la RNN simple: la ruta aditiva mantiene ∂cₜ/∂cₜ₋₁ ≈ fₜ en lugar de un factor que se contrae exponencialmente.

Una alternativa más ligera es la **GRU** (Cho et al. 2014), que fusiona las puertas de olvido y entrada en una sola puerta de actualización y prescinde del estado de celda separado, con menos parámetros y rendimiento a menudo comparable. Para el pronóstico, la salida h_T (o la de cada paso) pasa por una capa densa que produce el valor real predicho, y el entrenamiento minimiza un error de regresión como el **MSE**, L = (1/N) Σᵢ (ŷᵢ − yᵢ)². La evaluación reporta MAE, RMSE, MAPE y R², siempre comparando contra las líneas base clásicas de series temporales.

La formulación debe conectarse con cuatro elementos: representación de entrada, función del modelo, función de pérdida y regla de actualización. El notebook muestra las dimensiones de los tensores y conserva la misma implementación que el script de terminal.

### Por qué las puertas resuelven el desvanecimiento

La ruta 04 dejó el diagnóstico: el gradiente de una RNN simple se multiplica por W_hᵀ·diag(σ′) en cada paso, y ese producto se apaga exponencialmente. La LSTM no lo mitiga, **cambia la operación**, y ahí está toda su ventaja.

Derivando la actualización del estado de celda cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ c̃ₜ respecto del estado anterior:

∂cₜ/∂cₜ₋₁ = fₜ    (elemento a elemento),

de modo que el gradiente que atraviesa k pasos por la vía de la celda se multiplica por Π fₜ, un **producto de escalares entre 0 y 1**, y no por un producto de matrices con activaciones saturantes. La diferencia es cualitativa: cuando la puerta de olvido se mantiene cerca de 1 —el modelo ha decidido conservar esa memoria—, el factor es cerca de 1 y el gradiente **atraviesa cientos de pasos casi intacto**. Ese camino se conoce como *carrusel de error constante*, y es lo que permite aprender dependencias largas.

Obsérvese la estructura: la actualización es **aditiva**, cₜ = (algo)·cₜ₋₁ + (algo), mientras que en la RNN simple era completamente multiplicativa, hₜ = tanh(W·hₜ₋₁ + …). Es la misma idea que reaparece en las conexiones residuales de la ruta 03 y en los atajos de la 07: dejar un camino donde la señal se suma en vez de transformarse es lo que mantiene vivo el gradiente. Y aún así la LSTM no es inmune —si las puertas de olvido se cierran, la memoria y su gradiente se pierden—: la diferencia es que ahora eso es una **decisión aprendida** y no una fatalidad de la arquitectura.

De ahí una recomendación práctica bien establecida: inicializar el sesgo de la puerta de olvido en un valor positivo (típicamente 1). Con b_f = 1, la sigmoide arranca en σ(1) ≈ 0,73, así que la red empieza **conservando** memoria por defecto y aprende luego a olvidar. Con b_f = 0 arranca en 0,5 y el gradiente ya se reduce a la mitad por paso desde la primera época, justo cuando aún no ha aprendido nada que valga la pena conservar.

El costo de las puertas es lineal en parámetros. Con entrada de dimensión d y estado oculto h, cada una de las cuatro transformaciones consume (d + h)·h pesos más h sesgos, de modo que

|θ|_LSTM = 4 · ( (d + h)·h + h ),

cuatro veces una RNN simple del mismo tamaño. La **GRU** fusiona la celda con el estado oculto y usa tres puertas en vez de cuatro, así que cuesta 3·((d + h)·h + h): en torno a un 25 % menos, con rendimiento comparable en muchas tareas. Comparar ambas con el mismo presupuesto de parámetros —y no con la misma h— es la forma honesta de decidir entre ellas.

### Qué hace que una serie temporal no sea un dataset normal

La particularidad de este laboratorio no está en la arquitectura sino en el protocolo, y es donde se cometen los errores más caros.

El dato original es una única serie continua, no un conjunto de ejemplos independientes. Para entrenar se construyen ejemplos con una **ventana deslizante**: cada entrada es el tramo (x_(t−L+1), …, x_t) de longitud L y el objetivo es el valor en t + H, donde H es el **horizonte** de pronóstico. Elegir L y H no es cosmético: L acota cuánto pasado puede ver el modelo —si la serie tiene estacionalidad diaria de 24 horas, una ventana de 12 no puede capturarla— y H define un problema distinto, porque pronosticar la hora siguiente y pronosticar dentro de una semana no son la misma tarea ni admiten la misma comparación.

La partición **no puede ser aleatoria**. Repartir ventanas al azar entre `train`, `validation` y `test` coloca en el entrenamiento momentos posteriores a los que hay que predecir en la evaluación: el modelo aprende del futuro. Es una fuga de datos que no produce ningún síntoma —de hecho produce métricas excelentes— y por eso es tan peligrosa. La partición correcta es **cronológica**: un corte temporal, todo lo anterior a entrenamiento y lo posterior a evaluación, respetando el orden.

Hay un detalle más fino que se escapa incluso partiendo por fecha. Como las ventanas se solapan, una ventana de `train` que termine justo antes del corte puede tener su objetivo **después** del corte, dentro del periodo de validación. La solución estándar es dejar un hueco (*embargo*) de al menos H pasos entre particiones. Sin él, el solape filtra exactamente la información que se quería aislar.

El escalado arrastra el mismo principio: la media y la desviación se calculan **solo con el tramo de entrenamiento**. Estandarizar con las estadísticas de la serie completa introduce en el preprocesamiento información sobre el nivel y la variabilidad del futuro, que es justo lo que el modelo debería tener que inferir.

Y la línea base debe ser honesta. En series temporales, el modelo **ingenuo** —predecir que el valor siguiente será igual al último observado, ŷ_(t+H) = y_t— es sorprendentemente difícil de batir, y su versión estacional —ŷ_(t+H) = y_(t+H−s) con s el periodo— aún más. Una red que no supere claramente a ese piso no ha aprendido dinámica: ha aprendido a copiar. Por eso el error se reporta con métricas escaladas frente a esa referencia, del tipo MASE = MAE_modelo / MAE_ingenuo, cuyo valor 1 marca exactamente el punto en que el modelo deja de aportar.

> **La pregunta que deberías poder responder al terminar:** ¿El modelo supera persistencia en períodos de cambio?

### Qué se mide y con qué se decide

El laboratorio reporta `mae`, `rmse`, `mape`, `r2`. De todas ellas, la que **decide** qué modelo se conserva es `rmse`, y se mide siempre sobre `validation`: es la única forma de que `test` siga siendo una estimación honesta de lo que pasará con datos nuevos.

## 📓 Los tres cuadernos

El laboratorio se puede recorrer en Jupyter, y trae tres cuadernos con papeles distintos. Los tres siguen el mismo camino —descargar el dataset real, auditar la partición, entrenar, sellar el experimento y evaluar `test` una vez—; lo que cambia es qué te toca escribir a ti:

| Cuaderno | Qué trae | Cuándo usarlo |
|---|---|---|
| [📓 `notebook.ipynb`](notebook.ipynb) | El **recorrido de referencia**: 22 celdas (9 de código) con **todo el código escrito y ejecutable**, intercalado con las explicaciones. No trae ejercicios. | Para leer y ejecutar de principio a fin. |
| [✏️ `notebook_student.ipynb`](notebook_student.ipynb) | El mismo recorrido más **5 ejercicios evaluables** (37 celdas en total). Las celdas de ejercicio están marcadas con `# YOUR CODE HERE` y debajo de cada una hay una comprobación. | Para practicar. |
| [✅ `notebook_solution.ipynb`](notebook_solution.ipynb) | Los mismos ejercicios **resueltos**, marcados con `# SOLUCIÓN DE REFERENCIA`. Cada solución se ejecuta en la integración continua, así que se sabe que pasa. | Para contrastar después de intentarlo. |

### Qué se practica en los ejercicios

Cinco de ellos no son de arquitectura sino del **contrato experimental**, que es lo que distingue a estos laboratorios de un tutorial: auditar la partición, decidir con `validation`, compararse con la línea base, sellar antes de abrir `test` y dejar el plan por escrito. Se resuelven con Python estándar —**sin descargar el dataset ni entrenar**—, así que se corrigen en segundos y sin GPU, y cada uno está parametrizado con los valores de este laboratorio: su métrica de selección, su línea base y su experimento propio.

### Cómo abrirlos

Los cuadernos necesitan el extra `notebooks`, que instala Jupyter junto con el paquete:

```bash
pip install -e ".[dev,notebooks]"
jupyter lab labs/05_lstm_time_series/notebook.ipynb
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
| `--lab` | `05_lstm_time_series` | obligatorio | Qué laboratorio se ejecuta. Solo acepta los identificadores del catálogo. |
| `--quick` | desactivado | — | Usa una fracción real del dataset y pocas épocas. Sirve para comprobar la instalación, no para concluir nada sobre el modelo. |
| `--split-seed N` | `42` | entero | Semilla que decide **qué ejemplo cae en qué partición**. Se mantiene fija al comparar modelos. |
| `--training-seed N` | `42` | entero | Semilla de la inicialización de pesos y del barajado de lotes. Es la que se varía para medir cuánta diferencia es simple azar. |
| `--config` | `baseline` | `baseline` · `improved` | Cuál de las dos configuraciones del laboratorio se usa. |
| `--device` | `auto` | `auto` · `cpu` · `cuda` · `mps` | Dónde entrenar. `auto` elige GPU si está disponible y cae a CPU si no. |
| `--training-seeds A B C` | `41 42 43` | enteros | Solo en `benchmark`: la lista de semillas de entrenamiento que se repiten. |
| `--output-dir` | `runs` | ruta | Dónde se escribe el directorio de la ejecución. |

### El script del laboratorio

`labs/05_lstm_time_series/train.py` no es un programa distinto: fija el `--lab` y delega en la misma herramienta, de modo que estas dos líneas hacen exactamente lo mismo.

```bash
python labs/05_lstm_time_series/train.py --quick
neural-labs train --lab 05_lstm_time_series --quick
```

### Lo mismo desde Python

Si prefieres trabajar en un cuaderno o llamar al laboratorio desde tu propio código, la misma ejecución se lanza así. La función devuelve un objeto con el directorio de la ejecución, las métricas y el historial ya cargados:

```python
from neural_labs.experiments import run_lab

resultado = run_lab(
    "05_lstm_time_series",
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

datos = prepare_dataset("05_lstm_time_series", quick=True, seed=42)
print(datos.summary)       # tamaño de cada partición y metadatos de la fuente
```

## 🪜 Paso a paso

Cada paso dice qué ocurre por dentro, por qué se hace en ese orden y cómo comprobar que salió bien. El orden no es una convención de estilo: es el que ejecuta el código, y alterarlo invalida el resultado.

### Paso 1 — Traer el dataset real y partirlo

**Qué ocurre.** Descarga `seoul_bike` desde su proveedor y construye las tres particiones —`train`, `validation` y `test`— con la semilla de partición que le pases.

**Por qué.** La partición se fija **antes** de entrenar y su semilla (`--split-seed`) es independiente de la del entrenamiento (`--training-seed`). Si ambas fueran la misma, no podrías distinguir si un cambio en el resultado viene de haber repartido los datos de otro modo o de haber inicializado los pesos de otro modo.

```bash
neural-labs dataset --lab 05_lstm_time_series --quick --split-seed 42
```

**Cómo sabes que salió bien.** El comando termina sin error y deja el dataset en caché. Si no hay red, **falla**: no se rellena con datos inventados (`data/dataset.yaml` declara `fallback_to_generated_data: false`).

### Paso 2 — Comprobar que las particiones no se tocan

**Qué ocurre.** Recorre los identificadores de las tres particiones y verifica que ninguno aparece en dos de ellas.

**Por qué.** Una sola fila compartida entre `train` y `test` basta para que la métrica final quede inflada. Es el error más caro del oficio porque no produce ningún síntoma visible: el modelo simplemente «parece» mejor de lo que es.

```bash
neural-labs audit --lab 05_lstm_time_series --quick --split-seed 42
```

**Cómo sabes que salió bien.** La auditoría reporta cero solapamientos. Si reporta alguno, no sigas: el resto del laboratorio no significaría nada.

### Paso 3 — Mirar los datos antes de modelarlos

**Qué ocurre.** Genera el informe de calidad (valores faltantes, balance de clases, rangos) y el de deriva entre particiones.

**Por qué.** Un desbalance fuerte o una diferencia de distribución entre `train` y `test` cambia qué métrica tiene sentido leer. Descubrirlo después de entrenar obliga a repetirlo todo.

```bash
neural-labs quality --lab 05_lstm_time_series --quick --split-seed 42
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
python labs/05_lstm_time_series/train.py --quick
# o, con control explícito de las dos semillas:
neural-labs train --lab 05_lstm_time_series --config baseline --split-seed 42 --training-seed 43
```

**Cómo sabes que salió bien.** En `runs/05_lstm_time_series/<ejecución>/` aparecen `history.csv` y `best_model.pt`; la métrica de validación mejora respecto de la primera época.

### Paso 6 — Compararte con la línea base

**Qué ocurre.** El repositorio entrena por su cuenta **Persistencia, media móvil y Ridge** y guarda su resultado, primero sobre `validation` y —solo al final— sobre `test`.

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
neural-labs benchmark --lab 05_lstm_time_series --quick --split-seed 42 --training-seeds 41 42 43
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
| `model_spec.json` · `inference_contract.json` | Qué entrada espera el modelo y qué devuelve: lo que necesita quien lo despliegue. |
| `model_card.md` · `report.md` | La ficha del modelo y el informe legible de la ejecución. |

## ⚠️ Dónde suele perderse la gente

- **`--quick` no es una versión pequeña del resultado, es una prueba de que todo corre.** En esta ruta recorta a 1024 ejemplos de entrenamiento · 256 de validación · 256 de test · 2 épocas. Sirve para comprobar la instalación y la descarga; cualquier conclusión sobre el modelo exige la ejecución completa.
- **Cambiar algo después de ver `test` invalida la comparación.** Si al mirar el resultado final se te ocurre una mejora, la ruta correcta es volver a `validation`, decidir allí, y sellar de nuevo.
- **Las dos semillas no son intercambiables.** `--split-seed` cambia *qué datos* caen en cada partición; `--training-seed` cambia *cómo se inicializa y baraja* el entrenamiento. Para comparar modelos se fija la primera y se varía la segunda.
- **No hay `confusion_matrix.png`, y no es un error.** Es una tarea de regresión: no existen clases que confundir.
- **Límite declarado de este dataset.** 8.760 observaciones reales de arriendo de bicicletas y clima en Seúl.

### Riesgos al interpretar los resultados

8.760 observaciones reales de arriendo de bicicletas y clima en Seúl.

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

- Goodfellow, Bengio & Courville — *Deep Learning* (MIT Press 2016), cap. 10 — redes con puertas (LSTM/GRU) y dependencias de largo plazo.
- Hyndman & Athanasopoulos — *Forecasting: Principles and Practice* (3.ª ed., OTexts) — metodología de pronóstico, líneas base y evaluación de series temporales.
- Géron — *Hands-On Machine Learning* (3.ª ed., O'Reilly 2022), cap. 15 — procesamiento de secuencias y pronóstico con RNN/LSTM.
- Hochreiter & Schmidhuber (1997), *Long Short-Term Memory*, Neural Computation — celda LSTM original y solución al gradiente que se desvanece.
- Cho et al. (2014), *Learning Phrase Representations using RNN Encoder-Decoder (GRU)*, EMNLP — unidad recurrente con puertas simplificada.
- Fuente del dataset: https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand
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
| [🔁 RNN para texto](../../labs/04_rnn_sequences/README.md) | [Las 31 rutas](../../parts/README.md) | [🧬 Autoencoder para fraude](../../labs/06_autoencoder_anomaly/README.md) |

**En este laboratorio:** **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🔵 [Parte 2 — Arquitecturas según la forma del dato](../../parts/02-arquitecturas.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/05_lstm_time_series/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
