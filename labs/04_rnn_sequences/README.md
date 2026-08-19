# RNN para texto

<!-- nav-top -->
> 🧭 **Ruta 5 / 31** · 🔵 [Parte 2 — Arquitecturas según la forma del dato](../../parts/02-arquitecturas.md)
>
> [⬅️ 🖼️ CNN para visión](../../labs/03_cnn_vision/README.md) · [🏠 Índice de rutas](../../parts/README.md) · [📈 LSTM para series temporales ➡️](../../labs/05_lstm_time_series/README.md)
>
> **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

## 🎯 Qué vas a hacer aquí

Clasificar sentimiento en reseñas reales usando embeddings y recurrencia.

Es la **ruta 5 de 31** del recorrido y pertenece a 🔵 la parte 2, *Arquitecturas según la forma del dato*. Llegas desde **CNN para visión** y lo que hagas aquí lo da por supuesto **LSTM para series temporales**.

Trabajarás con el dataset **`imdb`** (Hugging Face / Stanford, licencia: Consultar dataset card), y tendrás que superar la línea base **TF-IDF + regresión logística**, decidiendo con la métrica `f1` medida sobre `validation`. Nivel intermedio, unas **6 horas** de dedicación.

**Lo que conviene traer resuelto de las rutas anteriores:** PyTorch básico, particiones train/validation/test, métricas de evaluación.

**Al terminar deberías ser capaz de:**

- Clasificar sentimiento en reseñas reales usando embeddings y recurrencia.
- Preparar y auditar el dataset real imdb sin fuga de datos.
- Entrenar y evaluar recurrencia sobre secuencias tokenizadas.
- Comparar contra la línea base: TF-IDF + regresión logística.
- Interpretar intervalos de confianza, errores y limitaciones.

## 🧠 La teoría de este laboratorio

Esta sección es la explicación completa del tema. No hace falta abrir otro archivo para entender lo que viene después: aquí está la idea, la matemática que la sostiene y sus límites. (El mismo texto vive en `theory.md`, que es la fuente desde la que se genera esta guía, junto con la bibliografía del final.)

### De qué trata

Este laboratorio estudia **recurrencia sobre secuencias tokenizadas** usando `imdb`, un dataset público real procedente de Hugging Face / Stanford.

El texto es una secuencia de longitud variable donde el orden importa: "no es buena" y "es buena, no" significan cosas distintas. Un MLP o una CNN de tamaño fijo no capturan bien esa dependencia temporal. La red recurrente (RNN) procesa la secuencia token a token manteniendo un **estado oculto** que resume todo lo visto hasta el momento, de modo que la predicción en cada paso depende del contexto acumulado. Es, en esencia, una memoria que se actualiza con cada palabra.

Dos piezas colaboran aquí. Primero, los **embeddings**: cada token del vocabulario se representa por un vector denso aprendible, de forma que palabras con uso similar acaban cerca en el espacio vectorial (la idea que popularizó word2vec). Segundo, la **recurrencia**, que integra esos vectores en el tiempo. El laboratorio clasifica el sentimiento de reseñas de cine reales y se contrasta contra un TF-IDF + regresión logística, que ignora el orden; la comparación revela cuándo la estructura secuencial realmente aporta.

### La matemática, paso a paso

Una RNN recibe la secuencia de vectores de embedding x₁, x₂, …, x_T (uno por token) y mantiene un estado oculto hₜ que se recalcula en cada paso combinando la entrada actual con el estado anterior:

hₜ = tanh(Wₓ·xₜ + W_h·hₜ₋₁ + b)

La intuición es una recursión: hₜ es una síntesis comprimida de todo el prefijo x₁…xₜ. La matriz Wₓ decide cómo entra la nueva palabra, W_h decide cómo se transforma la memoria previa, y la **tanh** (con rango en (−1, 1)) mantiene el estado acotado e introduce la no linealidad. Un detalle esencial es que Wₓ, W_h y b **se comparten en todos los pasos de tiempo**: es el mismo conjunto de pesos aplicado en cada instante, análogo al peso compartido de las CNN pero a lo largo del eje temporal. Esto permite procesar secuencias de cualquier longitud con un número fijo de parámetros. Para clasificar sentimiento se usa el último estado h_T (o un agregado de todos), que pasa por una capa densa y una sigmoide para dar la probabilidad de reseña positiva.

El entrenamiento usa **retropropagación en el tiempo** (BPTT): se "desenrolla" la red en T copias y el gradiente fluye hacia atrás desde h_T hasta h₁. Aquí surge el problema fundamental de las RNN simples. Al aplicar la regla de la cadena a lo largo de T pasos, el gradiente respecto a estados lejanos contiene un producto de T factores del tipo W_h⊤ diag(tanh′). Si los valores propios dominantes de ese producto son menores que 1, el gradiente **se desvanece** exponencialmente (∝ λᵀ con λ < 1) y la red no aprende dependencias largas; si son mayores que 1, **explota**.

∂h_T/∂h₁ = Πₜ (∂hₜ/∂hₜ₋₁)  →  se contrae o crece exponencialmente con T

Este es exactamente el motivo por el que las reseñas se truncan y por el que existen arquitecturas con puertas (LSTM/GRU), tema del laboratorio siguiente. La explosión se mitiga en la práctica con **recorte de gradiente** (gradient clipping), que limita la norma del gradiente a un umbral antes de actualizar; el desvanecimiento, en cambio, exige cambiar la propia celda recurrente.

La formulación debe conectarse con cuatro elementos: representación de entrada, función del modelo, función de pérdida y regla de actualización. El notebook muestra las dimensiones de los tensores y conserva la misma implementación que el script de terminal.

### Por qué el gradiente se desvanece: el producto de jacobianos

«El gradiente se desvanece» suele enunciarse como una advertencia vaga. Tiene una causa exacta y se puede escribir. Al derivar la pérdida en el paso T respecto del estado en un paso anterior t, la regla de la cadena encadena todos los pasos intermedios:

∂h_T/∂h_t = Π_(k=t+1..T) ∂h_k/∂h_(k−1) = Π_(k=t+1..T) W_hᵀ · diag( σ′(z_k) ).

Es un **producto de T − t matrices**. Y los productos largos de matrices tienen un comportamiento extremo: acotando la norma, ‖∂h_T/∂h_t‖ ≤ (‖W_h‖·γ)^(T−t), donde γ es la cota de la derivada de la activación —γ = 1 para tanh, y solo en el origen—. Si ‖W_h‖·γ < 1, esa potencia tiende a **cero exponencialmente** con la distancia temporal; si es mayor que 1, **explota** igual de rápido.

La consecuencia práctica es que el gradiente que llega desde el final de una reseña hasta sus primeras palabras se ha multiplicado por sí mismo decenas de veces. Con 200 tokens, un factor de 0,9 por paso deja 0,9²⁰⁰ ≈ 7·10⁻¹⁰: la señal es indistinguible de cero en punto flotante. La red no es que «prefiera» el corto plazo, es que **físicamente no recibe** información sobre el largo plazo.

Los dos extremos exigen remedios distintos y asimétricos. La explosión se ataca con **recorte de gradiente**: si la norma global supera un umbral τ, se reescala todo el gradiente conservando su dirección,

g ← g · min(1, τ / ‖g‖).

Es barato, no cambia la dirección de descenso y basta para evitar que un solo paso destruya el modelo; el repositorio lo aplica por defecto con `gradient_clip_norm`. El desvanecimiento, en cambio, **no tiene un remedio análogo**: no se puede amplificar una señal que ya se perdió. Solo se arregla cambiando la arquitectura para que exista un camino por el que el gradiente fluya sin multiplicarse —las puertas de la LSTM de la ruta siguiente, o la atención directa entre posiciones de la ruta 07—.

Un remedio parcial que sí ayuda es la **inicialización ortogonal** de W_h. Una matriz ortogonal tiene todos sus valores singulares iguales a 1, así que al inicio del entrenamiento no amplifica ni atenúa; el problema reaparece a medida que los pesos se alejan de esa condición, pero el arranque es mucho más sano.

### El embedding, el padding y lo que hay que enmascarar

Antes de la recurrencia hay una capa que suele pasar desapercibida y que concentra la mayor parte de los parámetros. La **capa de embedding** es una tabla de búsqueda: una matriz E ∈ ℝ^(V×d) con una fila por palabra del vocabulario, de la que el modelo simplemente **selecciona** la fila correspondiente a cada token. Es equivalente a multiplicar por un vector one-hot, e_i = onehot(i)·E, pero se implementa como indexación porque el producto sería un desperdicio.

Su tamaño domina el modelo: con un vocabulario de 20 000 palabras y d = 100, son 2 000 000 de parámetros, frente a los pocos miles de la capa recurrente. De ahí que el recorte del vocabulario sea la decisión de diseño con más impacto en el tamaño del modelo, y que las palabras raras se mapeen a un token `<unk>` en lugar de recibir cada una su fila.

Las reseñas tienen longitudes distintas y los tensores son rectangulares, así que se **rellenan** hasta una longitud común con un token `<pad>`. Ese relleno crea dos obligaciones que, si se olvidan, degradan el modelo sin dar error:

1. El embedding de `<pad>` debe quedar fijo en cero y sin gradiente, para que el relleno no aporte señal.
2. El estado que se usa para clasificar no puede ser h_T del tensor rellenado, porque para una reseña corta ese estado corresponde a haber procesado decenas de `<pad>`. Hay que tomar el estado en la **longitud real** de cada secuencia, o promediar solo sobre las posiciones válidas.

Ese segundo punto es el error más común del laboratorio: produce un modelo que funciona peor con las reseñas cortas y no lo dice.

Sobre qué representación pasar al clasificador hay dos opciones con lógicas distintas. Tomar el **último estado** h_T asume que la recurrencia ha resumido bien todo el prefijo, y sufre justo del desvanecimiento descrito arriba. Promediar los estados de todas las posiciones —*mean pooling* sobre la máscara— da a cada token el mismo peso y suele ser más robusto en clasificación de sentimiento, donde la evidencia puede estar en cualquier parte de la reseña. La **RNN bidireccional** resuelve la asimetría ejecutando dos recurrencias, una en cada sentido, y concatenando sus estados: duplica los parámetros recurrentes y permite que cada posición vea contexto por ambos lados, pero solo es lícita cuando la secuencia completa está disponible de antemano —cierto en una reseña, falso en una predicción en tiempo real—.

> **La pregunta que deberías poder responder al terminar:** ¿Qué información se pierde al truncar las reseñas?

### Qué se mide y con qué se decide

El laboratorio reporta `accuracy`, `balanced_accuracy`, `precision`, `recall`, `f1`, `roc_auc`, `pr_auc`. De todas ellas, la que **decide** qué modelo se conserva es `f1`, y se mide siempre sobre `validation`: es la única forma de que `test` siga siendo una estimación honesta de lo que pasará con datos nuevos.

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
jupyter lab labs/04_rnn_sequences/notebook.ipynb
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
| `--lab` | `04_rnn_sequences` | obligatorio | Qué laboratorio se ejecuta. Solo acepta los identificadores del catálogo. |
| `--quick` | desactivado | — | Usa una fracción real del dataset y pocas épocas. Sirve para comprobar la instalación, no para concluir nada sobre el modelo. |
| `--split-seed N` | `42` | entero | Semilla que decide **qué ejemplo cae en qué partición**. Se mantiene fija al comparar modelos. |
| `--training-seed N` | `42` | entero | Semilla de la inicialización de pesos y del barajado de lotes. Es la que se varía para medir cuánta diferencia es simple azar. |
| `--config` | `baseline` | `baseline` · `improved` | Cuál de las dos configuraciones del laboratorio se usa. |
| `--device` | `auto` | `auto` · `cpu` · `cuda` · `mps` | Dónde entrenar. `auto` elige GPU si está disponible y cae a CPU si no. |
| `--training-seeds A B C` | `41 42 43` | enteros | Solo en `benchmark`: la lista de semillas de entrenamiento que se repiten. |
| `--output-dir` | `runs` | ruta | Dónde se escribe el directorio de la ejecución. |

### El script del laboratorio

`labs/04_rnn_sequences/train.py` no es un programa distinto: fija el `--lab` y delega en la misma herramienta, de modo que estas dos líneas hacen exactamente lo mismo.

```bash
python labs/04_rnn_sequences/train.py --quick
neural-labs train --lab 04_rnn_sequences --quick
```

### Lo mismo desde Python

Si prefieres trabajar en un cuaderno o llamar al laboratorio desde tu propio código, la misma ejecución se lanza así. La función devuelve un objeto con el directorio de la ejecución, las métricas y el historial ya cargados:

```python
from neural_labs.experiments import run_lab

resultado = run_lab(
    "04_rnn_sequences",
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

datos = prepare_dataset("04_rnn_sequences", quick=True, seed=42)
print(datos.summary)       # tamaño de cada partición y metadatos de la fuente
```

## 🪜 Paso a paso

Cada paso dice qué ocurre por dentro, por qué se hace en ese orden y cómo comprobar que salió bien. El orden no es una convención de estilo: es el que ejecuta el código, y alterarlo invalida el resultado.

### Paso 1 — Traer el dataset real y partirlo

**Qué ocurre.** Descarga `imdb` desde su proveedor y construye las tres particiones —`train`, `validation` y `test`— con la semilla de partición que le pases.

**Por qué.** La partición se fija **antes** de entrenar y su semilla (`--split-seed`) es independiente de la del entrenamiento (`--training-seed`). Si ambas fueran la misma, no podrías distinguir si un cambio en el resultado viene de haber repartido los datos de otro modo o de haber inicializado los pesos de otro modo.

```bash
neural-labs dataset --lab 04_rnn_sequences --quick --split-seed 42
```

**Cómo sabes que salió bien.** El comando termina sin error y deja el dataset en caché. Si no hay red, **falla**: no se rellena con datos inventados (`data/dataset.yaml` declara `fallback_to_generated_data: false`).

### Paso 2 — Comprobar que las particiones no se tocan

**Qué ocurre.** Recorre los identificadores de las tres particiones y verifica que ninguno aparece en dos de ellas.

**Por qué.** Una sola fila compartida entre `train` y `test` basta para que la métrica final quede inflada. Es el error más caro del oficio porque no produce ningún síntoma visible: el modelo simplemente «parece» mejor de lo que es.

```bash
neural-labs audit --lab 04_rnn_sequences --quick --split-seed 42
```

**Cómo sabes que salió bien.** La auditoría reporta cero solapamientos. Si reporta alguno, no sigas: el resto del laboratorio no significaría nada.

### Paso 3 — Mirar los datos antes de modelarlos

**Qué ocurre.** Genera el informe de calidad (valores faltantes, balance de clases, rangos) y el de deriva entre particiones.

**Por qué.** Un desbalance fuerte o una diferencia de distribución entre `train` y `test` cambia qué métrica tiene sentido leer. Descubrirlo después de entrenar obliga a repetirlo todo.

```bash
neural-labs quality --lab 04_rnn_sequences --quick --split-seed 42
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
python labs/04_rnn_sequences/train.py --quick
# o, con control explícito de las dos semillas:
neural-labs train --lab 04_rnn_sequences --config baseline --split-seed 42 --training-seed 43
```

**Cómo sabes que salió bien.** En `runs/04_rnn_sequences/<ejecución>/` aparecen `history.csv` y `best_model.pt`; la métrica de validación mejora respecto de la primera época.

### Paso 6 — Compararte con la línea base

**Qué ocurre.** El repositorio entrena por su cuenta **TF-IDF + regresión logística** y guarda su resultado, primero sobre `validation` y —solo al final— sobre `test`.

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
neural-labs benchmark --lab 04_rnn_sequences --quick --split-seed 42 --training-seeds 41 42 43
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
- **Límite declarado de este dataset.** Reseñas cinematográficas reales con partición oficial.

### Riesgos al interpretar los resultados

Reseñas cinematográficas reales con partición oficial.

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

- Goodfellow, Bengio & Courville — *Deep Learning* (MIT Press 2016), cap. 10 — redes recurrentes, BPTT y el problema del gradiente que se desvanece.
- Géron — *Hands-On Machine Learning* (3.ª ed., O'Reilly 2022), cap. 16 — procesamiento de lenguaje con RNN y embeddings.
- Prince — *Understanding Deep Learning* (MIT Press 2024), cap. 12 — modelado de secuencias y arquitecturas recurrentes.
- Elman (1990), *Finding Structure in Time*, Cognitive Science — RNN fundacional con estado oculto recurrente.
- Mikolov et al. (2013), *Efficient Estimation of Word Representations in Vector Space (word2vec)* — embeddings distribuidos de palabras.
- Fuente del dataset: https://huggingface.co/datasets/stanfordnlp/imdb — **IMDB Large Movie Review Dataset** (Stanford AI Lab, La ficha de Hugging Face declara `other`); procedencia, versión y SHA-256 en el registro de fuentes, entrada `imdb-large-movie-review` — esta clase la usa para clasificar sentimiento en reseñas reales con embeddings y recurrencia, respetando la partición oficial.
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
| [🖼️ CNN para visión](../../labs/03_cnn_vision/README.md) | [Las 31 rutas](../../parts/README.md) | [📈 LSTM para series temporales](../../labs/05_lstm_time_series/README.md) |

**En este laboratorio:** **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🔵 [Parte 2 — Arquitecturas según la forma del dato](../../parts/02-arquitecturas.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/04_rnn_sequences/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
