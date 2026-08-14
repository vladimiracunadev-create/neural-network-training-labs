# Perceptrón con PyTorch

<!-- nav-top -->
> 🧭 **Ruta 2 / 31** · 🟢 [Parte 1 — Fundamentos: de la derivada a la primera red](../../parts/01-fundamentos.md)
>
> [⬅️ 🔢 Neurona con NumPy](../../labs/00_numpy_neuron/README.md) · [🏠 Índice de rutas](../../parts/README.md) · [🌀 MLP multiclase ➡️](../../labs/02_mlp_nonlinear/README.md)
>
> **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

## 🎯 Qué vas a hacer aquí

Aprender tensores, autograd, optimizadores y un clasificador lineal.

Es la **ruta 2 de 31** del recorrido y pertenece a 🟢 la parte 1, *Fundamentos: de la derivada a la primera red*. Llegas desde **Neurona con NumPy** y lo que hagas aquí lo da por supuesto **MLP multiclase**.

Trabajarás con el dataset **`banknote_authentication`** (UCI, licencia: Consultar ficha UCI), y tendrás que superar la línea base **Regresión logística**, decidiendo con la métrica `f1` medida sobre `validation`. Nivel fundamentos, unas **4 horas** de dedicación.

**Lo que conviene traer resuelto de las rutas anteriores:** Python básico, NumPy, álgebra lineal elemental.

**Al terminar deberías ser capaz de:**

- Aprender tensores, autograd, optimizadores y un clasificador lineal.
- Preparar y auditar el dataset real banknote_authentication sin fuga de datos.
- Entrenar y evaluar clasificador lineal con autograd.
- Comparar contra la línea base: Regresión logística.
- Interpretar intervalos de confianza, errores y limitaciones.

## 🧠 La teoría de este laboratorio

Esta sección es la explicación completa del tema. No hace falta abrir otro archivo para entender lo que viene después: aquí está la idea, la matemática que la sostiene y sus límites. (El mismo texto vive en `theory.md`, que es la fuente desde la que se genera esta guía, junto con la bibliografía del final.)

### De qué trata

Este laboratorio estudia **clasificador lineal con autograd** usando `banknote_authentication`, un dataset público real procedente de UCI.

El salto respecto al laboratorio anterior no está en las matemáticas —seguimos entrenando esencialmente una neurona logística— sino en la ingeniería: en lugar de derivar los gradientes a mano, dejamos que PyTorch los construya automáticamente. Cada operación sobre un tensor con `requires_grad=True` se registra en un grafo de cómputo dinámico; al invocar `loss.backward()`, el motor `autograd` recorre ese grafo en sentido inverso aplicando la regla de la cadena y deposita en `.grad` exactamente las mismas derivadas que en el lab 00 escribimos manualmente.

El problema —distinguir billetes auténticos de falsos a partir de cuatro estadísticos extraídos por transformada wavelet de imágenes reales— es casi linealmente separable, lo que lo hace ideal para verificar que la maquinaria (tensores, `Dataset`/`DataLoader`, optimizador, bucle de entrenamiento) funciona antes de abordar problemas donde un solo hiperplano ya no basta. La pregunta crítica del laboratorio anticipa precisamente esa limitación.

### La matemática, paso a paso

El modelo calcula un **logit** —una puntuación real sin normalizar— mediante una transformación afín de las entradas:

z = x·W + b

Nótese que ahora trabajamos con lotes (batches): x es una matriz de forma (N, d) y la multiplicación x·W produce un vector de N logits en paralelo, aprovechando el álgebra matricial vectorizada. La probabilidad se obtiene, igual que antes, con la sigmoide p = σ(z) = 1 / (1 + e⁻ᶻ), pero aquí introducimos una diferencia importante de estabilidad numérica.

En lugar de calcular σ(z) y luego la entropía cruzada por separado, PyTorch ofrece `BCEWithLogitsLoss`, que **fusiona la sigmoide y la log-verosimilitud en una sola operación numéricamente estable**. La razón es que combinar exponencial y logaritmo por separado desborda con logits grandes; la forma fusionada aplica el truco log-sum-exp:

L = (1/N) Σᵢ [ max(zᵢ, 0) − zᵢ·yᵢ + ln(1 + e^(−|zᵢ|)) ]

que es algebraicamente idéntica a −(1/N) Σᵢ [ yᵢ·ln σ(zᵢ) + (1−yᵢ)·ln(1−σ(zᵢ)) ] pero no produce ni ∞ ni NaN en los extremos. Por eso la buena práctica es que la última capa devuelva **logits crudos** y la pérdida se encargue internamente de la sigmoide.

La magia de `autograd` es que, definida L, no necesitamos escribir ∂L/∂W. El grafo sabe que ∂L/∂z = (σ(z) − y)/N y propaga hacia atrás por la regla de la cadena hasta ∂L/∂W = xᵀ·(σ(z) − y)/N. El optimizador (por ejemplo SGD o Adam) consume esos gradientes y actualiza los parámetros θ ← θ − η·∇_θ L, encapsulando la regla de actualización que en el lab 00 escribíamos línea por línea. El ciclo canónico es: `optimizer.zero_grad()` → `loss.backward()` → `optimizer.step()`; olvidar el `zero_grad` acumula gradientes de iteraciones previas, un error clásico que el laboratorio permite observar.

La formulación debe conectarse con cuatro elementos: representación de entrada, función del modelo, función de pérdida y regla de actualización. El notebook muestra las dimensiones de los tensores y conserva la misma implementación que el script de terminal.

### Qué es realmente `autograd`: diferenciación en modo inverso

Decir que «el grafo calcula la derivada sola» esconde el mecanismo, y conviene abrirlo porque explica el costo de todo entrenamiento posterior.

Al ejecutar el paso hacia adelante, PyTorch construye un **grafo acíclico dirigido** donde cada nodo es una operación y guarda lo necesario para derivarse. No es simbólico —no manipula fórmulas— ni numérico —no usa diferencias finitas—: es **diferenciación automática**, que aplica la regla de la cadena sobre operaciones elementales cuyas derivadas están programadas exactamente.

La regla de la cadena se puede recorrer en dos sentidos, y la elección importa. Para una función f: ℝⁿ → ℝᵐ, el modo **directo** propaga derivadas desde las entradas y calcula una columna del jacobiano por pasada, así que cuesta n pasadas. El modo **inverso** propaga desde la salida y calcula una fila por pasada: cuesta m pasadas. En una red neuronal, n son los parámetros —millones— y m es 1, porque la pérdida es un escalar. De ahí que el modo inverso sea el correcto: **una sola** pasada hacia atrás produce el gradiente respecto de todos los parámetros a la vez.

El resultado práctico es la regla de oro del costo: el paso hacia atrás cuesta aproximadamente **el doble** que el paso hacia adelante, sin importar cuántos parámetros haya. Lo que sí crece con la profundidad es la memoria, porque hay que conservar las activaciones intermedias hasta que el gradiente pase por ellas. Esa es la razón de que el consumo de memoria escale con el tamaño de lote y con el número de capas, y de que existan técnicas como el *gradient checkpointing*, que recalcula activaciones en vez de guardarlas.

Cada nodo no almacena una matriz jacobiana, que sería inmanejable, sino la operación **producto vector-jacobiano**: dado el gradiente que llega desde arriba, v, devuelve vᵀ·J sin construir J. Para la capa lineal z = x·W + b eso se traduce en las tres expresiones que el laboratorio puede verificar a mano:

∂L/∂W = xᵀ·(∂L/∂z),   ∂L/∂b = Σ_filas (∂L/∂z),   ∂L/∂x = (∂L/∂z)·Wᵀ.

La tercera es la que permite encadenar capas: es el gradiente que esta capa entrega a la anterior.

### Por qué hay que llamar a `zero_grad`, y qué dice el minilote

El detalle que más errores causa tiene una explicación de diseño. PyTorch **acumula** gradientes en el atributo `.grad` en lugar de sobrescribirlos, es decir, `backward()` hace `p.grad += nuevo` y no `p.grad = nuevo`. Eso es deliberado: permite sumar gradientes de varios pasos hacia atrás antes de actualizar, que es exactamente lo que se necesita para simular un lote grande sin memoria para él (*gradient accumulation*), o para redes con varias salidas. El precio es que, en el bucle normal, olvidar `optimizer.zero_grad()` hace que la actualización de la iteración k use la suma de los gradientes de las iteraciones 1..k, un error que no lanza excepción y solo se manifiesta como un entrenamiento que no converge.

El otro concepto que aparece aquí por primera vez es el **minilote**. El gradiente sobre un lote de tamaño B es un estimador **insesgado** del gradiente sobre todo el conjunto: si se muestrea uniformemente, 𝔼[∇L_B] = ∇L. Su varianza, en cambio, decrece como σ²/B, así que la desviación típica del estimador baja con **√B**. De ahí se sigue el compromiso que gobierna la elección del tamaño de lote: cuadruplicar B cuesta cuatro veces más cómputo por paso y solo reduce el ruido a la mitad. Lotes pequeños dan pasos ruidosos —y ese ruido, lejos de ser solo un defecto, ayuda a escapar de mínimos estrechos—; lotes grandes dan direcciones precisas pero aprovechan peor el cómputo y suelen necesitar una tasa de aprendizaje mayor para avanzar lo mismo.

Ese es también el motivo por el que el orden en que se barajan los ejemplos forma parte de `training_seed` y no de `split_seed`: no cambia qué datos hay en cada partición, cambia la trayectoria del entrenamiento.

> **La pregunta que deberías poder responder al terminar:** ¿Qué ejemplos no puede separar un único hiperplano?

### Qué se mide y con qué se decide

El laboratorio reporta `accuracy`, `balanced_accuracy`, `precision`, `recall`, `f1`, `roc_auc`, `pr_auc`. De todas ellas, la que **decide** qué modelo se conserva es `f1`, y se mide siempre sobre `validation`: es la única forma de que `test` siga siendo una estimación honesta de lo que pasará con datos nuevos.

## 🖥️ Los comandos, explicados

Todo el laboratorio se maneja con una sola herramienta de terminal, `neural-labs`, que se instala junto con el paquete (`pip install -e ".[dev,notebooks]"`). Cada subcomando hace **una** cosa del protocolo, y por eso se pueden ejecutar por separado: preparar datos, auditar la partición, entrenar, repetir con varias semillas.

La forma general es siempre la misma:

```bash
neural-labs <subcomando> --lab <identificador> [opciones]
```

| Opción | Valor por defecto | Valores | Qué hace y cuándo cambiarla |
|---|---|---|---|
| `--lab` | `01_pytorch_perceptron` | obligatorio | Qué laboratorio se ejecuta. Solo acepta los identificadores del catálogo. |
| `--quick` | desactivado | — | Usa una fracción real del dataset y pocas épocas. Sirve para comprobar la instalación, no para concluir nada sobre el modelo. |
| `--split-seed N` | `42` | entero | Semilla que decide **qué ejemplo cae en qué partición**. Se mantiene fija al comparar modelos. |
| `--training-seed N` | `42` | entero | Semilla de la inicialización de pesos y del barajado de lotes. Es la que se varía para medir cuánta diferencia es simple azar. |
| `--config` | `baseline` | `baseline` · `improved` | Cuál de las dos configuraciones del laboratorio se usa. |
| `--device` | `auto` | `auto` · `cpu` · `cuda` · `mps` | Dónde entrenar. `auto` elige GPU si está disponible y cae a CPU si no. |
| `--training-seeds A B C` | `41 42 43` | enteros | Solo en `benchmark`: la lista de semillas de entrenamiento que se repiten. |
| `--output-dir` | `runs` | ruta | Dónde se escribe el directorio de la ejecución. |

### El script del laboratorio

`labs/01_pytorch_perceptron/train.py` no es un programa distinto: fija el `--lab` y delega en la misma herramienta, de modo que estas dos líneas hacen exactamente lo mismo.

```bash
python labs/01_pytorch_perceptron/train.py --quick
neural-labs train --lab 01_pytorch_perceptron --quick
```

### Lo mismo desde Python

Si prefieres trabajar en un cuaderno o llamar al laboratorio desde tu propio código, la misma ejecución se lanza así. La función devuelve un objeto con el directorio de la ejecución, las métricas y el historial ya cargados:

```python
from neural_labs.experiments import run_lab

resultado = run_lab(
    "01_pytorch_perceptron",
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

datos = prepare_dataset("01_pytorch_perceptron", quick=True, seed=42)
print(datos.summary)       # tamaño de cada partición y metadatos de la fuente
```

## 🪜 Paso a paso

Cada paso dice qué ocurre por dentro, por qué se hace en ese orden y cómo comprobar que salió bien. El orden no es una convención de estilo: es el que ejecuta el código, y alterarlo invalida el resultado.

### Paso 1 — Traer el dataset real y partirlo

**Qué ocurre.** Descarga `banknote_authentication` desde su proveedor y construye las tres particiones —`train`, `validation` y `test`— con la semilla de partición que le pases.

**Por qué.** La partición se fija **antes** de entrenar y su semilla (`--split-seed`) es independiente de la del entrenamiento (`--training-seed`). Si ambas fueran la misma, no podrías distinguir si un cambio en el resultado viene de haber repartido los datos de otro modo o de haber inicializado los pesos de otro modo.

```bash
neural-labs dataset --lab 01_pytorch_perceptron --quick --split-seed 42
```

**Cómo sabes que salió bien.** El comando termina sin error y deja el dataset en caché. Si no hay red, **falla**: no se rellena con datos inventados (`data/dataset.yaml` declara `fallback_to_generated_data: false`).

### Paso 2 — Comprobar que las particiones no se tocan

**Qué ocurre.** Recorre los identificadores de las tres particiones y verifica que ninguno aparece en dos de ellas.

**Por qué.** Una sola fila compartida entre `train` y `test` basta para que la métrica final quede inflada. Es el error más caro del oficio porque no produce ningún síntoma visible: el modelo simplemente «parece» mejor de lo que es.

```bash
neural-labs audit --lab 01_pytorch_perceptron --quick --split-seed 42
```

**Cómo sabes que salió bien.** La auditoría reporta cero solapamientos. Si reporta alguno, no sigas: el resto del laboratorio no significaría nada.

### Paso 3 — Mirar los datos antes de modelarlos

**Qué ocurre.** Genera el informe de calidad (valores faltantes, balance de clases, rangos) y el de deriva entre particiones.

**Por qué.** Un desbalance fuerte o una diferencia de distribución entre `train` y `test` cambia qué métrica tiene sentido leer. Descubrirlo después de entrenar obliga a repetirlo todo.

```bash
neural-labs quality --lab 01_pytorch_perceptron --quick --split-seed 42
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
python labs/01_pytorch_perceptron/train.py --quick
# o, con control explícito de las dos semillas:
neural-labs train --lab 01_pytorch_perceptron --config baseline --split-seed 42 --training-seed 43
```

**Cómo sabes que salió bien.** En `runs/01_pytorch_perceptron/<ejecución>/` aparecen `history.csv` y `best_model.pt`; la métrica de validación mejora respecto de la primera época.

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
neural-labs benchmark --lab 01_pytorch_perceptron --quick --split-seed 42 --training-seeds 41 42 43
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
- **Límite declarado de este dataset.** Características extraídas de imágenes reales de billetes.

### Riesgos al interpretar los resultados

Características extraídas de imágenes reales de billetes.

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

- Géron — *Hands-On Machine Learning* (3.ª ed., O'Reilly 2022), cap. 10 — introducción a redes con Keras/PyTorch, logits y funciones de pérdida.
- Goodfellow, Bengio & Courville — *Deep Learning* (MIT Press 2016), cap. 6 — redes hacia adelante, entropía cruzada y descenso de gradiente basado en grafos.
- Zhang et al. — *Dive into Deep Learning* (d2l.ai, 2023), cap. 3–5 — regresión lineal/softmax en frameworks modernos y mecánica de entrenamiento.
- Paszke et al. (2019), *PyTorch: An Imperative Style, High-Performance Deep Learning Library*, NeurIPS — diseño del framework y del motor de diferenciación automática.
- Documentación oficial de PyTorch (autograd) — https://pytorch.org/docs/stable/notes/autograd.html — grafo dinámico y semántica de `backward()`.
- Fuente del dataset: https://archive.ics.uci.edu/dataset/267/banknote+authentication
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
| [🔢 Neurona con NumPy](../../labs/00_numpy_neuron/README.md) | [Las 31 rutas](../../parts/README.md) | [🌀 MLP multiclase](../../labs/02_mlp_nonlinear/README.md) |

**En este laboratorio:** **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🟢 [Parte 1 — Fundamentos: de la derivada a la primera red](../../parts/01-fundamentos.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/01_pytorch_perceptron/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
