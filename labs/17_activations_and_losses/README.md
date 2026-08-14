# Activaciones y funciones de pérdida

<!-- nav-top -->
> 🧭 **Ruta 18 / 31** · 🔴 [Parte 5 — La mecánica fina, ahora en profundidad](../../parts/05-mecanica-fina.md)
>
> [⬅️ ∂ Backpropagation manual](../../labs/16_backpropagation_manual/README.md) · [🏠 Índice de rutas](../../parts/README.md) · [⚙️ Optimizadores y schedulers ➡️](../../labs/18_optimizers_and_schedulers/README.md)
>
> **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

## 🎯 Qué vas a hacer aquí

Comparar ReLU, GELU, Tanh y pérdidas apropiadas en clases desbalanceadas.

Es la **ruta 18 de 31** del recorrido y pertenece a 🔴 la parte 5, *La mecánica fina, ahora en profundidad*. Llegas desde **Backpropagation manual** y lo que hagas aquí lo da por supuesto **Optimizadores y schedulers**.

Trabajarás con el dataset **`wine_quality`** (UCI, licencia: CC BY 4.0), y tendrás que superar la línea base **Regresión ordinal y Random Forest**, decidiendo con la métrica `macro_f1` medida sobre `validation`. Nivel fundamentos, unas **4 horas** de dedicación.

**Lo que conviene traer resuelto de las rutas anteriores:** Python básico, NumPy, álgebra lineal elemental.

**Al terminar deberías ser capaz de:**

- Comparar ReLU, GELU, Tanh y pérdidas apropiadas en clases desbalanceadas.
- Preparar y auditar el dataset real wine_quality sin fuga de datos.
- Entrenar y evaluar comparación controlada de activaciones y pérdidas.
- Comparar contra la línea base: Regresión ordinal y Random Forest.
- Interpretar intervalos de confianza, errores y limitaciones.

## 🧠 La teoría de este laboratorio

Esta sección es la explicación completa del tema. No hace falta abrir otro archivo para entender lo que viene después: aquí está la idea, la matemática que la sostiene y sus límites. (El mismo texto vive en `theory.md`, que es la fuente desde la que se genera esta guía, junto con la bibliografía del final.)

### De qué trata

Este laboratorio estudia **comparación controlada de activaciones y pérdidas** usando `wine_quality`, un dataset público real procedente de UCI. Dos decisiones de diseño gobiernan cómo aprende una red: qué **función de activación** introduce la no linealidad entre capas y qué **función de pérdida** define qué significa equivocarse. El laboratorio las aísla y las compara de forma controlada, cambiando una variable a la vez para atribuir con honestidad las diferencias de desempeño.

Sobre las **activaciones**: sin no linealidad, apilar capas lineales colapsa en una sola transformación lineal, incapaz de modelar fronteras complejas. Tanh satura en ambos extremos (su derivada tiende a 0 para entradas grandes), lo que frena el aprendizaje en redes profundas por gradientes que se desvanecen. ReLU evita esa saturación en el lado positivo manteniendo la derivada en 1, lo que acelera el entrenamiento y favorece representaciones dispersas, aunque puede "morir" si una neurona queda siempre en la zona negativa. GELU es una alternativa suave que pondera la entrada por su probabilidad bajo una gaussiana, combinando parte de la no saturación de ReLU con una transición diferenciable.

Sobre las **pérdidas**: el dataset de calidad de vino está desbalanceado (hay muchas más muestras de calidad media que de los extremos). La entropía cruzada estándar trata todos los ejemplos por igual y tiende a optimizar la clase mayoritaria, ignorando las minoritarias. La **Focal Loss** reescala la pérdida para bajar el peso de los ejemplos ya bien clasificados y concentrar el aprendizaje en los difíciles. La pregunta crítica —si la conclusión se mantiene en varias semillas— recuerda que en comparaciones finas la diferencia entre dos activaciones puede ser menor que el ruido de entrenamiento.

### La matemática, paso a paso

Una activación transforma cada preactivación z de forma no lineal. Sus definiciones y derivadas explican su comportamiento:

  Tanh:  σ(z) = (eᶻ − e⁻ᶻ)/(eᶻ + e⁻ᶻ),   σ′(z) = 1 − σ(z)²

  ReLU:  σ(z) = max(0, z),   σ′(z) = 1 si z > 0, 0 si z < 0

  GELU:  σ(z) = z · Φ(z),   con Φ la función de distribución acumulada de la normal estándar

La clave está en la derivada, porque es el factor por el que backpropagation multiplica el gradiente al atravesar la capa. Para Tanh, σ′(z) = 1 − σ(z)² tiende a 0 cuando |z| es grande: la neurona **satura** y el gradiente se desvanece. Para ReLU, σ′(z) = 1 en toda la región activa: el gradiente pasa sin atenuarse, lo que combate el desvanecimiento pero deja gradiente nulo (neuronas muertas) cuando z < 0. GELU suaviza esa transición, evitando el corte brusco en z = 0.

Para la salida de clasificación se usa softmax, ŷₖ = e^{zₖ} / Σⱼ e^{zⱼ}, y sobre él se define la pérdida. La **entropía cruzada** para la clase verdadera es:

  CE = −Σₖ yₖ · log ŷₖ

La **Focal Loss** añade un factor modulador (1 − p_t)^γ, donde p_t es la probabilidad asignada a la clase correcta y γ ≥ 0 controla cuánto se atenúan los ejemplos fáciles:

  FL = −α_t · (1 − p_t)^γ · log(p_t)

Cuando el modelo ya acierta con confianza, p_t → 1, el factor (1 − p_t)^γ → 0 y ese ejemplo casi no contribuye al gradiente; los ejemplos difíciles (p_t bajo) conservan casi toda su pérdida. Con γ = 0 la Focal Loss se reduce a la entropía cruzada ponderada. Por eso ayuda en clases desbalanceadas: reorienta la señal de aprendizaje ∇ hacia las clases minoritarias mal clasificadas en vez de reforzar la mayoría ya resuelta. Todo se optimiza con descenso de gradiente, θ ← θ − η · ∇_θ ℒ. La formulación conecta cuatro elementos: representación de entrada, función del modelo (con su activación), función de pérdida (CE o Focal) y regla de actualización (SGD con ∇). El notebook muestra las dimensiones de los tensores y conserva la misma implementación que el script de terminal.

### Las activaciones, comparadas por lo que le hacen al gradiente

Una activación se elige por cómo se comporta su **derivada**, no por la forma de su curva. Puestas una al lado de otra, las diferencias son concretas.

La **sigmoide** tiene derivada σ′(z) = σ(z)·(1 − σ(z)), cuyo máximo es 0,25 en z = 0. Encadenar L capas multiplica L de esos factores, así que el gradiente se atenúa como máximo por 0,25^L: con diez capas, un factor 10⁻⁶ en el mejor de los casos. Además su salida no está centrada en cero —siempre positiva—, lo que hace que todos los gradientes de una misma neurona compartan signo y la optimización avance en zigzag. La **tanh** corrige lo segundo: está centrada en cero y su derivada llega a 1 en el origen, por lo que atenúa menos, pero sigue saturando en ambos extremos.

La **ReLU** cambia el juego porque su derivada es exactamente 1 en toda la región positiva: no atenúa. A cambio es exactamente 0 en la negativa, y de ahí el fallo de la neurona muerta —si la preactivación queda negativa para todos los ejemplos, no vuelve a recibir gradiente nunca—. La **Leaky ReLU** deja pasar una pendiente pequeña α ≈ 0,01 en la zona negativa precisamente para que ese gradiente nunca sea nulo.

La **GELU** sustituye el corte duro por una compuerta suave: GELU(z) = z·Φ(z), donde Φ es la función de distribución acumulada de la normal estándar. En vez de decidir con un umbral, pondera la entrada por la probabilidad de que sea mayor que una variable normal. Su derivada es continua en todo punto —a diferencia de la ReLU, discontinua en 0— y no se anula en la zona negativa cercana al origen, lo que le da gradiente donde la ReLU ya no lo tiene. Es la activación por defecto en los transformers, y en este laboratorio se compara con las anteriores manteniendo todo lo demás fijo.

Una observación que ordena la comparación: lo que se está eligiendo no es «la mejor función», sino **el perfil de gradiente** que la red recibirá en cada capa. Por eso el efecto de la activación depende de la profundidad, y una comparación hecha con una sola capa oculta puede no extrapolarse a una red profunda.

### Qué pérdida usar cuando las clases están desbalanceadas

La segunda mitad del laboratorio trata la otra elección, y aquí el problema es que la entropía cruzada **trata todos los ejemplos por igual** en una situación en que no lo son.

Con una clase mayoritaria que domina el conjunto, la mayor parte del gradiente proviene de ejemplos fáciles y ya bien clasificados de esa clase. El modelo aprende rápido a predecir la mayoría y se estanca en la minoría, no por falta de capacidad sino porque la señal de la minoría queda ahogada. Hay tres respuestas, y conviene entender qué modifica cada una.

Los **pesos por clase** multiplican la contribución de cada ejemplo por un factor w_c inverso a la frecuencia:

ℒ = −(1/N) · Σᵢ w_(yᵢ) · log p_(i,yᵢ).

Es simple y directo, pero trata por igual a todos los ejemplos de la clase minoritaria, incluidos los que ya se clasifican perfectamente.

La **Focal Loss** cambia el eje: en vez de ponderar por clase, pondera por **dificultad**.

ℒ_focal = −(1 − p_t)^γ · log p_t,

donde p_t es la probabilidad asignada a la clase verdadera. El factor (1 − p_t)^γ vale casi 0 cuando el ejemplo ya está bien clasificado —p_t ≈ 1— y casi 1 cuando está mal. Con γ = 2, un ejemplo con p_t = 0,9 ve su pérdida reducida cien veces, mientras que uno con p_t = 0,1 apenas se toca. El efecto es que el gradiente se concentra en lo que el modelo aún no domina, sin necesidad de conocer las frecuencias de clase.

El **remuestreo** actúa antes de la pérdida, replicando ejemplos de la minoría o descartando de la mayoría. Cambia la distribución que ve el modelo, lo que tiene una consecuencia que se olvida a menudo: las probabilidades que produzca quedarán **descalibradas** respecto de la distribución real, y habrá que corregirlas si se van a interpretar como probabilidades. Es justo el problema que aborda la ruta 22.

Y la elección de la pérdida arrastra la elección de la métrica. Optimizar con pesos de clase y reportar exactitud es incoherente: se está pidiendo al modelo que priorice la minoría y midiéndolo con una cifra que premia la mayoría. Por eso este laboratorio decide con `macro_f1`, que promedia por clase y da a la minoritaria el mismo peso que a la mayoritaria.

> **La pregunta que deberías poder responder al terminar:** ¿La conclusión se mantiene en varias semillas?

### Qué se mide y con qué se decide

El laboratorio reporta `accuracy`, `balanced_accuracy`, `macro_f1`. De todas ellas, la que **decide** qué modelo se conserva es `macro_f1`, y se mide siempre sobre `validation`: es la única forma de que `test` siga siendo una estimación honesta de lo que pasará con datos nuevos.

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
jupyter lab labs/17_activations_and_losses/notebook.ipynb
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
| `--lab` | `17_activations_and_losses` | obligatorio | Qué laboratorio se ejecuta. Solo acepta los identificadores del catálogo. |
| `--quick` | desactivado | — | Usa una fracción real del dataset y pocas épocas. Sirve para comprobar la instalación, no para concluir nada sobre el modelo. |
| `--split-seed N` | `42` | entero | Semilla que decide **qué ejemplo cae en qué partición**. Se mantiene fija al comparar modelos. |
| `--training-seed N` | `42` | entero | Semilla de la inicialización de pesos y del barajado de lotes. Es la que se varía para medir cuánta diferencia es simple azar. |
| `--config` | `baseline` | `baseline` · `improved` | Cuál de las dos configuraciones del laboratorio se usa. |
| `--device` | `auto` | `auto` · `cpu` · `cuda` · `mps` | Dónde entrenar. `auto` elige GPU si está disponible y cae a CPU si no. |
| `--training-seeds A B C` | `41 42 43` | enteros | Solo en `benchmark`: la lista de semillas de entrenamiento que se repiten. |
| `--output-dir` | `runs` | ruta | Dónde se escribe el directorio de la ejecución. |

### El script del laboratorio

`labs/17_activations_and_losses/train.py` no es un programa distinto: fija el `--lab` y delega en la misma herramienta, de modo que estas dos líneas hacen exactamente lo mismo.

```bash
python labs/17_activations_and_losses/train.py --quick
neural-labs train --lab 17_activations_and_losses --quick
```

### Lo mismo desde Python

Si prefieres trabajar en un cuaderno o llamar al laboratorio desde tu propio código, la misma ejecución se lanza así. La función devuelve un objeto con el directorio de la ejecución, las métricas y el historial ya cargados:

```python
from neural_labs.experiments import run_lab

resultado = run_lab(
    "17_activations_and_losses",
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

datos = prepare_dataset("17_activations_and_losses", quick=True, seed=42)
print(datos.summary)       # tamaño de cada partición y metadatos de la fuente
```

## 🪜 Paso a paso

Cada paso dice qué ocurre por dentro, por qué se hace en ese orden y cómo comprobar que salió bien. El orden no es una convención de estilo: es el que ejecuta el código, y alterarlo invalida el resultado.

### Paso 1 — Traer el dataset real y partirlo

**Qué ocurre.** Descarga `wine_quality` desde su proveedor y construye las tres particiones —`train`, `validation` y `test`— con la semilla de partición que le pases.

**Por qué.** La partición se fija **antes** de entrenar y su semilla (`--split-seed`) es independiente de la del entrenamiento (`--training-seed`). Si ambas fueran la misma, no podrías distinguir si un cambio en el resultado viene de haber repartido los datos de otro modo o de haber inicializado los pesos de otro modo.

```bash
neural-labs dataset --lab 17_activations_and_losses --quick --split-seed 42
```

**Cómo sabes que salió bien.** El comando termina sin error y deja el dataset en caché. Si no hay red, **falla**: no se rellena con datos inventados (`data/dataset.yaml` declara `fallback_to_generated_data: false`).

### Paso 2 — Comprobar que las particiones no se tocan

**Qué ocurre.** Recorre los identificadores de las tres particiones y verifica que ninguno aparece en dos de ellas.

**Por qué.** Una sola fila compartida entre `train` y `test` basta para que la métrica final quede inflada. Es el error más caro del oficio porque no produce ningún síntoma visible: el modelo simplemente «parece» mejor de lo que es.

```bash
neural-labs audit --lab 17_activations_and_losses --quick --split-seed 42
```

**Cómo sabes que salió bien.** La auditoría reporta cero solapamientos. Si reporta alguno, no sigas: el resto del laboratorio no significaría nada.

### Paso 3 — Mirar los datos antes de modelarlos

**Qué ocurre.** Genera el informe de calidad (valores faltantes, balance de clases, rangos) y el de deriva entre particiones.

**Por qué.** Un desbalance fuerte o una diferencia de distribución entre `train` y `test` cambia qué métrica tiene sentido leer. Descubrirlo después de entrenar obliga a repetirlo todo.

```bash
neural-labs quality --lab 17_activations_and_losses --quick --split-seed 42
```

**Cómo sabes que salió bien.** Obtienes `data_quality.json` y `drift_report.json`; ábrelos antes de decidir la configuración.

### Paso 4 — Estudiar la teoría del laboratorio

**Qué ocurre.** Leer [`theory.md`](theory.md): la idea central, el desarrollo matemático, los riesgos de interpretación y la bibliografía de la que sale todo eso.

**Por qué.** Sin esto, el entrenamiento es una caja que devuelve números. La teoría es lo que te permite decidir qué mirar y reconocer cuándo un resultado es sospechoso.

**Cómo sabes que salió bien.** Puedes responder, con tus palabras, qué calcula el modelo y por qué esa arquitectura encaja con la tarea `multiclass_classification`.

### Paso 5 — Entrenar y seleccionar con `validation`

**Qué ocurre.** El entrenamiento recorre las épocas midiendo en `validation` después de cada una, y conserva el checkpoint con el mejor valor de `macro_f1`.

**Por qué.** El conjunto de validación existe para tomar decisiones —arquitectura, hiperparámetros, cuándo parar—. Si esas decisiones se tomaran mirando `test`, `test` dejaría de ser una estimación de lo que pasará con datos nuevos y pasaría a ser parte del entrenamiento.

```bash
python labs/17_activations_and_losses/train.py --quick
# o, con control explícito de las dos semillas:
neural-labs train --lab 17_activations_and_losses --config baseline --split-seed 42 --training-seed 43
```

**Cómo sabes que salió bien.** En `runs/17_activations_and_losses/<ejecución>/` aparecen `history.csv` y `best_model.pt`; la métrica de validación mejora respecto de la primera época.

### Paso 6 — Compararte con la línea base

**Qué ocurre.** El repositorio entrena por su cuenta **Regresión ordinal y Random Forest** y guarda su resultado, primero sobre `validation` y —solo al final— sobre `test`.

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
neural-labs benchmark --lab 17_activations_and_losses --quick --split-seed 42 --training-seeds 41 42 43
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
| `variant_comparison.json` | **Propio de esta ruta.** Una fila por variante comparada, con su métrica de validación. |

## ⚠️ Dónde suele perderse la gente

- **`--quick` no es una versión pequeña del resultado, es una prueba de que todo corre.** En esta ruta recorta a 1024 ejemplos de entrenamiento · 256 de validación · 256 de test · 2 épocas. Sirve para comprobar la instalación y la descarga; cualquier conclusión sobre el modelo exige la ejecución completa.
- **Cambiar algo después de ver `test` invalida la comparación.** Si al mirar el resultado final se te ocurre una mejora, la ruta correcta es volver a `validation`, decidir allí, y sellar de nuevo.
- **Las dos semillas no son intercambiables.** `--split-seed` cambia *qué datos* caen en cada partición; `--training-seed` cambia *cómo se inicializa y baraja* el entrenamiento. Para comparar modelos se fija la primera y se varía la segunda.
- **Límite declarado de este dataset.** Muestras reales de vinho verde con análisis fisicoquímico y evaluación sensorial.

### Riesgos al interpretar los resultados

Muestras reales de vinho verde con análisis fisicoquímico y evaluación sensorial.

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

- Goodfellow, Bengio & Courville — *Deep Learning* (MIT Press, 2016), cap. 6 — unidades de activación, no linealidades y funciones de salida con sus pérdidas asociadas.
- Géron — *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* (3.ª ed., O'Reilly 2022), cap. 10–11 — activaciones en la práctica y entrenamiento de redes profundas.
- Bishop — *Pattern Recognition and Machine Learning* (Springer, 2006), cap. 5 — funciones de error y su relación con la interpretación probabilística de la salida.
- Nair & Hinton (2010), *Rectified Linear Units Improve Restricted Boltzmann Machines (ReLU)*, ICML — introducción de la unidad ReLU.
- Glorot, Bordes & Bengio (2011), *Deep Sparse Rectifier Neural Networks*, AISTATS — evidencia de que los rectificadores facilitan el entrenamiento de redes profundas.
- Fuente del dataset: https://archive.ics.uci.edu/dataset/186/wine+quality
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
| [∂ Backpropagation manual](../../labs/16_backpropagation_manual/README.md) | [Las 31 rutas](../../parts/README.md) | [⚙️ Optimizadores y schedulers](../../labs/18_optimizers_and_schedulers/README.md) |

**En este laboratorio:** **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🔴 [Parte 5 — La mecánica fina, ahora en profundidad](../../parts/05-mecanica-fina.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/17_activations_and_losses/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
