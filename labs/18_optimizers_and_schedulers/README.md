# Optimizadores y schedulers

<!-- nav-top -->
> 🧭 **Ruta 19 / 31** · 🔴 [Parte 5 — La mecánica fina, ahora en profundidad](../../parts/05-mecanica-fina.md)
>
> [⬅️ 📐 Activaciones y funciones de pérdida](../../labs/17_activations_and_losses/README.md) · [🏠 Índice de rutas](../../parts/README.md) · [🛡️ Regularización ➡️](../../labs/19_regularization_dropout_batchnorm/README.md)
>
> **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

## 🎯 Qué vas a hacer aquí

Comparar SGD, Momentum, Adam y reducción de tasa de aprendizaje.

Es la **ruta 19 de 31** del recorrido y pertenece a 🔴 la parte 5, *La mecánica fina, ahora en profundidad*. Llegas desde **Activaciones y funciones de pérdida** y lo que hagas aquí lo da por supuesto **Regularización**.

Trabajarás con el dataset **`california_housing`** (scikit-learn / StatLib, licencia: Consultar fuente StatLib), y tendrás que superar la línea base **Media y Ridge**, decidiendo con la métrica `rmse` medida sobre `validation`. Nivel fundamentos, unas **4 horas** de dedicación.

**Lo que conviene traer resuelto de las rutas anteriores:** Python básico, NumPy, álgebra lineal elemental.

**Al terminar deberías ser capaz de:**

- Comparar SGD, Momentum, Adam y reducción de tasa de aprendizaje.
- Preparar y auditar el dataset real california_housing sin fuga de datos.
- Entrenar y evaluar comparación controlada de optimizadores y schedulers.
- Comparar contra la línea base: Media y Ridge.
- Interpretar intervalos de confianza, errores y limitaciones.

## 🧠 La teoría de este laboratorio

Esta sección es la explicación completa del tema. No hace falta abrir otro archivo para entender lo que viene después: aquí está la idea, la matemática que la sostiene y sus límites. (El mismo texto vive en `theory.md`, que es la fuente desde la que se genera esta guía, junto con la bibliografía del final.)

### De qué trata

Este laboratorio estudia **comparación controlada de optimizadores y schedulers** usando `california_housing`, un dataset público real procedente de scikit-learn / StatLib.

Entrenar una red neuronal es resolver un problema de **optimización**: buscar los parámetros θ que minimizan una función de pérdida ℒ(θ) sobre datos reales. Como no podemos evaluar el gradiente exacto sobre todo el conjunto (sería costoso y redundante), estimamos ∇ℒ con **mini-lotes** aleatorios. El optimizador es la regla que traduce ese gradiente ruidoso en un paso de actualización, y el *scheduler* es la política que hace variar la tasa de aprendizaje η a lo largo del entrenamiento. Dos decisiones aparentemente técnicas —qué optimizador y qué programación de η— determinan si el modelo converge rápido, se estanca en una meseta o diverge.

La comparación es "controlada" porque solo cambiamos el optimizador/scheduler mientras mantenemos fijos arquitectura, partición, semillas y presupuesto de cómputo. Así, cualquier diferencia observada en velocidad de convergencia o generalización se atribuye a la regla de actualización y no a factores de confusión. Sobre `california_housing` —una regresión del valor mediano de vivienda a partir de variables socioeconómicas— medimos cuán rápido baja la pérdida de entrenamiento y cuán bien se comporta el modelo en validación.

### La matemática, paso a paso

Actualizaciones de parámetros y programación de learning rate.

El **descenso de gradiente estocástico (SGD)** actualiza cada parámetro moviéndose en dirección opuesta al gradiente del mini-lote: θ ← θ − η · ∇θ ℒ(θ). El signo negativo es la intuición central: −∇ℒ apunta hacia donde la pérdida decrece más rápido, y η controla la longitud del paso. Un η demasiado grande hace que el paso "sobrepase" el mínimo y oscile o diverja; uno demasiado pequeño convierte el entrenamiento en un avance lentísimo. Como el gradiente proviene de un mini-lote, es un estimador *ruidoso* del gradiente verdadero, y ese ruido es a la vez un obstáculo (trayectoria zigzagueante) y una ayuda (permite escapar de mínimos pobres).

El **momentum** acumula una media exponencial de los gradientes recientes para suavizar la trayectoria: vₜ ← β·vₜ₋₁ + (1−β)·∇θ ℒ, luego θ ← θ − η·vₜ. La velocidad v actúa como inercia física: en direcciones donde el gradiente es consistente, los pasos se suman y el avance se acelera; en direcciones donde oscila, las contribuciones se cancelan y el zigzag se amortigua. El factor β ≈ 0.9 fija cuánta "memoria" se conserva.

**Adam** combina momentum con una normalización por la magnitud reciente de cada gradiente. Mantiene dos medias exponenciales, la del gradiente (primer momento mₜ) y la de su cuadrado (segundo momento vₜ), aplica una corrección de sesgo m̂ₜ, v̂ₜ (necesaria porque ambas medias arrancan en cero) y actualiza θ ← θ − η · m̂ₜ / (√v̂ₜ + ε). Dividir por √v̂ₜ da a cada parámetro una tasa de aprendizaje *efectiva* propia: los parámetros con gradientes grandes reciben pasos más cortos y los de gradientes pequeños pasos más largos, lo que hace a Adam robusto a la escala y suele acelerar las primeras épocas. **AdamW** corrige un detalle sutil: desacopla el *weight decay* (λ·θ) de la actualización adaptativa, aplicándolo directamente sobre θ en vez de mezclarlo con el gradiente, lo que restaura la interpretación de regularización L2 que Adam distorsiona.

El **scheduler** hace evolucionar η con el tiempo. La motivación es que conviene un η grande al principio, para avanzar deprisa por regiones lejanas del mínimo, y un η pequeño al final, para asentarse con precisión sin oscilar. Programaciones típicas son el decaimiento por pasos (η se divide por un factor cada cierto número de épocas), el decaimiento coseno η(t) = η_min + ½(η_max − η_min)(1 + cos(π·t/T)), o la reducción en meseta cuando la métrica de validación deja de mejorar. La regla práctica: el optimizador decide *la dirección y forma* del paso; el scheduler decide *su tamaño a lo largo del tiempo*.

### El momentum, leído como una media móvil

La actualización con momentum se escribe v ← β·v + ∇L, θ ← θ − η·v, y su efecto se entiende mejor desarrollando la recurrencia:

v_t = Σ_(k=0..t) β^k · ∇L_(t−k),

es decir, una **media móvil exponencial** de todos los gradientes pasados, con peso decreciente. La suma de los coeficientes tiende a 1/(1 − β), así que con β = 0,9 el paso efectivo es del orden de **diez veces** el de un gradiente aislado: por eso al activar momentum suele haber que reducir la tasa de aprendizaje.

Su utilidad se ve en un valle alargado, que es la forma típica de una superficie de pérdida mal condicionada. En las direcciones donde el gradiente oscila de signo, los términos se cancelan al promediarse; en la dirección donde el gradiente es consistente, se acumulan. El resultado es que el momentum **amortigua el zigzag y acelera el avance por el fondo del valle**, que es exactamente lo que el descenso de gradiente simple hace mal.

### Adam: por qué necesita corrección de sesgo

Adam mantiene dos medias móviles, la del gradiente y la de su cuadrado:

m_t = β₁·m_(t−1) + (1 − β₁)·g_t,   v_t = β₂·v_(t−1) + (1 − β₂)·g_t².

Ambas se inicializan en cero, y ahí está el problema. En el primer paso, m₁ = (1 − β₁)·g₁ = 0,1·g₁ con β₁ = 0,9: la estimación vale **una décima** del valor real. Con β₂ = 0,999 la distorsión de v es aún peor, un factor 0,001. Tomando esperanza se comprueba que 𝔼[m_t] = (1 − β₁ᵗ)·𝔼[g], de donde sale la corrección exacta:

m̂_t = m_t / (1 − β₁ᵗ),   v̂_t = v_t / (1 − β₂ᵗ),   θ ← θ − η · m̂_t / (√v̂_t + ε).

El divisor tiende a 1 conforme t crece, así que la corrección solo actúa en las primeras iteraciones —justo donde el sesgo es grande—. Sin ella, los primeros pasos serían minúsculos y el entrenamiento arrancaría con retraso.

La división por √v̂ es lo que hace de Adam un método **adaptativo por parámetro**: cada peso recibe un paso inversamente proporcional a la magnitud típica de su gradiente. Los parámetros con gradientes grandes avanzan poco a poco; los de gradientes pequeños —capas iniciales, características raras— avanzan más. Esa normalización es lo que le da robustez frente a la elección de η y lo que explica su popularidad. El precio es que la varianza de los primeros pasos, cuando v̂ se ha estimado con pocas muestras, puede ser alta: es la motivación del **calentamiento** que se describe abajo.

### AdamW: el weight decay que Adam rompía

Regularizar con L2 y aplicar weight decay son la misma cosa en SGD, y no lo son en Adam. Merece verse porque el error estuvo presente en implementaciones muy usadas durante años.

En SGD, añadir (λ/2)·‖θ‖² a la pérdida aporta un término λ·θ al gradiente, y la actualización queda θ ← θ − η·(g + λ·θ) = (1 − η·λ)·θ − η·g: un encogimiento proporcional al propio peso. En Adam, ese mismo término λ·θ entra en g **antes** de dividirse por √v̂, así que el encogimiento efectivo de cada parámetro acaba siendo λ·θ/√v̂: los pesos con gradientes históricamente grandes se regularizan **menos** que los de gradientes pequeños. La regularización deja de ser uniforme y pasa a depender de la historia del gradiente, que no es lo que nadie quería.

**AdamW** lo corrige desacoplando: aplica el decaimiento fuera del mecanismo adaptativo,

θ ← θ − η · m̂/(√v̂ + ε) − η·λ·θ,

restituyendo un encogimiento uniforme. La diferencia se nota sobre todo en generalización, y es la razón de que AdamW sea hoy el estándar en visión y en transformers.

### Schedulers: bajar la tasa y por qué calentar

Una tasa fija es un compromiso permanente: alta para avanzar rápido al principio, baja para afinar al final, y no puede ser ambas. De ahí los **schedulers**.

El **recocido coseno** es el más usado y su forma es explícita:

η_t = η_min + ½·(η_max − η_min)·(1 + cos(π·t/T)),

que baja suavemente de η_max a η_min a lo largo de T pasos. Frente a la reducción escalonada, evita los saltos bruscos que desestabilizan el entrenamiento justo después de cada bajada. El **`ReduceLROnPlateau`** sigue otra lógica: en vez de un calendario fijo, reduce la tasa cuando la métrica de validación deja de mejorar, adaptándose al problema a costa de introducir una dependencia de la señal de validación en el propio entrenamiento.

El **calentamiento** hace lo contrario al principio: sube la tasa linealmente desde casi cero durante las primeras iteraciones. Su justificación es la de arriba —con pocos pasos acumulados, las estimaciones de m̂ y v̂ son ruidosas y un paso grande basado en ellas puede desplazar los pesos a una región mala— y se vuelve casi obligatorio con lotes grandes y en transformers.

Una precisión sobre el laboratorio: como aquí la tarea es de **regresión** y se decide con `rmse`, conviene recordar qué implica esa elección. El RMSE, al elevar al cuadrado, penaliza desproporcionadamente los errores grandes y es sensible a valores atípicos; el MAE los trata linealmente. Optimizar error cuadrático y reportar RMSE es coherente, pero significa que el modelo dedicará capacidad a no equivocarse mucho en pocos casos extremos antes que a acertar un poco mejor en la mayoría. Si eso no es lo que el problema pide, la pérdida —y no solo el optimizador— es lo que hay que cambiar.

> **La pregunta que deberías poder responder al terminar:** ¿Cuál mejora más rápido y cuál generaliza mejor?

### Qué se mide y con qué se decide

El laboratorio reporta `mae`, `rmse`, `r2`. De todas ellas, la que **decide** qué modelo se conserva es `rmse`, y se mide siempre sobre `validation`: es la única forma de que `test` siga siendo una estimación honesta de lo que pasará con datos nuevos.

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
jupyter lab labs/18_optimizers_and_schedulers/notebook.ipynb
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
| `--lab` | `18_optimizers_and_schedulers` | obligatorio | Qué laboratorio se ejecuta. Solo acepta los identificadores del catálogo. |
| `--quick` | desactivado | — | Usa una fracción real del dataset y pocas épocas. Sirve para comprobar la instalación, no para concluir nada sobre el modelo. |
| `--split-seed N` | `42` | entero | Semilla que decide **qué ejemplo cae en qué partición**. Se mantiene fija al comparar modelos. |
| `--training-seed N` | `42` | entero | Semilla de la inicialización de pesos y del barajado de lotes. Es la que se varía para medir cuánta diferencia es simple azar. |
| `--config` | `baseline` | `baseline` · `improved` | Cuál de las dos configuraciones del laboratorio se usa. |
| `--device` | `auto` | `auto` · `cpu` · `cuda` · `mps` | Dónde entrenar. `auto` elige GPU si está disponible y cae a CPU si no. |
| `--training-seeds A B C` | `41 42 43` | enteros | Solo en `benchmark`: la lista de semillas de entrenamiento que se repiten. |
| `--output-dir` | `runs` | ruta | Dónde se escribe el directorio de la ejecución. |

### El script del laboratorio

`labs/18_optimizers_and_schedulers/train.py` no es un programa distinto: fija el `--lab` y delega en la misma herramienta, de modo que estas dos líneas hacen exactamente lo mismo.

```bash
python labs/18_optimizers_and_schedulers/train.py --quick
neural-labs train --lab 18_optimizers_and_schedulers --quick
```

### Lo mismo desde Python

Si prefieres trabajar en un cuaderno o llamar al laboratorio desde tu propio código, la misma ejecución se lanza así. La función devuelve un objeto con el directorio de la ejecución, las métricas y el historial ya cargados:

```python
from neural_labs.experiments import run_lab

resultado = run_lab(
    "18_optimizers_and_schedulers",
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

datos = prepare_dataset("18_optimizers_and_schedulers", quick=True, seed=42)
print(datos.summary)       # tamaño de cada partición y metadatos de la fuente
```

## 🪜 Paso a paso

Cada paso dice qué ocurre por dentro, por qué se hace en ese orden y cómo comprobar que salió bien. El orden no es una convención de estilo: es el que ejecuta el código, y alterarlo invalida el resultado.

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
| `variant_comparison.json` | **Propio de esta ruta.** Una fila por variante comparada, con su métrica de validación. |

## ⚠️ Dónde suele perderse la gente

- **`--quick` no es una versión pequeña del resultado, es una prueba de que todo corre.** En esta ruta recorta a 1024 ejemplos de entrenamiento · 256 de validación · 256 de test · 2 épocas. Sirve para comprobar la instalación y la descarga; cualquier conclusión sobre el modelo exige la ejecución completa.
- **Cambiar algo después de ver `test` invalida la comparación.** Si al mirar el resultado final se te ocurre una mejora, la ruta correcta es volver a `validation`, decidir allí, y sellar de nuevo.
- **Las dos semillas no son intercambiables.** `--split-seed` cambia *qué datos* caen en cada partición; `--training-seed` cambia *cómo se inicializa y baraja* el entrenamiento. Para comparar modelos se fija la primera y se varía la segunda.
- **No hay `confusion_matrix.png`, y no es un error.** Es una tarea de regresión: no existen clases que confundir.
- **Límite declarado de este dataset.** Datos reales del censo de California de 1990.

### Riesgos al interpretar los resultados

Datos reales del censo de California de 1990.

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

- Goodfellow, Bengio y Courville — *Deep Learning* (MIT Press, 2016), cap. 8 — tratamiento formal de la optimización para entrenamiento de redes profundas: SGD, momentum y métodos adaptativos.
- Ruder (2016), *An overview of gradient descent optimization algorithms*, arXiv — panorámica comparada de SGD, momentum, Adagrad, RMSProp y Adam con intuición geométrica.
- Robbins y Monro (1951), *A Stochastic Approximation Method*, Annals of Mathematical Statistics — origen teórico del descenso estocástico y las condiciones de convergencia.
- Kingma y Ba (2015), *Adam: A Method for Stochastic Optimization*, ICLR — definición del optimizador Adam y su corrección de sesgo de momentos.
- Loshchilov y Hutter (2019), *Decoupled Weight Decay Regularization (AdamW)*, ICLR — desacoplamiento del weight decay respecto de la actualización adaptativa.
- Fuente del dataset: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html — **California Housing (censo de 1990)** (StatLib (Carnegie Mellon University), StatLib no declara una licencia formal); procedencia, versión y SHA-256 en el registro de fuentes, entrada `california-housing-statlib` — esta clase la usa para comparar SGD, Momentum, Adam y la reducción de la tasa de aprendizaje en una regresión sobre datos censales reales.
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
| [📐 Activaciones y funciones de pérdida](../../labs/17_activations_and_losses/README.md) | [Las 31 rutas](../../parts/README.md) | [🛡️ Regularización](../../labs/19_regularization_dropout_batchnorm/README.md) |

**En este laboratorio:** **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🔴 [Parte 5 — La mecánica fina, ahora en profundidad](../../parts/05-mecanica-fina.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/18_optimizers_and_schedulers/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
