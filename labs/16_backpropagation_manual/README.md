# Backpropagation manual

<!-- nav-top -->
> 🧭 **Ruta 17 / 31** · 🔴 [Parte 5 — La mecánica fina, ahora en profundidad](../../parts/05-mecanica-fina.md)
>
> [⬅️ 🌐 Aprendizaje federado por participante](../../labs/15_federated_learning/README.md) · [🏠 Índice de rutas](../../parts/README.md) · [📐 Activaciones y funciones de pérdida ➡️](../../labs/17_activations_and_losses/README.md)
>
> **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

## 🎯 Qué vas a hacer aquí

Derivar y programar backpropagation en una MLP pequeña.

Es la **ruta 17 de 31** del recorrido y pertenece a 🔴 la parte 5, *La mecánica fina, ahora en profundidad*. Llegas desde **Aprendizaje federado por participante** y lo que hagas aquí lo da por supuesto **Activaciones y funciones de pérdida**.

Trabajarás con el dataset **`iris`** (UCI, licencia: CC BY 4.0), y tendrás que superar la línea base **Regresión logística multinomial**, decidiendo con la métrica `macro_f1` medida sobre `validation`. Nivel fundamentos, unas **4 horas** de dedicación.

**Lo que conviene traer resuelto de las rutas anteriores:** Python básico, NumPy, álgebra lineal elemental.

**Al terminar deberías ser capaz de:**

- Derivar y programar backpropagation en una MLP pequeña.
- Preparar y auditar el dataset real iris sin fuga de datos.
- Entrenar y evaluar backpropagation manual.
- Comparar contra la línea base: Regresión logística multinomial.
- Interpretar intervalos de confianza, errores y limitaciones.

## 🧠 La teoría de este laboratorio

Esta sección es la explicación completa del tema. No hace falta abrir otro archivo para entender lo que viene después: aquí está la idea, la matemática que la sostiene y sus límites. (El mismo texto vive en `theory.md`, que es la fuente desde la que se genera esta guía, junto con la bibliografía del final.)

### De qué trata

Este laboratorio estudia **backpropagation manual** usando `iris`, un dataset público real procedente de UCI. El objetivo es abrir la caja negra: en lugar de llamar a `loss.backward()` y confiar en el autodiferenciador, se derivan a mano los gradientes de una perceptrón multicapa (MLP) de dos capas y se programan paso a paso. Entender este mecanismo es entender *cómo aprenden* de verdad las redes neuronales.

La retropropagación no es más que la **regla de la cadena** del cálculo aplicada con orden. Una red es una composición de funciones: entrada → capa 1 → activación → capa 2 → softmax → pérdida. Para saber cómo cambiar cada peso y reducir la pérdida, necesitamos la derivada de la pérdida respecto a ese peso. La regla de la cadena nos dice que esa derivada es un producto de derivadas locales encadenadas desde la salida hacia atrás. La idea brillante de backprop es reutilizar cálculos: se calcula una vez el "error" en cada capa y se propaga hacia la capa anterior, evitando recomputar el mismo camino muchas veces.

El flujo tiene dos fases. En el **paso hacia adelante** (forward) se calculan y se guardan las activaciones de cada capa. En el **paso hacia atrás** (backward) se parte del error en la salida y se lo empuja capa por capa hacia la entrada, acumulando en el camino los gradientes de pesos y sesgos. La pregunta crítica del laboratorio —dónde aparecen gradientes que explotan o desaparecen— se vuelve tangible al ver cómo cada capa multiplica el gradiente por factores que pueden encogerlo o amplificarlo.

### La matemática, paso a paso

Consideremos una MLP con una capa oculta. Con entrada x, la propagación hacia adelante es:

  z₁ = W₁ x + b₁,  a₁ = σ(z₁),  z₂ = W₂ a₁ + b₂,  ŷ = softmax(z₂)

y la pérdida de entropía cruzada para la etiqueta one-hot y es ℒ = −Σₖ yₖ · log ŷₖ.

La retropropagación calcula ∂ℒ/∂W₂, ∂ℒ/∂b₂, ∂ℒ/∂W₁ y ∂ℒ/∂b₁ aplicando la regla de la cadena desde la salida. Definimos el **error de la capa de salida**; con softmax + entropía cruzada este error se simplifica de forma notable:

  δ₂ = ∂ℒ/∂z₂ = ŷ − y

De ahí bajan directamente los gradientes de la segunda capa:

  ∂ℒ/∂W₂ = δ₂ · a₁ᵀ,   ∂ℒ/∂b₂ = δ₂

El error se propaga a la capa oculta multiplicando por la matriz de pesos transpuesta y por la derivada de la activación, usando el producto de Hadamard ⊙ (elemento a elemento):

  δ₁ = (W₂ᵀ δ₂) ⊙ σ′(z₁)

  ∂ℒ/∂W₁ = δ₁ · xᵀ,   ∂ℒ/∂b₁ = δ₁

Finalmente, todos los parámetros se actualizan con descenso de gradiente: W ← W − η · ∂ℒ/∂W y b ← b − η · ∂ℒ/∂b, con η la tasa de aprendizaje.

Aquí se ven los **gradientes que se desvanecen o explotan**. El término δ₁ contiene el producto W₂ᵀ δ₂ ⊙ σ′(z₁): si σ es una sigmoide o tanh saturada, σ′(z₁) ≈ 0 y el gradiente se apaga (vanishing); si los pesos son grandes, los factores se acumulan y el gradiente crece sin control (exploding). En una red de L capas, este patrón se repite L veces, así que el gradiente en las capas iniciales es un producto de L factores y su magnitud depende críticamente de que esos factores ronden 1. Comprobar los gradientes analíticos contra una estimación numérica (∂ℒ/∂θ ≈ [ℒ(θ+ε) − ℒ(θ−ε)] / 2ε) es la prueba de que la derivación es correcta. La formulación conecta cuatro elementos: representación de entrada x, función del modelo (MLP), función de pérdida (entropía cruzada) y regla de actualización (SGD con ∇). El notebook muestra las dimensiones de los tensores y conserva la misma implementación que el script de terminal.

### La comprobación numérica, que es el objetivo real del laboratorio

Programar una derivada a mano sin verificarla es programar un error silencioso: un gradiente equivocado no lanza excepción, simplemente entrena peor, y ese síntoma se confunde con «hace falta ajustar la tasa de aprendizaje». Por eso la parte central de esta ruta no es derivar, sino **comprobar**.

El método es contrastar cada derivada analítica contra una **diferencia finita central**:

∂L/∂θᵢ ≈ ( L(θ + ε·eᵢ) − L(θ − ε·eᵢ) ) / (2ε),

donde eᵢ es el vector unitario de la coordenada i. Se usa la versión central y no la de un lado porque su error de truncamiento es O(ε²) en vez de O(ε): con el mismo ε se obtienen varios dígitos más de precisión, a costa de una evaluación adicional de la pérdida.

Elegir ε es un compromiso entre dos errores opuestos. Si es grande, domina el **error de truncamiento** de la aproximación de Taylor; si es minúsculo, la resta L(θ+ε) − L(θ−ε) opera sobre números casi iguales y domina la **cancelación catastrófica** del punto flotante. El óptimo está alrededor de la raíz cúbica del épsilon de máquina, lo que en float64 sitúa el valor práctico en torno a 10⁻⁵ a 10⁻⁶. En float32 la comprobación directamente no es fiable: hay que hacerla en doble precisión.

El criterio de aceptación no se mide con una resta sino con el **error relativo**, para que la escala del gradiente no distorsione el juicio:

error = ‖g_analítico − g_numérico‖ / max( ‖g_analítico‖ + ‖g_numérico‖, δ ).

En doble precisión, por debajo de 10⁻⁷ la implementación es correcta; entre 10⁻⁷ y 10⁻⁴ conviene sospechar; por encima de 10⁻⁴ hay un error real que hay que localizar. El δ del denominador evita dividir por cero cuando ambos gradientes son nulos.

Dos precauciones hacen que la comprobación sirva. La primera: **desactivar toda fuente de aleatoriedad** —dropout, aumentaciones, barajado— porque L debe ser una función determinista de θ; si cada evaluación usa una máscara distinta, la diferencia finita mide ruido. La segunda: no comprobar en puntos donde la función **no es diferenciable**. La ReLU tiene un codo en 0, y si una preactivación cae exactamente ahí, la diferencia finita salta entre dos pendientes distintas y produce un fallo espurio; conviene comprobar con pesos aleatorios en una región donde ninguna preactivación esté cerca de cero.

Como comprobar todas las coordenadas cuesta dos evaluaciones cada una, en redes grandes se verifica un **subconjunto aleatorio** de parámetros por capa. Y la comprobación se hace capa por capa: si el gradiente de la última capa es correcto y el de la anterior no, el error está localizado en el paso intermedio, que es exactamente la información que se necesita para depurar.

### Lo que esta ruta deja preparado

Escribir la retropropagación a mano deja tres ideas que las rutas siguientes dan por sabidas.

La primera es que el paso hacia atrás **reutiliza** cantidades del paso hacia adelante —las activaciones, las máscaras de la ReLU— y por eso hay que conservarlas. Esa es la razón concreta de que el consumo de memoria de un entrenamiento crezca con la profundidad y con el tamaño de lote, y de que existan técnicas que recalculan activaciones para ahorrarla.

La segunda es que la retropropagación **no es más que la regla de la cadena organizada** para no repetir cálculos: se calcula δ una vez por capa y se reutiliza para los pesos y para propagar hacia atrás. Sin esa organización, derivar cada parámetro por separado costaría un número de operaciones proporcional al número de parámetros; con ella, el costo total es del orden del doble del paso hacia adelante, independientemente de cuántos parámetros haya. Es exactamente lo que `autograd` automatiza en la ruta 01.

La tercera es que el producto de jacobianos que aparece al encadenar capas es el origen del desvanecimiento y la explosión del gradiente. Aquí se ve en una red pequeña y sin consecuencias graves; en la ruta 04 es lo que impide aprender dependencias largas, y en la 05 lo que las puertas de la LSTM vienen a resolver.

> **La pregunta que deberías poder responder al terminar:** ¿Dónde aparecen gradientes que explotan o desaparecen?

### Qué se mide y con qué se decide

El laboratorio reporta `accuracy`, `macro_f1`. De todas ellas, la que **decide** qué modelo se conserva es `macro_f1`, y se mide siempre sobre `validation`: es la única forma de que `test` siga siendo una estimación honesta de lo que pasará con datos nuevos.

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
jupyter lab labs/16_backpropagation_manual/notebook.ipynb
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
| `--lab` | `16_backpropagation_manual` | obligatorio | Qué laboratorio se ejecuta. Solo acepta los identificadores del catálogo. |
| `--quick` | desactivado | — | Usa una fracción real del dataset y pocas épocas. Sirve para comprobar la instalación, no para concluir nada sobre el modelo. |
| `--split-seed N` | `42` | entero | Semilla que decide **qué ejemplo cae en qué partición**. Se mantiene fija al comparar modelos. |
| `--training-seed N` | `42` | entero | Semilla de la inicialización de pesos y del barajado de lotes. Es la que se varía para medir cuánta diferencia es simple azar. |
| `--config` | `baseline` | `baseline` · `improved` | Cuál de las dos configuraciones del laboratorio se usa. |
| `--device` | `auto` | `auto` · `cpu` · `cuda` · `mps` | Dónde entrenar. `auto` elige GPU si está disponible y cae a CPU si no. |
| `--training-seeds A B C` | `41 42 43` | enteros | Solo en `benchmark`: la lista de semillas de entrenamiento que se repiten. |
| `--output-dir` | `runs` | ruta | Dónde se escribe el directorio de la ejecución. |

### El script del laboratorio

`labs/16_backpropagation_manual/train.py` no es un programa distinto: fija el `--lab` y delega en la misma herramienta, de modo que estas dos líneas hacen exactamente lo mismo.

```bash
python labs/16_backpropagation_manual/train.py --quick
neural-labs train --lab 16_backpropagation_manual --quick
```

### Lo mismo desde Python

Si prefieres trabajar en un cuaderno o llamar al laboratorio desde tu propio código, la misma ejecución se lanza así. La función devuelve un objeto con el directorio de la ejecución, las métricas y el historial ya cargados:

```python
from neural_labs.experiments import run_lab

resultado = run_lab(
    "16_backpropagation_manual",
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

datos = prepare_dataset("16_backpropagation_manual", quick=True, seed=42)
print(datos.summary)       # tamaño de cada partición y metadatos de la fuente
```

## 🪜 Paso a paso

Cada paso dice qué ocurre por dentro, por qué se hace en ese orden y cómo comprobar que salió bien. El orden no es una convención de estilo: es el que ejecuta el código, y alterarlo invalida el resultado.

### Paso 1 — Traer el dataset real y partirlo

**Qué ocurre.** Descarga `iris` desde su proveedor y construye las tres particiones —`train`, `validation` y `test`— con la semilla de partición que le pases.

**Por qué.** La partición se fija **antes** de entrenar y su semilla (`--split-seed`) es independiente de la del entrenamiento (`--training-seed`). Si ambas fueran la misma, no podrías distinguir si un cambio en el resultado viene de haber repartido los datos de otro modo o de haber inicializado los pesos de otro modo.

```bash
neural-labs dataset --lab 16_backpropagation_manual --quick --split-seed 42
```

**Cómo sabes que salió bien.** El comando termina sin error y deja el dataset en caché. Si no hay red, **falla**: no se rellena con datos inventados (`data/dataset.yaml` declara `fallback_to_generated_data: false`).

### Paso 2 — Comprobar que las particiones no se tocan

**Qué ocurre.** Recorre los identificadores de las tres particiones y verifica que ninguno aparece en dos de ellas.

**Por qué.** Una sola fila compartida entre `train` y `test` basta para que la métrica final quede inflada. Es el error más caro del oficio porque no produce ningún síntoma visible: el modelo simplemente «parece» mejor de lo que es.

```bash
neural-labs audit --lab 16_backpropagation_manual --quick --split-seed 42
```

**Cómo sabes que salió bien.** La auditoría reporta cero solapamientos. Si reporta alguno, no sigas: el resto del laboratorio no significaría nada.

### Paso 3 — Mirar los datos antes de modelarlos

**Qué ocurre.** Genera el informe de calidad (valores faltantes, balance de clases, rangos) y el de deriva entre particiones.

**Por qué.** Un desbalance fuerte o una diferencia de distribución entre `train` y `test` cambia qué métrica tiene sentido leer. Descubrirlo después de entrenar obliga a repetirlo todo.

```bash
neural-labs quality --lab 16_backpropagation_manual --quick --split-seed 42
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
python labs/16_backpropagation_manual/train.py --quick
# o, con control explícito de las dos semillas:
neural-labs train --lab 16_backpropagation_manual --config baseline --split-seed 42 --training-seed 43
```

**Cómo sabes que salió bien.** En `runs/16_backpropagation_manual/<ejecución>/` aparecen `history.csv` y `best_model.npz`; la métrica de validación mejora respecto de la primera época.

### Paso 6 — Compararte con la línea base

**Qué ocurre.** El repositorio entrena por su cuenta **Regresión logística multinomial** y guarda su resultado, primero sobre `validation` y —solo al final— sobre `test`.

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
neural-labs benchmark --lab 16_backpropagation_manual --quick --split-seed 42 --training-seeds 41 42 43
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
| `best_model.npz` | El checkpoint elegido por validación. Esta ruta se implementa en NumPy, así que no hay un `.pt` de PyTorch. |
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
- **Límite declarado de este dataset.** 150 mediciones botánicas reales de tres especies de Iris.

### Riesgos al interpretar los resultados

150 mediciones botánicas reales de tres especies de Iris.

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

- Goodfellow, Bengio & Courville — *Deep Learning* (MIT Press, 2016), cap. 6 — redes feedforward y el algoritmo de retropropagación como aplicación de la regla de la cadena.
- Nielsen — *Neural Networks and Deep Learning* (online, 2015), cap. 2 — derivación paso a paso de backpropagation con las cuatro ecuaciones fundamentales.
- Bishop — *Pattern Recognition and Machine Learning* (Springer, 2006), cap. 5 — redes neuronales, propagación de errores y verificación numérica de gradientes.
- Rumelhart, Hinton & Williams (1986), *Learning representations by back-propagating errors*, Nature — artículo que popularizó la retropropagación para entrenar redes multicapa.
- Baydin et al. (2018), *Automatic Differentiation in Machine Learning: a Survey*, JMLR — panorama de la diferenciación automática que generaliza el backprop manual de este laboratorio.
- Fuente del dataset: https://archive.ics.uci.edu/dataset/53/iris
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
| [🌐 Aprendizaje federado por participante](../../labs/15_federated_learning/README.md) | [Las 31 rutas](../../parts/README.md) | [📐 Activaciones y funciones de pérdida](../../labs/17_activations_and_losses/README.md) |

**En este laboratorio:** **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🔴 [Parte 5 — La mecánica fina, ahora en profundidad](../../parts/05-mecanica-fina.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/16_backpropagation_manual/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
