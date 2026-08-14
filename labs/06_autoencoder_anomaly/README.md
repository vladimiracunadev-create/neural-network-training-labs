# Autoencoder para fraude

<!-- nav-top -->
> 🧭 **Ruta 7 / 31** · 🔵 [Parte 2 — Arquitecturas según la forma del dato](../../parts/02-arquitecturas.md)
>
> [⬅️ 📈 LSTM para series temporales](../../labs/05_lstm_time_series/README.md) · [🏠 Índice de rutas](../../parts/README.md) · [🔭 Transformer para noticias ➡️](../../labs/07_transformer_attention/README.md)
>
> **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

## 🎯 Qué vas a hacer aquí

Detectar transacciones fraudulentas mediante error de reconstrucción.

Es la **ruta 7 de 31** del recorrido y pertenece a 🔵 la parte 2, *Arquitecturas según la forma del dato*. Llegas desde **LSTM para series temporales** y lo que hagas aquí lo da por supuesto **Transformer para noticias**.

Trabajarás con el dataset **`credit_card_fraud`** (Kaggle / ULB, licencia: Uso sujeto a términos de Kaggle y autor), y tendrás que superar la línea base **Isolation Forest**, decidiendo con la métrica `f1` medida sobre `validation`. Nivel intermedio, unas **6 horas** de dedicación.

**Lo que conviene traer resuelto de las rutas anteriores:** PyTorch básico, particiones train/validation/test, métricas de evaluación.

**Al terminar deberías ser capaz de:**

- Detectar transacciones fraudulentas mediante error de reconstrucción.
- Preparar y auditar el dataset real credit_card_fraud sin fuga de datos.
- Entrenar y evaluar reconstrucción para detección de anomalías.
- Comparar contra la línea base: Isolation Forest.
- Interpretar intervalos de confianza, errores y limitaciones.

## 🧠 La teoría de este laboratorio

Esta sección es la explicación completa del tema. No hace falta abrir otro archivo para entender lo que viene después: aquí está la idea, la matemática que la sostiene y sus límites. (El mismo texto vive en `theory.md`, que es la fuente desde la que se genera esta guía, junto con la bibliografía del final.)

### De qué trata

Este laboratorio estudia **reconstrucción para detección de anomalías** usando `credit_card_fraud`, un dataset público real procedente de Kaggle / ULB.

La idea rectora es entrenar un modelo que solo aprenda a describir bien lo *normal*. Un **autoencoder** es una red con forma de reloj de arena: un codificador comprime la entrada x a una representación latente z de baja dimensión (el "cuello de botella"), y un decodificador intenta reconstruir x a partir de z. Si únicamente mostramos transacciones legítimas durante el entrenamiento, la red se especializa en la geometría de esa mayoría y aprende a copiar sus regularidades. Cuando más tarde le presentamos una transacción fraudulenta —que vive fuera de esa variedad aprendida— el decodificador falla y el **error de reconstrucción** se dispara. Ese error es, en la práctica, un detector de anomalías: no clasificamos "fraude vs. no fraude" directamente, sino que medimos cuánto se desvía cada caso del patrón normal.

El cuello de botella es lo que hace que esto funcione: al forzar z a tener menos dimensiones que x, la red no puede memorizar la identidad y debe descubrir los factores latentes que explican las transacciones comunes. Esto es especialmente valioso en fraude, donde los positivos son rarísimos (≈0,17 % del total) y un clasificador supervisado clásico tiende a ignorarlos; el enfoque no supervisado por reconstrucción esquiva ese desbalance porque nunca necesita ejemplos de fraude para aprender.

### La matemática, paso a paso

Formalmente, el codificador es una función f con parámetros θ que produce el código z = f(x; θ) ∈ ℝᵏ, y el decodificador g con parámetros φ produce la reconstrucción x̂ = g(z; φ) = g(f(x; θ); φ). Con k ≪ d (dimensión de x), la composición está obligada a ser una **proyección con pérdida** sobre una variedad de baja dimensión. El objetivo de entrenamiento minimiza el error cuadrático medio de reconstrucción sobre las transacciones normales:

    ℒ(θ, φ) = 𝔼ₓ ‖ x − g(f(x; θ); φ) ‖²  ≈  (1/N) Σᵢ ‖ xᵢ − x̂ᵢ ‖²

El gradiente ∇_{θ,φ} ℒ se propaga por retropropagación a través de decodificador y codificador, y los pesos se actualizan con descenso de gradiente estocástico o Adam: θ ← θ − η ∇_θ ℒ. La conexión con los cuatro elementos del laboratorio es: la **representación de entrada** es el vector x de características de la transacción (28 componentes PCA anonimizadas más `Time` y `Amount` normalizados); la **función del modelo** es la composición g∘f; la **función de pérdida** es el MSE de reconstrucción arriba; y la **regla de actualización** es el paso de gradiente. El notebook muestra las dimensiones de los tensores en cada capa y conserva la misma implementación que el script de terminal.

¿Por qué el MSE minimizado sobre datos normales sirve como puntaje de anomalía? Si asumimos que la reconstrucción está sujeta a un ruido gaussiano isótropo, minimizar ‖x − x̂‖² equivale a maximizar la log-verosimilitud de x bajo el modelo. Tras entrenar, el error r(x) = ‖x − g(f(x))‖² es bajo para lo que la red sabe reconstruir (lo normal) y alto para lo que nunca vio (el fraude). La regla de decisión es un simple umbral: se marca anomalía cuando r(x) > τ. El umbral τ **no se elige a ojo**: se calibra en `validation`, por ejemplo tomando un percentil alto (p. ej. el 99) de la distribución de errores sobre datos legítimos, o el punto que optimiza F1/coste esperado. Variar τ recorre la curva precision–recall completa, y por eso el laboratorio reporta ROC-AUC y PR-AUC en lugar de una sola métrica puntual.

### Qué relación tiene esto con PCA, y cómo se dimensiona el cuello de botella

Hay un resultado que fija exactamente qué está aprendiendo la red. Si el autoencoder fuera **lineal** —sin activaciones— y se entrenara con error cuadrático, su solución óptima abarcaría el mismo subespacio que las k primeras componentes principales de los datos. No es que el autoencoder «se parezca» a PCA: en el caso lineal es equivalente a él. La consecuencia metodológica es directa: **PCA es la referencia que hay que superar**, porque si la variedad donde viven las transacciones normales fuera un subespacio plano, la red no aportaría nada que un método de álgebra lineal cerrada no diera más rápido y de forma determinista. Las activaciones se justifican solo si esa variedad es curva.

Eso también explica por qué el cuello de botella es el hiperparámetro crítico. Si la dimensión latente k igualara la de la entrada, la red podría aprender la **función identidad**: reconstruiría todo a la perfección —incluido el fraude— y el error dejaría de discriminar. El poder de detección viene precisamente de que la red *no puede* reconstruirlo todo y se ve obligada a gastar su capacidad en lo que vio con más frecuencia. Con k demasiado pequeño, en cambio, ni siquiera lo normal se reconstruye bien: el error de fondo sube, la distribución de puntuaciones de ambas clases se solapa y la separación se pierde. La curva de PR-AUC frente a k tiene por eso forma de campana, y encontrar su máximo es trabajo de `validation`.

El error de reconstrucción es además **sensible a la escala** de las variables. Al sumar ‖x − x̂‖² sobre componentes con rangos distintos, una sola variable de magnitud grande puede aportar casi todo el error, y entonces lo que se está detectando no es «anomalía» sino «valor alto en esa columna». En este dataset las 28 componentes vienen ya de una PCA y están en escalas comparables, pero `Time` y `Amount` no: normalizarlas con las estadísticas de `train` es lo que hace que el error signifique algo.

### La contaminación del entrenamiento y el piso de las métricas

El método asume que el conjunto de entrenamiento contiene **solo** transacciones normales, y esa premisa casi nunca se cumple del todo: si un pequeño porcentaje de fraudes se cuela, la red aprende a reconstruirlos también y su error baja, justo el efecto contrario al deseado. La consecuencia es contraintuitiva y conviene anticiparla: **entrenar más tiempo puede empeorar la detección**, porque la capacidad sobrante acaba dedicándose a memorizar los pocos casos raros del entrenamiento. Es una razón adicional para seleccionar el checkpoint por la métrica de `validation` y no por la pérdida de reconstrucción, que seguirá bajando.

Sobre la interpretación de las métricas en un problema tan desbalanceado, conviene tener presente dónde está el piso de cada una. La **exactitud** es inservible: predecir «todo normal» supera el 99 % en este dataset. El **ROC-AUC** parece bueno con facilidad, porque su eje de falsos positivos se normaliza por la clase mayoritaria y multiplicar por diez las falsas alarmas apenas mueve la curva. El **PR-AUC**, en cambio, tiene como piso la **prevalencia** de la clase positiva —una fracción muy pequeña aquí—, así que cualquier valor apreciable representa una mejora real sobre el azar. Por eso, entre las dos cifras que el laboratorio reporta, la que hay que mirar primero es la segunda.

Y para decidir si el sistema sirve en operación, la cifra más honesta suele ser la **precisión a un presupuesto fijo**: de las N transacciones con mayor puntuación —las N que un equipo humano puede revisar en un turno—, cuántas eran fraude. Traduce el modelo a la restricción real, que no es un umbral abstracto sino cuántas revisiones caben en un día.

Una extensión conceptual importante es el **autoencoder variacional (VAE)**: en lugar de un código puntual z, el codificador produce una distribución q(z|x) = 𝒩(μ(x), σ²(x)) y se optimiza el ELBO, que suma el término de reconstrucción y una regularización KL, ℒ = 𝔼_q[‖x − x̂‖²] + β·D_KL(q(z|x) ‖ 𝒩(0, I)). El término KL empuja el espacio latente hacia una gaussiana estándar y da un puntaje de anomalía probabilístico más estable; entender el autoencoder determinista de este laboratorio es el paso previo natural hacia esa formulación.

> **La pregunta que deberías poder responder al terminar:** ¿Qué costo tiene priorizar recall frente a precision?

### Qué se mide y con qué se decide

El laboratorio reporta `precision`, `recall`, `f1`, `roc_auc`, `pr_auc`. De todas ellas, la que **decide** qué modelo se conserva es `f1`, y se mide siempre sobre `validation`: es la única forma de que `test` siga siendo una estimación honesta de lo que pasará con datos nuevos.

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
jupyter lab labs/06_autoencoder_anomaly/notebook.ipynb
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
| `--lab` | `06_autoencoder_anomaly` | obligatorio | Qué laboratorio se ejecuta. Solo acepta los identificadores del catálogo. |
| `--quick` | desactivado | — | Usa una fracción real del dataset y pocas épocas. Sirve para comprobar la instalación, no para concluir nada sobre el modelo. |
| `--split-seed N` | `42` | entero | Semilla que decide **qué ejemplo cae en qué partición**. Se mantiene fija al comparar modelos. |
| `--training-seed N` | `42` | entero | Semilla de la inicialización de pesos y del barajado de lotes. Es la que se varía para medir cuánta diferencia es simple azar. |
| `--config` | `baseline` | `baseline` · `improved` | Cuál de las dos configuraciones del laboratorio se usa. |
| `--device` | `auto` | `auto` · `cpu` · `cuda` · `mps` | Dónde entrenar. `auto` elige GPU si está disponible y cae a CPU si no. |
| `--training-seeds A B C` | `41 42 43` | enteros | Solo en `benchmark`: la lista de semillas de entrenamiento que se repiten. |
| `--output-dir` | `runs` | ruta | Dónde se escribe el directorio de la ejecución. |

### El script del laboratorio

`labs/06_autoencoder_anomaly/train.py` no es un programa distinto: fija el `--lab` y delega en la misma herramienta, de modo que estas dos líneas hacen exactamente lo mismo.

```bash
python labs/06_autoencoder_anomaly/train.py --quick
neural-labs train --lab 06_autoencoder_anomaly --quick
```

### Lo mismo desde Python

Si prefieres trabajar en un cuaderno o llamar al laboratorio desde tu propio código, la misma ejecución se lanza así. La función devuelve un objeto con el directorio de la ejecución, las métricas y el historial ya cargados:

```python
from neural_labs.experiments import run_lab

resultado = run_lab(
    "06_autoencoder_anomaly",
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

datos = prepare_dataset("06_autoencoder_anomaly", quick=True, seed=42)
print(datos.summary)       # tamaño de cada partición y metadatos de la fuente
```

## 🪜 Paso a paso

Cada paso dice qué ocurre por dentro, por qué se hace en ese orden y cómo comprobar que salió bien. El orden no es una convención de estilo: es el que ejecuta el código, y alterarlo invalida el resultado.

### Paso 1 — Traer el dataset real y partirlo

**Qué ocurre.** Descarga `credit_card_fraud` desde su proveedor y construye las tres particiones —`train`, `validation` y `test`— con la semilla de partición que le pases.

**Por qué.** La partición se fija **antes** de entrenar y su semilla (`--split-seed`) es independiente de la del entrenamiento (`--training-seed`). Si ambas fueran la misma, no podrías distinguir si un cambio en el resultado viene de haber repartido los datos de otro modo o de haber inicializado los pesos de otro modo.

```bash
neural-labs dataset --lab 06_autoencoder_anomaly --quick --split-seed 42
```

**Cómo sabes que salió bien.** El comando termina sin error y deja el dataset en caché. Si no hay red, **falla**: no se rellena con datos inventados (`data/dataset.yaml` declara `fallback_to_generated_data: false`).

### Paso 2 — Comprobar que las particiones no se tocan

**Qué ocurre.** Recorre los identificadores de las tres particiones y verifica que ninguno aparece en dos de ellas.

**Por qué.** Una sola fila compartida entre `train` y `test` basta para que la métrica final quede inflada. Es el error más caro del oficio porque no produce ningún síntoma visible: el modelo simplemente «parece» mejor de lo que es.

```bash
neural-labs audit --lab 06_autoencoder_anomaly --quick --split-seed 42
```

**Cómo sabes que salió bien.** La auditoría reporta cero solapamientos. Si reporta alguno, no sigas: el resto del laboratorio no significaría nada.

### Paso 3 — Mirar los datos antes de modelarlos

**Qué ocurre.** Genera el informe de calidad (valores faltantes, balance de clases, rangos) y el de deriva entre particiones.

**Por qué.** Un desbalance fuerte o una diferencia de distribución entre `train` y `test` cambia qué métrica tiene sentido leer. Descubrirlo después de entrenar obliga a repetirlo todo.

```bash
neural-labs quality --lab 06_autoencoder_anomaly --quick --split-seed 42
```

**Cómo sabes que salió bien.** Obtienes `data_quality.json` y `drift_report.json`; ábrelos antes de decidir la configuración.

### Paso 4 — Estudiar la teoría del laboratorio

**Qué ocurre.** Leer [`theory.md`](theory.md): la idea central, el desarrollo matemático, los riesgos de interpretación y la bibliografía de la que sale todo eso.

**Por qué.** Sin esto, el entrenamiento es una caja que devuelve números. La teoría es lo que te permite decidir qué mirar y reconocer cuándo un resultado es sospechoso.

**Cómo sabes que salió bien.** Puedes responder, con tus palabras, qué calcula el modelo y por qué esa arquitectura encaja con la tarea `anomaly_detection`.

### Paso 5 — Entrenar y seleccionar con `validation`

**Qué ocurre.** El entrenamiento recorre las épocas midiendo en `validation` después de cada una, y conserva el checkpoint con el mejor valor de `f1`.

**Por qué.** El conjunto de validación existe para tomar decisiones —arquitectura, hiperparámetros, cuándo parar—. Si esas decisiones se tomaran mirando `test`, `test` dejaría de ser una estimación de lo que pasará con datos nuevos y pasaría a ser parte del entrenamiento.

```bash
python labs/06_autoencoder_anomaly/train.py --quick
# o, con control explícito de las dos semillas:
neural-labs train --lab 06_autoencoder_anomaly --config baseline --split-seed 42 --training-seed 43
```

**Cómo sabes que salió bien.** En `runs/06_autoencoder_anomaly/<ejecución>/` aparecen `history.csv` y `best_model.pt`; la métrica de validación mejora respecto de la primera época.

### Paso 6 — Compararte con la línea base

**Qué ocurre.** El repositorio entrena por su cuenta **Isolation Forest** y guarda su resultado, primero sobre `validation` y —solo al final— sobre `test`.

**Por qué.** Una métrica sola no dice si el modelo aporta algo. Puede que un método mucho más simple llegue igual de lejos, y entonces la complejidad añadida no está justificada. Esta comparación es la que convierte un número en un argumento.

**Cómo sabes que salió bien.** Comparas `metrics.json` con `baseline_metrics.json`. Si tu modelo no supera la línea base, el resultado del laboratorio es exactamente ese, y hay que reportarlo.

### Paso 7 — El sellado: `experiment.lock.json`

**Qué ocurre.** Antes de tocar `test`, el código escribe un archivo que fija el laboratorio, las dos semillas, la configuración, la métrica de selección, el checkpoint elegido y el hash del dataset.

**Por qué.** Es la frontera del experimento. A partir de ahí, cualquier ajuste que hagas mirando `test` queda a la vista: el sello dice qué habías decidido *antes* de ver el resultado final. Sin ese archivo, nadie —incluido tú dentro de un mes— puede distinguir una predicción de una racionalización.

**Cómo sabes que salió bien.** El archivo existe y su contenido coincide con lo que creías haber ejecutado.

### Paso 8 — Evaluar `test` una sola vez y medir la incertidumbre

**Qué ocurre.** Con el checkpoint congelado se evalúa `test`. En esta ruta la tarea es `anomaly_detection`, así que el resultado se resume en las métricas propias de ese régimen y no en una predicción por ejemplo.

**Por qué.** Un número puntual esconde cuánto podría moverse. Por eso el paso siguiente —repetir con varias semillas— no es opcional aquí: es la única forma de saber cuánta de la diferencia observada es señal.

**Cómo sabes que salió bien.** Tienes `metrics.json` con el resultado final, y sabes que la comparación honesta llega con las repeticiones del paso siguiente.

### Paso 9 — Repetir con varias semillas de entrenamiento

**Qué ocurre.** Se repite el entrenamiento manteniendo **fija** la partición y cambiando solo la semilla de entrenamiento.

**Por qué.** Dos ejecuciones idénticas salvo por la inicialización pueden diferir bastante. Si no mides esa dispersión, corres el riesgo de celebrar una mejora que era una semilla afortunada.

```bash
neural-labs benchmark --lab 06_autoencoder_anomaly --quick --split-seed 42 --training-seeds 41 42 43
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
| `model_spec.json` · `inference_contract.json` | Qué entrada espera el modelo y qué devuelve: lo que necesita quien lo despliegue. |
| `model_card.md` · `report.md` | La ficha del modelo y el informe legible de la ejecución. |

## ⚠️ Dónde suele perderse la gente

- **`--quick` no es una versión pequeña del resultado, es una prueba de que todo corre.** En esta ruta recorta a 1024 ejemplos de entrenamiento · 256 de validación · 256 de test · 2 épocas. Sirve para comprobar la instalación y la descarga; cualquier conclusión sobre el modelo exige la ejecución completa.
- **Cambiar algo después de ver `test` invalida la comparación.** Si al mirar el resultado final se te ocurre una mejora, la ruta correcta es volver a `validation`, decidir allí, y sellar de nuevo.
- **Las dos semillas no son intercambiables.** `--split-seed` cambia *qué datos* caen en cada partición; `--training-seed` cambia *cómo se inicializa y baraja* el entrenamiento. Para comparar modelos se fija la primera y se varía la segunda.
- **Aquí no vas a ver `predictions.csv` ni `confusion_matrix.png`, y no es un error.** La tarea es `anomaly_detection`, y el código solo genera esos archivos cuando hay una predicción por ejemplo comparable contra una etiqueta.
- **Límite declarado de este dataset.** 284.807 transacciones reales; el laboratorio evita reequilibrar el conjunto de test.

### Riesgos al interpretar los resultados

284.807 transacciones reales; el laboratorio evita reequilibrar el conjunto de test.

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

- Goodfellow, Bengio & Courville — *Deep Learning* (MIT Press, 2016), cap. 14 — teoría de autoencoders, cuello de botella y autoencoders regularizados/de reducción de dimensión.
- Géron — *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* (3.ª ed., O'Reilly), cap. 17 — autoencoders y GANs en la práctica, detección de anomalías por reconstrucción.
- Hinton & Salakhutdinov (2006), *Reducing the Dimensionality of Data with Neural Networks*, Science — mostró que un autoencoder profundo aprende códigos compactos mejores que PCA.
- Kingma & Welling (2014), *Auto-Encoding Variational Bayes (VAE)*, ICLR — formulación variacional del autoencoder y base del puntaje de anomalía probabilístico.
- Fuente del dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
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
| [📈 LSTM para series temporales](../../labs/05_lstm_time_series/README.md) | [Las 31 rutas](../../parts/README.md) | [🔭 Transformer para noticias](../../labs/07_transformer_attention/README.md) |

**En este laboratorio:** **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🔵 [Parte 2 — Arquitecturas según la forma del dato](../../parts/02-arquitecturas.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/06_autoencoder_anomaly/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
