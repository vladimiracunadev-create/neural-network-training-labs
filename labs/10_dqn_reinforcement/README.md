# DQN para inventario con demanda real

<!-- nav-top -->
> 🧭 **Ruta 11 / 31** · 🟣 [Parte 3 — Familias especializadas: generar, decidir, relacionar](../../parts/03-familias-especializadas.md)
>
> [⬅️ 🕸️ GNN sobre red de citas](../../labs/09_gnn_graphs/README.md) · [🏠 Índice de rutas](../../parts/README.md) · [♻️ Transfer learning con mascotas ➡️](../../labs/11_transfer_learning/README.md)
>
> **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

## 🎯 Qué vas a hacer aquí

Aprender una política de reposición usando una secuencia de demanda observada en transacciones reales.

Es la **ruta 11 de 31** del recorrido y pertenece a 🟣 la parte 3, *Familias especializadas: generar, decidir, relacionar*. Llegas desde **GNN sobre red de citas** y lo que hagas aquí lo da por supuesto **Transfer learning con mascotas**.

Trabajarás con el dataset **`online_retail`** (UCI, licencia: CC BY 4.0), y tendrás que superar la línea base **Política de reposición periódica basada en demanda media histórica**, decidiendo con la métrica `mean_return` medida sobre `validation`. Nivel avanzado, unas **8 horas** de dedicación.

**Lo que conviene traer resuelto de las rutas anteriores:** PyTorch intermedio, optimización, lectura de artículos técnicos.

**Al terminar deberías ser capaz de:**

- Aprender una política de reposición usando una secuencia de demanda observada en transacciones reales.
- Preparar y auditar el dataset real online_retail sin fuga de datos.
- Entrenar y evaluar valor de acciones con demanda histórica.
- Comparar contra la línea base: Política de reposición periódica basada en demanda media histórica.
- Interpretar intervalos de confianza, errores y limitaciones.

## 🧠 La teoría de este laboratorio

Esta sección es la explicación completa del tema. No hace falta abrir otro archivo para entender lo que viene después: aquí está la idea, la matemática que la sostiene y sus límites. (El mismo texto vive en `theory.md`, que es la fuente desde la que se genera esta guía, junto con la bibliografía del final.)

### De qué trata

Este laboratorio estudia **valor de acciones con demanda histórica** usando `online_retail`, un dataset público real procedente de UCI.

El problema se plantea como **aprendizaje por refuerzo**: un agente observa el estado del inventario, elige cuánto reponer, y recibe una recompensa que penaliza tanto quedarse sin stock (ventas perdidas) como mantener inventario en exceso (coste de almacenamiento). No hay etiquetas de "acción correcta"; el agente debe descubrir una **política** —una regla que mapea estados a acciones— probando y observando consecuencias a lo largo del tiempo. La dificultad propia del refuerzo es que las decisiones tienen efectos diferidos: reponer poco hoy puede ahorrar coste ahora pero causar un quiebre de stock costoso mañana. El agente debe optimizar la recompensa *acumulada*, no la inmediata.

La pieza clave es aprender el **valor** de cada acción en cada estado: cuánta recompensa futura total cabe esperar si tomo esta acción y luego actúo bien. Con esa función de valor Q(s, a), la política óptima es trivial: en cada estado elegir la acción de mayor Q. **DQN** (Deep Q-Network) aproxima Q con una red neuronal, lo que permite manejar estados continuos (inventario, demanda reciente, posición temporal) sin tabular todos los casos. Lo distintivo de este laboratorio es que la demanda de cada paso no la genera un simulador arbitrario: proviene del **historial real** de transacciones de Online Retail, de modo que la política se enfrenta a la variabilidad genuina de la demanda.

### La matemática, paso a paso

El valor Q óptimo satisface la **ecuación de Bellman de optimalidad**, que expresa el valor de un par (s, a) como la recompensa inmediata más el mejor valor posible del estado siguiente, descontado:

    Q*(s, a) = 𝔼[ r + γ · max_{a′} Q*(s′, a′) | s, a ]

Aquí r es la recompensa recibida al ejecutar a en s, s′ es el estado siguiente, y γ ∈ [0, 1) es el **factor de descuento**, que fija cuánto pesan las recompensas futuras frente a las inmediatas (γ cercano a 1 → agente previsor). DQN entrena una red Q(s, a; θ) para satisfacer esta ecuación minimizando el **error de diferencia temporal (TD)** contra un objetivo (target):

    y = r + γ · max_{a′} Q_target(s′, a′; θ⁻)        ℒ(θ) = 𝔼_{(s,a,r,s′)∼𝒟}[ ( y − Q(s, a; θ) )² ]

Dos ingredientes hacen esto estable. Primero, la **repetición de experiencias** (replay buffer 𝒟): las transiciones (s, a, r, s′) se guardan y se muestrean en minibatches aleatorios, rompiendo la correlación temporal entre muestras consecutivas. Segundo, la **red objetivo** con parámetros θ⁻: una copia rezagada de θ que se actualiza cada cierto tiempo; usarla para calcular y evita que el objetivo persiga a la propia red en cada paso, lo que provocaría oscilaciones. La demanda de cada paso proviene del historial real, no de un generador. Conectando con los cuatro elementos: la **representación de entrada** es el vector de estado s (inventario, demanda reciente, tiempo); la **función del modelo** es la red Q que produce un valor por cada acción discreta de reposición; la **función de pérdida** es el error TD cuadrático de arriba; y la **regla de actualización** es descenso de gradiente, θ ← θ − η ∇_θ ℒ. El notebook muestra las dimensiones de los tensores y conserva la misma implementación que el script de terminal.

Este laboratorio incorpora dos mejoras estándar sobre el DQN original. **Double DQN** corrige la sobreestimación del valor: el operador max en el target tiende a elegir acciones cuyo Q está inflado por ruido, así que se **desacopla** la selección de la evaluación —la red en línea elige la acción y la red objetivo la valora: y = r + γ · Q_target(s′, argmax_{a′} Q(s′, a′; θ); θ⁻). **Dueling DQN** reorganiza la arquitectura separando el valor del estado V(s) de la **ventaja** A(s, a) de cada acción, y las recombina como Q(s, a) = V(s) + ( A(s, a) − (1/|𝒜|) Σ_{a′} A(s, a′) ). La resta de la ventaja media es un truco de identificabilidad que estabiliza el aprendizaje; la intuición es que en muchos estados el valor depende poco de la acción concreta, y estimar V(s) por separado hace el aprendizaje más eficiente.

Por último, el agente equilibra **exploración y explotación** típicamente con una política ε-greedy: con probabilidad ε toma una acción aleatoria (explora) y con probabilidad 1−ε toma argmax_a Q(s, a) (explota), reduciendo ε a lo largo del entrenamiento. Sin exploración suficiente, el agente podría fijar prematuramente una política de reposición subóptima.

### Por qué la Q aprendida sobreestima, con la desigualdad delante

El motivo de Double DQN se puede demostrar en una línea, y conviene verlo porque el sesgo es sistemático, no un accidente de implementación.

El objetivo TD usa max_a Q(s′, a). Como Q es una estimación con error —ruidosa por el muestreo, por la aproximación de la red y por la propia iteración—, se está tomando el máximo de variables aleatorias. Y para cualquier conjunto de variables aleatorias se cumple, por la desigualdad de Jensen aplicada al máximo, que

𝔼[ max_a Q̂(s′, a) ] ≥ max_a 𝔼[ Q̂(s′, a) ].

Es decir: **el máximo de estimaciones ruidosas es, en promedio, mayor que el máximo real**. La razón intuitiva es que el operador máximo selecciona precisamente la acción cuyo error apuntó más hacia arriba. Y como ese valor inflado se usa para construir el objetivo del paso siguiente, el sesgo **se propaga y se acumula** a lo largo del entrenamiento: la red acaba creyendo que todas sus acciones valen más de lo que valen, y su ordenamiento relativo —que es lo único que determina la política— se distorsiona.

La corrección de Double DQN es quirúrgica: se **separa** quién elige la acción de quién la evalúa. En vez de max_a Q_target(s′, a), el objetivo se construye como

y = r + γ · Q_target( s′, argmax_a Q_online(s′, a) ).

La red online elige, la red objetivo puntúa. Como sus errores no están correlacionados, la acción que la primera sobreestimó no tiene por qué estar sobreestimada por la segunda, y el sesgo se reduce drásticamente. No desaparece —ambas redes comparten datos— pero deja de acumularse.

### La tríada mortal, y por qué hacen falta los dos parches

El aprendizaje por refuerzo profundo combina tres ingredientes que, por separado, son inofensivos, y juntos pueden divergir. Se les llama la **tríada mortal**:

1. **Bootstrapping**: el objetivo se construye con la propia estimación, y = r + γ·max Q, en lugar de con retornos reales observados.
2. **Muestreo fuera de política**: se aprende de transiciones generadas por una política distinta de la actual —las del búfer, recogidas hace miles de pasos—.
3. **Aproximación de función**: Q no es una tabla sino una red, así que actualizar un estado modifica la estimación de todos los demás.

Con los tres a la vez, las garantías de convergencia del Q-learning tabular se pierden y el entrenamiento puede diverger. Los dos parches de DQN atacan cada uno un vértice de la tríada. La **repetición de experiencias** rompe la correlación temporal entre muestras consecutivas —el descenso de gradiente supone muestras aproximadamente independientes, y una trayectoria no lo es en absoluto— y permite reutilizar cada transición muchas veces, algo valioso cuando obtenerlas es caro. La **red objetivo** congela el blanco durante unos miles de pasos: sin ella, cada actualización mueve simultáneamente la predicción y el objetivo, y la red persigue una diana que se desplaza con cada paso, un bucle de realimentación que amplifica cualquier error.

Vale la pena notar que esto convierte el aprendizaje por refuerzo en un problema de regresión **no estacionario**: la distribución de datos cambia según mejora la política, y el objetivo cambia según se actualizan los pesos. Ninguna de las dos cosas ocurre en los laboratorios supervisados anteriores, y es la razón de fondo por la que aquí todo es más frágil.

### Cómo se evalúa una política, y por qué una corrida no dice nada

La evaluación en refuerzo tiene reglas propias que este laboratorio hace explícitas.

Primero, **la pérdida no mide desempeño**. Una pérdida TD baja significa que la red predice bien sus propios objetivos, incluso si esos objetivos son malos: es perfectamente posible tener error TD mínimo con una política pésima. Lo que se mide es el **retorno** obtenido al ejecutar la política, y en este dominio, además, sus indicadores operativos —nivel de servicio, roturas de stock, costo de inventario— que es lo que un responsable de reposición miraría de verdad.

Segundo, **la evaluación se hace sin exploración**. Durante el entrenamiento se necesita ε > 0 para descubrir acciones nuevas; al evaluar, cada acción aleatoria mete ruido que no forma parte de la política aprendida. Se evalúa con ε = 0 o con un valor mínimo fijo, y hay que declarar cuál.

Tercero, y es lo que más se incumple en la literatura: la varianza entre semillas en refuerzo profundo es **enorme**, mucho mayor que en aprendizaje supervisado. Dos entrenamientos idénticos salvo por la semilla pueden acabar en políticas de calidad muy distinta, porque las primeras decisiones aleatorias determinan qué experiencias entran al búfer y eso condiciona todo lo que viene después. Reportar una única corrida no es una medida imprecisa: es una anécdota. Por eso la comparación honesta necesita varias semillas y, además, evaluar cada política sobre **varios episodios**, ya que el propio entorno tiene aleatoriedad en la demanda.

Y la línea base debe ser el heurístico que se usaría en la práctica —una política de punto de reorden, del tipo «pedir Q unidades cuando el inventario baje de R»—. Esas reglas están muy bien afinadas en logística, y un DQN que no las supere claramente no justifica su costo de desarrollo, entrenamiento y mantenimiento. Es la comparación que decide si el aprendizaje por refuerzo aportaba algo en este problema.

> **La pregunta que deberías poder responder al terminar:** ¿La política es robusta a cambios en costo y demanda?

### Qué se mide y con qué se decide

El laboratorio reporta `mean_return`, `stockout_rate`, `holding_cost`, `service_level`. De todas ellas, la que **decide** qué modelo se conserva es `mean_return`, y se mide siempre sobre `validation`: es la única forma de que `test` siga siendo una estimación honesta de lo que pasará con datos nuevos.

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
jupyter lab labs/10_dqn_reinforcement/notebook.ipynb
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
| `--lab` | `10_dqn_reinforcement` | obligatorio | Qué laboratorio se ejecuta. Solo acepta los identificadores del catálogo. |
| `--quick` | desactivado | — | Usa una fracción real del dataset y pocas épocas. Sirve para comprobar la instalación, no para concluir nada sobre el modelo. |
| `--split-seed N` | `42` | entero | Semilla que decide **qué ejemplo cae en qué partición**. Se mantiene fija al comparar modelos. |
| `--training-seed N` | `42` | entero | Semilla de la inicialización de pesos y del barajado de lotes. Es la que se varía para medir cuánta diferencia es simple azar. |
| `--config` | `baseline` | `baseline` · `improved` | Cuál de las dos configuraciones del laboratorio se usa. |
| `--device` | `auto` | `auto` · `cpu` · `cuda` · `mps` | Dónde entrenar. `auto` elige GPU si está disponible y cae a CPU si no. |
| `--training-seeds A B C` | `41 42 43` | enteros | Solo en `benchmark`: la lista de semillas de entrenamiento que se repiten. |
| `--output-dir` | `runs` | ruta | Dónde se escribe el directorio de la ejecución. |

### El script del laboratorio

`labs/10_dqn_reinforcement/train.py` no es un programa distinto: fija el `--lab` y delega en la misma herramienta, de modo que estas dos líneas hacen exactamente lo mismo.

```bash
python labs/10_dqn_reinforcement/train.py --quick
neural-labs train --lab 10_dqn_reinforcement --quick
```

### Lo mismo desde Python

Si prefieres trabajar en un cuaderno o llamar al laboratorio desde tu propio código, la misma ejecución se lanza así. La función devuelve un objeto con el directorio de la ejecución, las métricas y el historial ya cargados:

```python
from neural_labs.experiments import run_lab

resultado = run_lab(
    "10_dqn_reinforcement",
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

datos = prepare_dataset("10_dqn_reinforcement", quick=True, seed=42)
print(datos.summary)       # tamaño de cada partición y metadatos de la fuente
```

## 🪜 Paso a paso

Cada paso dice qué ocurre por dentro, por qué se hace en ese orden y cómo comprobar que salió bien. El orden no es una convención de estilo: es el que ejecuta el código, y alterarlo invalida el resultado.

### Paso 1 — Traer el dataset real y partirlo

**Qué ocurre.** Descarga `online_retail` desde su proveedor y construye las tres particiones —`train`, `validation` y `test`— con la semilla de partición que le pases.

**Por qué.** La partición se fija **antes** de entrenar y su semilla (`--split-seed`) es independiente de la del entrenamiento (`--training-seed`). Si ambas fueran la misma, no podrías distinguir si un cambio en el resultado viene de haber repartido los datos de otro modo o de haber inicializado los pesos de otro modo.

```bash
neural-labs dataset --lab 10_dqn_reinforcement --quick --split-seed 42
```

**Cómo sabes que salió bien.** El comando termina sin error y deja el dataset en caché. Si no hay red, **falla**: no se rellena con datos inventados (`data/dataset.yaml` declara `fallback_to_generated_data: false`).

### Paso 2 — Comprobar que las particiones no se tocan

**Qué ocurre.** Recorre los identificadores de las tres particiones y verifica que ninguno aparece en dos de ellas.

**Por qué.** Una sola fila compartida entre `train` y `test` basta para que la métrica final quede inflada. Es el error más caro del oficio porque no produce ningún síntoma visible: el modelo simplemente «parece» mejor de lo que es.

```bash
neural-labs audit --lab 10_dqn_reinforcement --quick --split-seed 42
```

**Cómo sabes que salió bien.** La auditoría reporta cero solapamientos. Si reporta alguno, no sigas: el resto del laboratorio no significaría nada.

### Paso 3 — Mirar los datos antes de modelarlos

**Qué ocurre.** Genera el informe de calidad (valores faltantes, balance de clases, rangos) y el de deriva entre particiones.

**Por qué.** Un desbalance fuerte o una diferencia de distribución entre `train` y `test` cambia qué métrica tiene sentido leer. Descubrirlo después de entrenar obliga a repetirlo todo.

```bash
neural-labs quality --lab 10_dqn_reinforcement --quick --split-seed 42
```

**Cómo sabes que salió bien.** Obtienes `data_quality.json` y `drift_report.json`; ábrelos antes de decidir la configuración.

### Paso 4 — Estudiar la teoría del laboratorio

**Qué ocurre.** Leer [`theory.md`](theory.md): la idea central, el desarrollo matemático, los riesgos de interpretación y la bibliografía de la que sale todo eso.

**Por qué.** Sin esto, el entrenamiento es una caja que devuelve números. La teoría es lo que te permite decidir qué mirar y reconocer cuándo un resultado es sospechoso.

**Cómo sabes que salió bien.** Puedes responder, con tus palabras, qué calcula el modelo y por qué esa arquitectura encaja con la tarea `reinforcement_learning`.

### Paso 5 — Entrenar y seleccionar con `validation`

**Qué ocurre.** El entrenamiento recorre las épocas midiendo en `validation` después de cada una, y conserva el checkpoint con el mejor valor de `mean_return`.

**Por qué.** El conjunto de validación existe para tomar decisiones —arquitectura, hiperparámetros, cuándo parar—. Si esas decisiones se tomaran mirando `test`, `test` dejaría de ser una estimación de lo que pasará con datos nuevos y pasaría a ser parte del entrenamiento.

```bash
python labs/10_dqn_reinforcement/train.py --quick
# o, con control explícito de las dos semillas:
neural-labs train --lab 10_dqn_reinforcement --config baseline --split-seed 42 --training-seed 43
```

**Cómo sabes que salió bien.** En `runs/10_dqn_reinforcement/<ejecución>/` aparecen `history.csv` y `best_model.pt`; la métrica de validación mejora respecto de la primera época.

### Paso 6 — Compararte con la línea base

**Qué ocurre.** El repositorio entrena por su cuenta **Política de reposición periódica basada en demanda media histórica** y guarda su resultado, primero sobre `validation` y —solo al final— sobre `test`.

**Por qué.** Una métrica sola no dice si el modelo aporta algo. Puede que un método mucho más simple llegue igual de lejos, y entonces la complejidad añadida no está justificada. Esta comparación es la que convierte un número en un argumento.

**Cómo sabes que salió bien.** Comparas `metrics.json` con `baseline_metrics.json`. Si tu modelo no supera la línea base, el resultado del laboratorio es exactamente ese, y hay que reportarlo.

### Paso 7 — El sellado: `experiment.lock.json`

**Qué ocurre.** Antes de tocar `test`, el código escribe un archivo que fija el laboratorio, las dos semillas, la configuración, la métrica de selección, el checkpoint elegido y el hash del dataset.

**Por qué.** Es la frontera del experimento. A partir de ahí, cualquier ajuste que hagas mirando `test` queda a la vista: el sello dice qué habías decidido *antes* de ver el resultado final. Sin ese archivo, nadie —incluido tú dentro de un mes— puede distinguir una predicción de una racionalización.

**Cómo sabes que salió bien.** El archivo existe y su contenido coincide con lo que creías haber ejecutado.

### Paso 8 — Evaluar `test` una sola vez y medir la incertidumbre

**Qué ocurre.** Con el checkpoint congelado se evalúa `test`. En esta ruta la tarea es `reinforcement_learning`, así que el resultado se resume en las métricas propias de ese régimen y no en una predicción por ejemplo.

**Por qué.** Un número puntual esconde cuánto podría moverse. Por eso el paso siguiente —repetir con varias semillas— no es opcional aquí: es la única forma de saber cuánta de la diferencia observada es señal.

**Cómo sabes que salió bien.** Tienes `metrics.json` con el resultado final, y sabes que la comparación honesta llega con las repeticiones del paso siguiente.

### Paso 9 — Repetir con varias semillas de entrenamiento

**Qué ocurre.** Se repite el entrenamiento manteniendo **fija** la partición y cambiando solo la semilla de entrenamiento.

**Por qué.** Dos ejecuciones idénticas salvo por la inicialización pueden diferir bastante. Si no mides esa dispersión, corres el riesgo de celebrar una mejora que era una semilla afortunada.

```bash
neural-labs benchmark --lab 10_dqn_reinforcement --quick --split-seed 42 --training-seeds 41 42 43
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
- **Aquí no vas a ver `predictions.csv` ni `confusion_matrix.png`, y no es un error.** La tarea es `reinforcement_learning`, y el código solo genera esos archivos cuando hay una predicción por ejemplo comparable contra una etiqueta.
- **Límite declarado de este dataset.** La dinámica de inventario es un entorno educativo, pero la demanda diaria se construye exclusivamente desde transacciones reales de Online Retail.

### Riesgos al interpretar los resultados

La dinámica de inventario es un entorno educativo, pero la demanda diaria se construye exclusivamente desde transacciones reales de Online Retail.

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

- Sutton & Barto — *Reinforcement Learning: An Introduction* (2.ª ed., MIT Press) — texto canónico: procesos de decisión de Markov, ecuación de Bellman, Q-learning y equilibrio exploración–explotación.
- Mnih et al. (2015), *Human-level control through deep reinforcement learning (DQN)*, Nature — DQN con replay buffer y red objetivo, base del laboratorio.
- van Hasselt, Guez & Silver (2016), *Deep Reinforcement Learning with Double Q-learning*, AAAI — corrección de la sobreestimación desacoplando selección y evaluación.
- Wang et al. (2016), *Dueling Network Architectures for Deep Reinforcement Learning*, ICML — separación de valor de estado y ventaja de acción.
- Fuente del dataset: https://archive.ics.uci.edu/dataset/352/online+retail — **Online Retail** (UCI Machine Learning Repository, CC BY 4.0); procedencia, versión y SHA-256 en el registro de fuentes, entrada `uci-online-retail` — esta clase la usa para construir la señal de demanda diaria a partir de transacciones reales y aprender sobre ella una política de reposición.
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
| [🕸️ GNN sobre red de citas](../../labs/09_gnn_graphs/README.md) | [Las 31 rutas](../../parts/README.md) | [♻️ Transfer learning con mascotas](../../labs/11_transfer_learning/README.md) |

**En este laboratorio:** **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🟣 [Parte 3 — Familias especializadas: generar, decidir, relacionar](../../parts/03-familias-especializadas.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/10_dqn_reinforcement/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
