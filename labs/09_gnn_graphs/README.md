# GNN sobre red de citas

<!-- nav-top -->
> 🧭 **Ruta 10 / 31** · 🟣 [Parte 3 — Familias especializadas: generar, decidir, relacionar](../../parts/03-familias-especializadas.md)
>
> [⬅️ 🎨 GAN generativa](../../labs/08_gan_generation/README.md) · [🏠 Índice de rutas](../../parts/README.md) · [🕹️ DQN para inventario con demanda real ➡️](../../labs/10_dqn_reinforcement/README.md)
>
> **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

## 🎯 Qué vas a hacer aquí

Clasificar publicaciones científicas usando texto y enlaces de citas.

Es la **ruta 10 de 31** del recorrido y pertenece a 🟣 la parte 3, *Familias especializadas: generar, decidir, relacionar*. Llegas desde **GAN generativa** y lo que hagas aquí lo da por supuesto **DQN para inventario con demanda real**.

Trabajarás con el dataset **`cora`** (PyTorch Geometric / Planetoid, licencia: Consultar dataset original), y tendrás que superar la línea base **MLP sin aristas**, decidiendo con la métrica `f1` medida sobre `validation`. Nivel avanzado, unas **8 horas** de dedicación.

**Lo que conviene traer resuelto de las rutas anteriores:** PyTorch intermedio, optimización, lectura de artículos técnicos.

**Al terminar deberías ser capaz de:**

- Clasificar publicaciones científicas usando texto y enlaces de citas.
- Preparar y auditar el dataset real cora sin fuga de datos.
- Entrenar y evaluar propagación de mensajes sobre grafos.
- Comparar contra la línea base: MLP sin aristas.
- Interpretar intervalos de confianza, errores y limitaciones.

## 🧠 La teoría de este laboratorio

Esta sección es la explicación completa del tema. No hace falta abrir otro archivo para entender lo que viene después: aquí está la idea, la matemática que la sostiene y sus límites. (El mismo texto vive en `theory.md`, que es la fuente desde la que se genera esta guía, junto con la bibliografía del final.)

### De qué trata

Este laboratorio estudia **propagación de mensajes sobre grafos** usando `cora`, un dataset público real procedente de PyTorch Geometric / Planetoid.

Cora es una red de citas: cada nodo es un artículo científico descrito por un vector de palabras (bolsa de términos), y cada arista es una cita entre dos artículos. La hipótesis que da sentido al laboratorio es la **homofilia**: los artículos que se citan tienden a tratar temas afines, de modo que la *estructura* del grafo aporta información que el texto por sí solo no captura. Una **red neuronal de grafos (GNN)** explota esa estructura haciendo que cada nodo actualice su representación combinando la suya con la de sus vecinos. Al apilar varias capas, la información se propaga a vecinos de vecinos, y cada nodo termina con un embedding que resume su vecindario local en el grafo.

El mecanismo general se llama **paso de mensajes** (message passing): en cada capa, cada nodo (1) recibe "mensajes" de sus vecinos, (2) los agrega con una función permutación-invariante (suma, media, máximo o atención) y (3) actualiza su estado con esa agregación. La línea base del laboratorio —un MLP que ignora las aristas— sirve justo para cuantificar cuánto aporta la estructura de citaciones frente a usar solo el texto de cada artículo.

### La matemática, paso a paso

La **red convolucional de grafos (GCN)** de Kipf & Welling define la actualización de una capa como:

    H^(l+1) = σ( D̃^{−1/2} Ã D̃^{−1/2} H^(l) W^(l) )

Desglosemos cada símbolo. H^(l) ∈ ℝ^{N×d_l} apila las representaciones de los N nodos en la capa l (H^(0) son las características de entrada). Ã = A + I es la matriz de adyacencia con **auto-lazos** añadidos, para que cada nodo se incluya a sí mismo en la agregación y no pierda su propia información. D̃ es la matriz diagonal de grados de Ã, con D̃ᵢᵢ = Σⱼ Ãᵢⱼ. W^(l) es la matriz de pesos aprendible que transforma las características, y σ es una no linealidad (ReLU). El término D̃^{−1/2} Ã D̃^{−1/2} es la **adyacencia normalizada simétricamente**: propaga las representaciones a los vecinos pero reescalando cada mensaje por 1/√(dᵢ·dⱼ), de modo que los nodos de grado muy alto (muy citados) no dominen la suma ni disparen la escala de las activaciones.

Intuitivamente, cada fila de esa multiplicación calcula, para el nodo i, un **promedio ponderado normalizado** de las características transformadas de i y de sus vecinos: hᵢ^(l+1) = σ( Σ_{j∈𝒩(i)∪{i}} (1/√(d̃ᵢ d̃ⱼ)) · hⱼ^(l) W^(l) ). Apilar L capas equivale a difundir información hasta L saltos de distancia; con L=2, cada artículo "ve" a los artículos que cita y a los que citan a esos. Un exceso de capas provoca **sobre-suavizado** (over-smoothing): las representaciones de todos los nodos convergen y se vuelven indistinguibles, por lo que en la práctica las GCN son poco profundas.

Conectando con los cuatro elementos: la **representación de entrada** es la matriz H^(0) de vectores de palabras por nodo más la estructura del grafo en A; la **función del modelo** es el apilamiento de capas GCN que termina en un softmax sobre las 7 clases temáticas; la **función de pérdida** es la entropía cruzada calculada *solo sobre los nodos de entrenamiento* enmascarados, ℒ = −Σ_{i∈train} Σ_c y_{ic} log ŷ_{ic}; y la **regla de actualización** es descenso de gradiente (Adam), θ ← θ − η ∇_θ ℒ. Es un problema **transductivo**: el grafo completo (con todos los nodos y aristas) participa en cada forward, pero el gradiente solo usa las etiquetas de la máscara de train. El notebook muestra las dimensiones de los tensores (N, d_l) en cada capa y conserva la misma implementación que el script de terminal.

El laboratorio compara variantes del paso de mensajes. **GraphSAGE** (Hamilton et al.) reemplaza la agregación por una que muestrea un subconjunto de vecinos y concatena el estado propio con el agregado, lo que la hace **inductiva** (generaliza a nodos nuevos no vistos). **GAT** (Veličković et al.) sustituye los pesos fijos de normalización por **coeficientes de atención aprendidos** α_{ij} = softmax_j( LeakyReLU(aᵀ[W hᵢ ‖ W hⱼ]) ), de modo que cada nodo decide cuánto pesar a cada vecino en lugar de usar solo el grado. Comparar GCN, GraphSAGE y GAT ilustra cómo cambia el resultado según cómo se agregan los mensajes.

### El esquema general: paso de mensajes

Las tres variantes son casos particulares de un mismo patrón, y verlo así evita aprenderlas como recetas sueltas. Toda capa de una GNN calcula, para cada nodo v:

h_v^(ℓ+1) = ACTUALIZAR( h_v^(ℓ), AGREGAR( { h_u^(ℓ) : u ∈ 𝒩(v) } ) ).

La GCN usa como agregador un promedio con pesos fijos por el grado; GraphSAGE muestrea vecinos y admite media, máximo o LSTM como agregador; GAT aprende los pesos con atención. Cambia el agregador, no el esquema.

Hay una restricción que ese agregador debe cumplir y que determina qué se puede usar: tiene que ser **invariante a permutaciones**. Los vecinos de un nodo son un conjunto, no una lista; si se numeran en otro orden, la representación no puede cambiar. Suma, media y máximo cumplen; concatenar en orden, no. Esa es la razón matemática de que las GNN se construyan con esas operaciones y no con una capa densa sobre los vecinos concatenados.

La elección tampoco es neutra en poder expresivo. La **media** pierde la información del grado: un nodo con dos vecinos idénticos y otro con veinte producen la misma representación. El **máximo** pierde multiplicidades: registra qué tipos de vecino hay, no cuántos. La **suma** conserva ambas cosas, y por eso es la única de las tres que alcanza el poder de distinción del test de isomorfismo de Weisfeiler-Lehman, la cota superior conocida para esta familia de arquitecturas. Si dos grafos no se distinguen con ese test, ninguna GNN de paso de mensajes los distinguirá.

### Qué hace la normalización simétrica, y por qué solo dos capas

La matriz Â = D̃^(−1/2)·Ã·D̃^(−1/2) parece una convención arbitraria y no lo es. Sin normalizar, multiplicar por A suma las representaciones de los vecinos, así que un nodo muy conectado acumula valores mucho mayores que uno periférico y las activaciones se descompensan con la profundidad. Normalizar por el grado a ambos lados hace que los autovalores de Â queden acotados en [−1, 1], y con los auto-lazos el mayor queda en 1: la propagación **no amplifica**, y por eso la red se puede apilar sin que las activaciones exploten.

Esa misma propiedad explica el límite. Aplicar Â repetidamente es un promediado iterado, y un promediado iterado sobre un grafo conexo converge a un punto fijo donde todos los nodos comparten la misma representación, proporcional al autovector dominante. Es el **sobre-suavizado**: con muchas capas, la señal que distingue a un nodo de otro se disuelve y la exactitud cae. De ahí un hecho que sorprende a quien viene de las CNN —donde más profundidad casi siempre ayuda—: las GNN de paso de mensajes suelen rendir mejor con **dos o tres capas**, y ese es el número que este laboratorio explora. El campo receptivo crece muy rápido de todos modos: dos capas ya cubren los vecinos a distancia dos, que en una red de citas puede ser una fracción notable del grafo.

### La fuga de datos en un grafo no es como en una tabla

Este es el punto donde el protocolo del repositorio se vuelve más delicado, y merece atención porque el error es invisible.

Cora se estudia en régimen **transductivo**: el grafo completo —todos los nodos y todas las aristas— está disponible durante el entrenamiento, y lo que se divide en `train`, `validation` y `test` son las **etiquetas**, no los nodos. Solo se calcula la pérdida sobre los nodos etiquetados como entrenamiento. Que un nodo de test participe en el paso de mensajes no es una fuga: es la definición del problema, y así se compara con la literatura.

Lo que sí es una fuga es usar sus **etiquetas** de cualquier forma —directa o indirectamente— antes de la evaluación final. Y hay dos vías sutiles por las que se cuela. La primera es la selección: parar el entrenamiento o elegir arquitectura mirando la exactitud de test es, aquí igual que en cualquier otro laboratorio, contaminar la estimación. La segunda es más específica de los grafos: cualquier característica derivada del grafo que incorpore etiquetas de otros nodos —por ejemplo, «proporción de vecinos de la clase X»— transporta las etiquetas de test al conjunto de entrada por la puerta de atrás.

En el régimen **inductivo**, que es el que GraphSAGE hace posible, las reglas cambian: los nodos de evaluación no existen durante el entrenamiento y deben eliminarse del grafo junto con sus aristas. Es más exigente y más parecido al uso real —clasificar una publicación nueva que acaba de aparecer—, y ambos regímenes no son comparables entre sí. Declarar cuál se está usando forma parte del reporte, porque una exactitud transductiva y una inductiva no miden lo mismo.

Por último, una comparación que este laboratorio pide y que conviene entender: un **MLP que ignore las aristas**, alimentado solo con los atributos de texto de cada publicación. Si la GNN no lo supera con claridad, la estructura de citas no estaba aportando información y toda la maquinaria de paso de mensajes es complejidad sin retorno. Es el equivalente en grafos de la línea base honesta.

> **La pregunta que deberías poder responder al terminar:** ¿Cuánto aporta la estructura de citaciones?

### Qué se mide y con qué se decide

El laboratorio reporta `accuracy`, `macro_f1`. De todas ellas, la que **decide** qué modelo se conserva es `f1`, y se mide siempre sobre `validation`: es la única forma de que `test` siga siendo una estimación honesta de lo que pasará con datos nuevos.

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
jupyter lab labs/09_gnn_graphs/notebook.ipynb
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
| `--lab` | `09_gnn_graphs` | obligatorio | Qué laboratorio se ejecuta. Solo acepta los identificadores del catálogo. |
| `--quick` | desactivado | — | Usa una fracción real del dataset y pocas épocas. Sirve para comprobar la instalación, no para concluir nada sobre el modelo. |
| `--split-seed N` | `42` | entero | Semilla que decide **qué ejemplo cae en qué partición**. Se mantiene fija al comparar modelos. |
| `--training-seed N` | `42` | entero | Semilla de la inicialización de pesos y del barajado de lotes. Es la que se varía para medir cuánta diferencia es simple azar. |
| `--config` | `baseline` | `baseline` · `improved` | Cuál de las dos configuraciones del laboratorio se usa. |
| `--device` | `auto` | `auto` · `cpu` · `cuda` · `mps` | Dónde entrenar. `auto` elige GPU si está disponible y cae a CPU si no. |
| `--training-seeds A B C` | `41 42 43` | enteros | Solo en `benchmark`: la lista de semillas de entrenamiento que se repiten. |
| `--output-dir` | `runs` | ruta | Dónde se escribe el directorio de la ejecución. |

### El script del laboratorio

`labs/09_gnn_graphs/train.py` no es un programa distinto: fija el `--lab` y delega en la misma herramienta, de modo que estas dos líneas hacen exactamente lo mismo.

```bash
python labs/09_gnn_graphs/train.py --quick
neural-labs train --lab 09_gnn_graphs --quick
```

### Lo mismo desde Python

Si prefieres trabajar en un cuaderno o llamar al laboratorio desde tu propio código, la misma ejecución se lanza así. La función devuelve un objeto con el directorio de la ejecución, las métricas y el historial ya cargados:

```python
from neural_labs.experiments import run_lab

resultado = run_lab(
    "09_gnn_graphs",
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

datos = prepare_dataset("09_gnn_graphs", quick=True, seed=42)
print(datos.summary)       # tamaño de cada partición y metadatos de la fuente
```

## 🪜 Paso a paso

Cada paso dice qué ocurre por dentro, por qué se hace en ese orden y cómo comprobar que salió bien. El orden no es una convención de estilo: es el que ejecuta el código, y alterarlo invalida el resultado.

### Paso 1 — Traer el dataset real y partirlo

**Qué ocurre.** Descarga `cora` desde su proveedor y construye las tres particiones —`train`, `validation` y `test`— con la semilla de partición que le pases.

**Por qué.** La partición se fija **antes** de entrenar y su semilla (`--split-seed`) es independiente de la del entrenamiento (`--training-seed`). Si ambas fueran la misma, no podrías distinguir si un cambio en el resultado viene de haber repartido los datos de otro modo o de haber inicializado los pesos de otro modo.

```bash
neural-labs dataset --lab 09_gnn_graphs --quick --split-seed 42
```

**Cómo sabes que salió bien.** El comando termina sin error y deja el dataset en caché. Si no hay red, **falla**: no se rellena con datos inventados (`data/dataset.yaml` declara `fallback_to_generated_data: false`).

### Paso 2 — Comprobar que las particiones no se tocan

**Qué ocurre.** Recorre los identificadores de las tres particiones y verifica que ninguno aparece en dos de ellas.

**Por qué.** Una sola fila compartida entre `train` y `test` basta para que la métrica final quede inflada. Es el error más caro del oficio porque no produce ningún síntoma visible: el modelo simplemente «parece» mejor de lo que es.

```bash
neural-labs audit --lab 09_gnn_graphs --quick --split-seed 42
```

**Cómo sabes que salió bien.** La auditoría reporta cero solapamientos. Si reporta alguno, no sigas: el resto del laboratorio no significaría nada.

### Paso 3 — Mirar los datos antes de modelarlos

**Qué ocurre.** Genera el informe de calidad (valores faltantes, balance de clases, rangos) y el de deriva entre particiones.

**Por qué.** Un desbalance fuerte o una diferencia de distribución entre `train` y `test` cambia qué métrica tiene sentido leer. Descubrirlo después de entrenar obliga a repetirlo todo.

```bash
neural-labs quality --lab 09_gnn_graphs --quick --split-seed 42
```

**Cómo sabes que salió bien.** Obtienes `data_quality.json` y `drift_report.json`; ábrelos antes de decidir la configuración.

### Paso 4 — Estudiar la teoría del laboratorio

**Qué ocurre.** Leer [`theory.md`](theory.md): la idea central, el desarrollo matemático, los riesgos de interpretación y la bibliografía de la que sale todo eso.

**Por qué.** Sin esto, el entrenamiento es una caja que devuelve números. La teoría es lo que te permite decidir qué mirar y reconocer cuándo un resultado es sospechoso.

**Cómo sabes que salió bien.** Puedes responder, con tus palabras, qué calcula el modelo y por qué esa arquitectura encaja con la tarea `node_classification`.

### Paso 5 — Entrenar y seleccionar con `validation`

**Qué ocurre.** El entrenamiento recorre las épocas midiendo en `validation` después de cada una, y conserva el checkpoint con el mejor valor de `f1`.

**Por qué.** El conjunto de validación existe para tomar decisiones —arquitectura, hiperparámetros, cuándo parar—. Si esas decisiones se tomaran mirando `test`, `test` dejaría de ser una estimación de lo que pasará con datos nuevos y pasaría a ser parte del entrenamiento.

```bash
python labs/09_gnn_graphs/train.py --quick
# o, con control explícito de las dos semillas:
neural-labs train --lab 09_gnn_graphs --config baseline --split-seed 42 --training-seed 43
```

**Cómo sabes que salió bien.** En `runs/09_gnn_graphs/<ejecución>/` aparecen `history.csv` y `best_model.pt`; la métrica de validación mejora respecto de la primera época.

### Paso 6 — Compararte con la línea base

**Qué ocurre.** El repositorio entrena por su cuenta **MLP sin aristas** y guarda su resultado, primero sobre `validation` y —solo al final— sobre `test`.

**Por qué.** Una métrica sola no dice si el modelo aporta algo. Puede que un método mucho más simple llegue igual de lejos, y entonces la complejidad añadida no está justificada. Esta comparación es la que convierte un número en un argumento.

**Cómo sabes que salió bien.** Comparas `metrics.json` con `baseline_metrics.json`. Si tu modelo no supera la línea base, el resultado del laboratorio es exactamente ese, y hay que reportarlo.

### Paso 7 — El sellado: `experiment.lock.json`

**Qué ocurre.** Antes de tocar `test`, el código escribe un archivo que fija el laboratorio, las dos semillas, la configuración, la métrica de selección, el checkpoint elegido y el hash del dataset.

**Por qué.** Es la frontera del experimento. A partir de ahí, cualquier ajuste que hagas mirando `test` queda a la vista: el sello dice qué habías decidido *antes* de ver el resultado final. Sin ese archivo, nadie —incluido tú dentro de un mes— puede distinguir una predicción de una racionalización.

**Cómo sabes que salió bien.** El archivo existe y su contenido coincide con lo que creías haber ejecutado.

### Paso 8 — Evaluar `test` una sola vez y medir la incertidumbre

**Qué ocurre.** Con el checkpoint congelado se evalúa `test`. En esta ruta la tarea es `node_classification`, así que el resultado se resume en las métricas propias de ese régimen y no en una predicción por ejemplo.

**Por qué.** Un número puntual esconde cuánto podría moverse. Por eso el paso siguiente —repetir con varias semillas— no es opcional aquí: es la única forma de saber cuánta de la diferencia observada es señal.

**Cómo sabes que salió bien.** Tienes `metrics.json` con el resultado final, y sabes que la comparación honesta llega con las repeticiones del paso siguiente.

### Paso 9 — Repetir con varias semillas de entrenamiento

**Qué ocurre.** Se repite el entrenamiento manteniendo **fija** la partición y cambiando solo la semilla de entrenamiento.

**Por qué.** Dos ejecuciones idénticas salvo por la inicialización pueden diferir bastante. Si no mides esa dispersión, corres el riesgo de celebrar una mejora que era una semilla afortunada.

```bash
neural-labs benchmark --lab 09_gnn_graphs --quick --split-seed 42 --training-seeds 41 42 43
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
| `graph_model_comparison.json` | **Propio de esta ruta.** Puntaje de GCN, GraphSAGE y GAT, y cuál se seleccionó. |

## ⚠️ Dónde suele perderse la gente

- **`--quick` no es una versión pequeña del resultado, es una prueba de que todo corre.** En esta ruta recorta a 1024 ejemplos de entrenamiento · 256 de validación · 256 de test · 2 épocas. Sirve para comprobar la instalación y la descarga; cualquier conclusión sobre el modelo exige la ejecución completa.
- **Cambiar algo después de ver `test` invalida la comparación.** Si al mirar el resultado final se te ocurre una mejora, la ruta correcta es volver a `validation`, decidir allí, y sellar de nuevo.
- **Las dos semillas no son intercambiables.** `--split-seed` cambia *qué datos* caen en cada partición; `--training-seed` cambia *cómo se inicializa y baraja* el entrenamiento. Para comparar modelos se fija la primera y se varía la segunda.
- **Aquí no vas a ver `predictions.csv` ni `confusion_matrix.png`, y no es un error.** La tarea es `node_classification`, y el código solo genera esos archivos cuando hay una predicción por ejemplo comparable contra una etiqueta.
- **Límite declarado de este dataset.** Usa las máscaras públicas fijas de train, validación y test.

### Riesgos al interpretar los resultados

Usa las máscaras públicas fijas de train, validación y test.

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

- Hamilton — *Graph Representation Learning* (Morgan & Claypool, 2020) — texto de referencia sobre embeddings de grafos, paso de mensajes y GNN.
- Kipf & Welling (2017), *Semi-Supervised Classification with Graph Convolutional Networks*, ICLR — la GCN y la normalización simétrica de la adyacencia usada en este laboratorio.
- Hamilton, Ying & Leskovec (2017), *Inductive Representation Learning on Large Graphs (GraphSAGE)*, NeurIPS — agregación por muestreo de vecinos y aprendizaje inductivo.
- Veličković et al. (2018), *Graph Attention Networks*, ICLR — atención sobre vecinos para ponderar mensajes de forma aprendida.
- Fuente del dataset: https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.datasets.Planetoid.html
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
| [🎨 GAN generativa](../../labs/08_gan_generation/README.md) | [Las 31 rutas](../../parts/README.md) | [🕹️ DQN para inventario con demanda real](../../labs/10_dqn_reinforcement/README.md) |

**En este laboratorio:** **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🟣 [Parte 3 — Familias especializadas: generar, decidir, relacionar](../../parts/03-familias-especializadas.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/09_gnn_graphs/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
