# Destilación de conocimiento

<!-- nav-top -->
> 🧭 **Ruta 15 / 31** · 🟠 [Parte 4 — Entrenar mejor, más barato y sin centralizar datos](../../parts/04-entrenamiento-eficiente.md)
>
> [⬅️ 🎛️ Búsqueda de hiperparámetros](../../labs/13_hyperparameter_search/README.md) · [🏠 Índice de rutas](../../parts/README.md) · [🌐 Aprendizaje federado por participante ➡️](../../labs/15_federated_learning/README.md)
>
> **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

## 🎯 Qué vas a hacer aquí

Transferir conocimiento de una CNN profesora a una estudiante compacta.

Es la **ruta 15 de 31** del recorrido y pertenece a 🟠 la parte 4, *Entrenar mejor, más barato y sin centralizar datos*. Llegas desde **Búsqueda de hiperparámetros** y lo que hagas aquí lo da por supuesto **Aprendizaje federado por participante**.

Trabajarás con el dataset **`cifar10`** (Torchvision / University of Toronto, licencia: Consultar términos CIFAR-10), y tendrás que superar la línea base **Estudiante entrenado solo con etiquetas**, decidiendo con la métrica `macro_f1` medida sobre `validation`. Nivel avanzado, unas **8 horas** de dedicación.

**Lo que conviene traer resuelto de las rutas anteriores:** PyTorch intermedio, optimización, lectura de artículos técnicos.

**Al terminar deberías ser capaz de:**

- Transferir conocimiento de una CNN profesora a una estudiante compacta.
- Preparar y auditar el dataset real cifar10 sin fuga de datos.
- Entrenar y evaluar transferencia de conocimiento profesor-estudiante.
- Comparar contra la línea base: Estudiante entrenado solo con etiquetas.
- Interpretar intervalos de confianza, errores y limitaciones.

## 🧠 La teoría de este laboratorio

Esta sección es la explicación completa del tema. No hace falta abrir otro archivo para entender lo que viene después: aquí está la idea, la matemática que la sostiene y sus límites. (El mismo texto vive en `theory.md`, que es la fuente desde la que se genera esta guía, junto con la bibliografía del final.)

### De qué trata

Este laboratorio estudia **transferencia de conocimiento profesor-estudiante** usando `cifar10`, un dataset público real procedente de Torchvision / University of Toronto. La observación de partida es que un modelo grande y preciso (el profesor) no solo predice la clase correcta: en su distribución de salida codifica *cómo de parecidas* considera a las clases entre sí. Por ejemplo, una imagen de gato puede recibir alta probabilidad en "gato", algo en "perro" y casi nada en "camión". Esas probabilidades relativas —las **etiquetas blandas** (soft labels)— son información rica que la etiqueta dura (solo "gato") descarta.

La destilación entrena a un modelo pequeño (el estudiante) para que imite esa distribución blanda del profesor, además de acertar la etiqueta verdadera. El estudiante recibe así una señal de aprendizaje mucho más informativa por ejemplo: en lugar de un único bit correcto/incorrecto, aprende la estructura de similitud que el profesor descubrió con más capacidad y más cómputo. El resultado es un modelo compacto que se acerca a la exactitud del grande con una fracción de los parámetros y la latencia, útil para desplegar en dispositivos con recursos limitados.

La **temperatura** T es la palanca central. Al dividir los logits por T antes del softmax se suavizan las probabilidades: con T alto, las diferencias entre clases se atenúan y emergen las señales pequeñas (esa pizca de "perro" en la imagen de gato) que de otro modo quedarían aplastadas cerca de cero. La pregunta crítica del laboratorio es qué temperatura equilibra mejor la señal dura (la etiqueta verdadera) con la señal blanda (el conocimiento del profesor).

### La matemática, paso a paso

Sean z^t los logits del profesor y z^s los del estudiante para las K clases. El softmax con temperatura T produce distribuciones suavizadas:

  p_k(z; T) = e^{z_k / T} / Σⱼ e^{z_j / T}

Con T = 1 se recupera el softmax normal; con T > 1 la distribución se aplana y revela las probabilidades pequeñas. La pérdida de destilación combina dos términos:

  ℒ = α · CE(y, softmax(z^s)) + (1 − α) · T² · KL( softmax(z^t / T) ‖ softmax(z^s / T) )

El primer término es la **entropía cruzada** contra la etiqueta dura y (aprender lo correcto). El segundo es la **divergencia de Kullback–Leibler** entre la distribución blanda del profesor y la del estudiante, ambas a temperatura T (imitar al profesor). El coeficiente α ∈ [0,1] pondera cuánto pesa cada objetivo.

El factor T² tiene una justificación precisa. Los gradientes del término blando respecto a z^s escalan aproximadamente como 1/T² (porque tanto el softmax suavizado como su derivada introducen un factor 1/T). Multiplicar por T² **reescala** esos gradientes para que su magnitud sea comparable a la del término duro, de modo que al variar T no haya que reajustar α ni el learning rate. La divergencia KL entre profesor p y estudiante q es:

  KL(p ‖ q) = Σₖ p_k · log( p_k / q_k )

y se minimiza cuando q iguala a p. El estudiante actualiza sus parámetros por descenso de gradiente, θ^s ← θ^s − η · ∇_{θ^s} ℒ, mientras el profesor permanece congelado. La formulación conecta cuatro elementos: representación de entrada (la imagen), función del modelo (estudiante), función de pérdida (CE + KL con temperatura) y regla de actualización (SGD con ∇). El notebook muestra las dimensiones de los tensores y conserva la misma implementación que el script de terminal.

### Qué información contienen las etiquetas suaves

La pregunta de fondo del método es por qué un estudiante aprende mejor imitando a un profesor que entrenando con las etiquetas verdaderas, si las etiquetas verdaderas son, por definición, correctas.

La respuesta está en lo que cada señal transporta. Una etiqueta dura sobre CIFAR-10 es un vector one-hot: dice «esto es un gato» y nada más. La salida del profesor para esa misma imagen podría ser p = (gato 0,85 · perro 0,10 · caballo 0,03 · …): dice «esto es un gato, se parece bastante a un perro y algo a un caballo, y no tiene nada que ver con un camión». Esa estructura de similitudes entre clases —lo que Hinton llamó **conocimiento oscuro**— es información que la etiqueta dura no contiene y que el estudiante recibe gratis en cada ejemplo.

Cuantifíquese: una etiqueta dura sobre C clases aporta como mucho log₂ C bits, aquí unos 3,3. Una distribución completa aporta C − 1 números reales, y sobre todo aporta **restricciones sobre la geometría** del espacio de salida. Por eso el estudiante converge con menos datos y menos épocas: cada ejemplo es más informativo.

La temperatura es lo que hace visible esa información. Con T = 1, un profesor bien entrenado produce distribuciones muy picudas —0,999 en la clase correcta— y las probabilidades del resto son tan pequeñas que su contribución al gradiente es despreciable. Elevar T aplana:

p_i^(T) = exp(z_i / T) / Σ_j exp(z_j / T),

y en el límite T → ∞ la distribución tiende a la uniforme, mientras que con T → 0 tiende al one-hot y la destilación degenera en entrenamiento normal. El valor útil está en el medio, típicamente entre 2 y 5, y es un hiperparámetro que se ajusta en `validation`.

El factor T² del término de destilación tiene una razón exacta. Al derivar el softmax con temperatura, el gradiente respecto de los logits escala como 1/T; como la pérdida combina dos términos y solo uno lleva temperatura, sin corrección el término destilado perdería peso relativo al subir T. Multiplicar por T² restablece la escala y permite variar T sin tener que reajustar λ ni la tasa de aprendizaje.

### Qué se gana y qué se paga al comprimir

La comparación que el laboratorio pide es tridimensional, y conviene tener claro qué mide cada eje.

Los **parámetros** miden el tamaño en disco y en memoria. Los **FLOPs** miden el trabajo aritmético. Y la **latencia** mide el tiempo real, que no es proporcional a ninguno de los dos: una red con menos operaciones puede ser más lenta si sus accesos a memoria son irregulares o si sus capas son demasiado pequeñas para saturar el hardware paralelo. Reportar solo la reducción de parámetros y llamarla «aceleración» es un error frecuente; por eso este laboratorio mide la latencia directamente.

Hay un supuesto sin el cual todo el método se cae, y merece enunciarse: la destilación presupone que **existe** una red pequeña capaz de resolver la tarea, y que el problema era encontrarla, no que faltara capacidad. Si la arquitectura del estudiante no tiene capacidad suficiente para representar la función, ninguna cantidad de destilación la creará; el profesor solo puede guiar la búsqueda hacia una buena solución dentro del espacio que el estudiante ya podía representar. De ahí que un estudiante demasiado pequeño no mejore con destilación, y que la elección de su arquitectura sea parte del experimento.

Conviene además situar la destilación entre sus alternativas, porque resuelven el mismo problema por vías distintas y son combinables. La **poda** elimina pesos o canales de la red grande según su importancia; la **cuantización** —que se estudia en la ruta 23— reduce la precisión numérica de 32 a 8 bits, con un factor 4 de reducción casi garantizado y soporte de hardware. Frente a ambas, la destilación tiene una ventaja específica: permite cambiar la **arquitectura** por completo, no solo encoger la existente.

### Cómo evaluar honestamente al estudiante

Tres reglas que este laboratorio hace explícitas y que se incumplen con facilidad.

La primera: la comparación correcta no es estudiante destilado frente a profesor, sino **estudiante destilado frente al mismo estudiante entrenado con etiquetas duras**. Esa es la única que aísla el aporte de la destilación; comparar contra el profesor solo mide cuánta calidad se perdió al comprimir, que es otra pregunta.

La segunda: el profesor debe estar **congelado** durante la destilación y no puede haberse entrenado con ninguna información de `validation` ni de `test`. Un profesor que vio esos datos filtra su conocimiento al estudiante a través de las etiquetas suaves, y la fuga es indirecta pero real.

La tercera: la latencia se mide en el **hardware objetivo**, con el modelo en modo evaluación, tras un calentamiento y promediando varias repeticiones. Una medición única incluye el costo de inicializar los núcleos de cómputo y puede sobreestimar el tiempo en un orden de magnitud.

> **La pregunta que deberías poder responder al terminar:** ¿Qué temperatura equilibra mejor señales duras y blandas?

### Qué se mide y con qué se decide

El laboratorio reporta `accuracy`, `macro_f1`, `parameters`, `latency_ms`. De todas ellas, la que **decide** qué modelo se conserva es `macro_f1`, y se mide siempre sobre `validation`: es la única forma de que `test` siga siendo una estimación honesta de lo que pasará con datos nuevos.

## 🖥️ Los comandos, explicados

Todo el laboratorio se maneja con una sola herramienta de terminal, `neural-labs`, que se instala junto con el paquete (`pip install -e ".[dev,notebooks]"`). Cada subcomando hace **una** cosa del protocolo, y por eso se pueden ejecutar por separado: preparar datos, auditar la partición, entrenar, repetir con varias semillas.

La forma general es siempre la misma:

```bash
neural-labs <subcomando> --lab <identificador> [opciones]
```

| Opción | Valor por defecto | Valores | Qué hace y cuándo cambiarla |
|---|---|---|---|
| `--lab` | `14_knowledge_distillation` | obligatorio | Qué laboratorio se ejecuta. Solo acepta los identificadores del catálogo. |
| `--quick` | desactivado | — | Usa una fracción real del dataset y pocas épocas. Sirve para comprobar la instalación, no para concluir nada sobre el modelo. |
| `--split-seed N` | `42` | entero | Semilla que decide **qué ejemplo cae en qué partición**. Se mantiene fija al comparar modelos. |
| `--training-seed N` | `42` | entero | Semilla de la inicialización de pesos y del barajado de lotes. Es la que se varía para medir cuánta diferencia es simple azar. |
| `--config` | `baseline` | `baseline` · `improved` | Cuál de las dos configuraciones del laboratorio se usa. |
| `--device` | `auto` | `auto` · `cpu` · `cuda` · `mps` | Dónde entrenar. `auto` elige GPU si está disponible y cae a CPU si no. |
| `--training-seeds A B C` | `41 42 43` | enteros | Solo en `benchmark`: la lista de semillas de entrenamiento que se repiten. |
| `--output-dir` | `runs` | ruta | Dónde se escribe el directorio de la ejecución. |

### El script del laboratorio

`labs/14_knowledge_distillation/train.py` no es un programa distinto: fija el `--lab` y delega en la misma herramienta, de modo que estas dos líneas hacen exactamente lo mismo.

```bash
python labs/14_knowledge_distillation/train.py --quick
neural-labs train --lab 14_knowledge_distillation --quick
```

### Lo mismo desde Python

Si prefieres trabajar en un cuaderno o llamar al laboratorio desde tu propio código, la misma ejecución se lanza así. La función devuelve un objeto con el directorio de la ejecución, las métricas y el historial ya cargados:

```python
from neural_labs.experiments import run_lab

resultado = run_lab(
    "14_knowledge_distillation",
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

datos = prepare_dataset("14_knowledge_distillation", quick=True, seed=42)
print(datos.summary)       # tamaño de cada partición y metadatos de la fuente
```

## 🪜 Paso a paso

Cada paso dice qué ocurre por dentro, por qué se hace en ese orden y cómo comprobar que salió bien. El orden no es una convención de estilo: es el que ejecuta el código, y alterarlo invalida el resultado.

### Paso 1 — Traer el dataset real y partirlo

**Qué ocurre.** Descarga `cifar10` desde su proveedor y construye las tres particiones —`train`, `validation` y `test`— con la semilla de partición que le pases.

**Por qué.** La partición se fija **antes** de entrenar y su semilla (`--split-seed`) es independiente de la del entrenamiento (`--training-seed`). Si ambas fueran la misma, no podrías distinguir si un cambio en el resultado viene de haber repartido los datos de otro modo o de haber inicializado los pesos de otro modo.

```bash
neural-labs dataset --lab 14_knowledge_distillation --quick --split-seed 42
```

**Cómo sabes que salió bien.** El comando termina sin error y deja el dataset en caché. Si no hay red, **falla**: no se rellena con datos inventados (`data/dataset.yaml` declara `fallback_to_generated_data: false`).

### Paso 2 — Comprobar que las particiones no se tocan

**Qué ocurre.** Recorre los identificadores de las tres particiones y verifica que ninguno aparece en dos de ellas.

**Por qué.** Una sola fila compartida entre `train` y `test` basta para que la métrica final quede inflada. Es el error más caro del oficio porque no produce ningún síntoma visible: el modelo simplemente «parece» mejor de lo que es.

```bash
neural-labs audit --lab 14_knowledge_distillation --quick --split-seed 42
```

**Cómo sabes que salió bien.** La auditoría reporta cero solapamientos. Si reporta alguno, no sigas: el resto del laboratorio no significaría nada.

### Paso 3 — Mirar los datos antes de modelarlos

**Qué ocurre.** Genera el informe de calidad (valores faltantes, balance de clases, rangos) y el de deriva entre particiones.

**Por qué.** Un desbalance fuerte o una diferencia de distribución entre `train` y `test` cambia qué métrica tiene sentido leer. Descubrirlo después de entrenar obliga a repetirlo todo.

```bash
neural-labs quality --lab 14_knowledge_distillation --quick --split-seed 42
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
python labs/14_knowledge_distillation/train.py --quick
# o, con control explícito de las dos semillas:
neural-labs train --lab 14_knowledge_distillation --config baseline --split-seed 42 --training-seed 43
```

**Cómo sabes que salió bien.** En `runs/14_knowledge_distillation/<ejecución>/` aparecen `history.csv` y `best_model.pt`; la métrica de validación mejora respecto de la primera época.

### Paso 6 — Compararte con la línea base

**Qué ocurre.** El repositorio entrena por su cuenta **Estudiante entrenado solo con etiquetas** y guarda su resultado, primero sobre `validation` y —solo al final— sobre `test`.

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
neural-labs benchmark --lab 14_knowledge_distillation --quick --split-seed 42 --training-seeds 41 42 43
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
- **Límite declarado de este dataset.** Mismo test real para profesor, estudiante base y estudiante destilada.

### Riesgos al interpretar los resultados

Mismo test real para profesor, estudiante base y estudiante destilada.

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

- Goodfellow, Bengio & Courville — *Deep Learning* (MIT Press, 2016) — fundamentos de softmax, entropía cruzada y compresión de modelos.
- Buciluă, Caruana & Niculescu-Mizil (2006), *Model Compression*, KDD — idea seminal de comprimir un conjunto grande en un modelo pequeño que imita sus salidas.
- Hinton, Vinyals & Dean (2015), *Distilling the Knowledge in a Neural Network*, NeurIPS Deep Learning Workshop — formulación de la destilación con temperatura y etiquetas blandas usada en este laboratorio.
- Sanh et al. (2019), *DistilBERT, a distilled version of BERT* — aplicación a gran escala que muestra estudiantes compactos cercanos al profesor.
- Fuente del dataset: https://www.cs.toronto.edu/~kriz/cifar.html
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
| [🎛️ Búsqueda de hiperparámetros](../../labs/13_hyperparameter_search/README.md) | [Las 31 rutas](../../parts/README.md) | [🌐 Aprendizaje federado por participante](../../labs/15_federated_learning/README.md) |

**En este laboratorio:** **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🟠 [Parte 4 — Entrenar mejor, más barato y sin centralizar datos](../../parts/04-entrenamiento-eficiente.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/14_knowledge_distillation/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
