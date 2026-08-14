# Fine-tuning eficiente de transformer

<!-- nav-top -->
> 🧭 **Ruta 26 / 31** · 🔬 [Parte 7 — Especializaciones avanzadas](../../parts/07-especializaciones-avanzadas.md)
>
> [⬅️ 🏁 Proyecto final: churn de telecomunicaciones](../../labs/24_capstone_real_project/README.md) · [🏠 Índice de rutas](../../parts/README.md) · [🧷 Segmentación semántica con U-Net ➡️](../../advanced_labs/26_segmentation_unet/README.md)
>
> **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

## 🎯 Qué vas a hacer aquí

Comparar fine-tuning completo y LoRA sin tocar test durante selección.

Es la **ruta 26 de 31** del recorrido y pertenece a 🔬 la parte 7, *Especializaciones avanzadas*. Llegas desde **Proyecto final: churn de telecomunicaciones** y lo que hagas aquí lo da por supuesto **Segmentación semántica con U-Net**.

Trabajarás con el dataset **`ag_news`** (Hugging Face Datasets, licencia: Consultar ficha AG News), y tendrás que superar la línea base **TF-IDF + regresión logística**, decidiendo con la métrica `accuracy` medida sobre `validation`. Nivel avanzado.

**Qué recibe el modelo como entrada:** texto en inglés.

**Lo que conviene traer resuelto de las rutas anteriores:** PyTorch, NLP, Transformers.

**Al terminar deberías ser capaz de:**

- Comparar fine-tuning completo y LoRA sin tocar test durante selección.
- Interpretar accuracy, macro_f1
- Aplicar sellado de test y reproducibilidad

## 🧠 La teoría de este laboratorio

Esta sección es la explicación completa del tema. No hace falta abrir otro archivo para entender lo que viene después: aquí está la idea, la matemática que la sostiene y sus límites. (El mismo texto vive en `theory.md`, que es la fuente desde la que se genera esta guía, junto con la bibliografía del final.)

### De qué trata

Hasta aquí el recorrido entrenaba modelos desde cero. Este laboratorio invierte el punto de partida: se toma un modelo que **ya sabe leer** —preentrenado sobre miles de millones de palabras— y se pregunta cuánto hay que moverlo para que resuelva una tarea concreta. Clasificar titulares en cuatro categorías no requiere volver a aprender qué es un sustantivo ni cómo se relacionan las palabras de una frase: eso ya está en los pesos. Requiere, como mucho, reorientar esa competencia.

La pregunta que organiza el laboratorio es **cuánto del modelo hay que tocar**. El fine-tuning completo mueve los 67 millones de parámetros de DistilBERT; funciona, pero obliga a almacenar una copia entera del modelo por cada tarea, y en modelos grandes el costo de memoria del optimizador multiplica el problema. La alternativa parte de una observación empírica: la *diferencia* entre los pesos preentrenados y los ajustados tiene rango efectivo bajo, es decir, cabe en muchas menos dimensiones de las que ocupa. Si eso es cierto, se puede congelar el modelo y aprender solo esa corrección de rango pequeño.

Ese es el contraste que se mide aquí: **fine-tuning completo frente a LoRA**, comparando no solo la calidad sino el número de parámetros entrenables y la latencia. Y por debajo de ambos, una línea base de TF-IDF con regresión logística, que representa el texto por frecuencias de palabras sin orden ni contexto. Si la línea base queda cerca, la conclusión honesta es que la tarea no necesitaba un transformer; si queda lejos, la distancia mide exactamente cuánto aporta la comprensión lingüística preentrenada.

### La matemática, paso a paso

Un transformer procesa el texto tras convertirlo en tokens subword (WordPiece en el caso de DistilBERT). Cada token se representa por un vector de embedding al que se le suma una codificación posicional, formando una matriz X ∈ ℝ^(n×d), donde n es la longitud de secuencia y d la dimensión oculta. El corazón del modelo es la **autoatención**: a partir de X se proyectan consultas, claves y valores mediante matrices aprendidas, Q = X·W_Q, K = X·W_K, V = X·W_V, y se calcula

Attention(Q, K, V) = softmax( (Q·Kᵀ) / √d_k ) · V.

La división por √d_k evita que los productos escalares crezcan con la dimensión y saturen el softmax; la matriz softmax(Q·Kᵀ/√d_k) es la que se visualiza como "mapa de atención", pues cada fila indica cuánto pesa cada token del contexto al construir la representación de un token dado. Con varias cabezas (multi-head) el modelo atiende simultáneamente a distintos subespacios de relación.

El preentrenamiento (BERT) optimiza un objetivo de **modelado de lenguaje enmascarado**: se ocultan tokens aleatorios y la red minimiza la entropía cruzada al predecirlos, ℒ = −Σᵢ log p(xᵢ | x_contexto). Así el modelo aprende representaciones lingüísticas generales antes de ver la tarea final. DistilBERT es una versión **destilada**: un modelo "estudiante" más pequeño se entrena para imitar al "profesor" BERT, combinando la pérdida supervisada con un término de destilación sobre las distribuciones suaves del profesor (softmax con temperatura), conservando ~97% del rendimiento con ~40% menos parámetros.

En el **fine-tuning completo** se añade una capa de clasificación sobre el embedding del token especial [CLS] y se actualizan *todos* los pesos θ del modelo mediante descenso de gradiente sobre la entropía cruzada de la tarea, θ ← θ − α·∇_θ ℒ. Es potente pero costoso: hay que almacenar y actualizar decenas de millones de parámetros por tarea. La **adaptación eficiente LoRA** parte de la observación de que la actualización necesaria ΔW tiene rango efectivo bajo. En lugar de modificar la matriz preentrenada W₀ ∈ ℝ^(d×k), LoRA la congela y aprende una corrección factorizada de rango r pequeño:

W = W₀ + ΔW = W₀ + (α/r)·B·A,   con B ∈ ℝ^(d×r), A ∈ ℝ^(r×k), r ≪ min(d, k).

Solo se entrenan A y B (más el factor de escala α/r), reduciendo los parámetros entrenables en órdenes de magnitud sin añadir latencia en inferencia (las matrices pueden fusionarse en W). Esta es la esencia del *parameter-efficient transfer learning*, emparentada con los **adapters** de Houlsby et al., que insertan pequeños módulos entrenables entre capas congeladas. La línea base TF-IDF + regresión logística sirve de contraste: representa el texto por frecuencias de término ponderadas, sin capturar orden ni contexto, y ayuda a medir cuánto aporta realmente la atención preentrenada.

### La cuenta de parámetros, que es el argumento entero

Conviene hacer la aritmética, porque es donde se ve la magnitud del ahorro. Adaptar una matriz W₀ ∈ ℝ^(d×k) por la vía completa exige entrenar sus d·k pesos. Por la vía LoRA se entrenan las dos matrices factorizadas: r·d entradas en B más r·k en A, es decir

parámetros_LoRA = r·(d + k),   frente a   parámetros_completo = d·k.

La razón entre ambas es r·(d + k) / (d·k), que para una matriz cuadrada d = k se simplifica a **2r/d**. Con la dimensión oculta de DistilBERT, d = 768, y un rango r = 8, esa fracción es 16/768 ≈ 2,1 %: se entrena una cincuentava parte de lo que se entrenaría adaptando esa matriz por completo. El ahorro real es aún mayor, porque LoRA suele aplicarse solo a las proyecciones de atención (W_Q y W_V) y deja intactas las capas densas intermedias.

El factor de escala α/r no es decorativo: mantiene la magnitud de la actualización aproximadamente constante cuando se cambia r, de modo que el rango se pueda variar sin tener que reajustar la tasa de aprendizaje. Y A se inicializa con ruido pequeño mientras B se inicializa en cero, así que al empezar ΔW = B·A = 0 y el modelo arranca siendo exactamente el preentrenado: el ajuste parte de donde quedó el preentrenamiento, no de una perturbación aleatoria.

En inferencia no hay penalización porque W₀ + (α/r)·B·A se puede calcular una vez y guardarse como una sola matriz. Esa es la diferencia práctica con los adapters, que añaden módulos en serie y por tanto suman latencia en cada paso.

### El bloque completo, más allá de la atención

La atención no actúa sola. Cada bloque del transformer la envuelve en dos operaciones que son las que permiten apilar capas sin que el entrenamiento se degrade: una **conexión residual** y una **normalización de capa**. El bloque calcula

h = LayerNorm( x + MHA(x) ),   y = LayerNorm( h + FFN(h) ),

donde MHA es la atención multi-cabeza y FFN una red densa de dos capas aplicada posición a posición, FFN(h) = W₂ · GELU(W₁·h + b₁) + b₂, típicamente con dimensión interna 4d. La suma residual crea un camino directo por el que el gradiente llega a las capas iniciales sin atenuarse; la normalización de capa estandariza cada vector de activación sobre su propia dimensión —a diferencia de la normalización por lotes, que promedia sobre el lote— y por eso funciona con secuencias de longitud variable y lotes pequeños.

Conviene notar dónde vive el costo: la atención tiene complejidad **O(n²·d)** en la longitud de secuencia n, porque construye la matriz Q·Kᵀ de n×n. Duplicar la longitud máxima de los titulares cuadruplica ese término. La FFN, en cambio, es O(n·d²). Para secuencias cortas como las de este laboratorio domina la segunda; en documentos largos, la primera.

La cabeza de clasificación se conecta al vector del token especial [CLS], que por construcción atiende a toda la secuencia y actúa como resumen agregado. Sobre él se aplica una capa lineal a las C clases y un softmax, y se optimiza la entropía cruzada ℒ = −Σ_c y_c · log ŷ_c. Una consecuencia práctica: como esa capa es la única inicializada al azar mientras el resto viene preentrenado, se suele usar una tasa de aprendizaje pequeña (del orden de 2·10⁻⁵ a 5·10⁻⁵) para todo el modelo, evitando que los primeros gradientes —grandes, porque la cabeza no sabe nada— destruyan las representaciones aprendidas. Ese fenómeno se conoce como **olvido catastrófico**, y es la razón de que un fine-tuning con la tasa de un entrenamiento desde cero suela empeorar el resultado en vez de mejorarlo.

### Qué hereda DistilBERT de la destilación

El modelo de partida no es BERT sino su versión destilada, y eso también es matemática explícita. La destilación de Hinton, Vinyals y Dean entrena al estudiante para reproducir las **distribuciones suaves** del profesor, no solo la etiqueta correcta. Con logits z, la distribución suavizada por temperatura T es

p_i^(T) = exp(z_i / T) / Σ_j exp(z_j / T),

y la pérdida combina el término supervisado habitual con la divergencia respecto del profesor:

ℒ = (1 − λ)·ℒ_CE(y, p^(1)_estudiante) + λ·T²·KL( p^(T)_profesor ‖ p^(T)_estudiante ).

Una temperatura T > 1 aplana las distribuciones y revela información que la etiqueta dura esconde —qué clases confunde el profesor, y cuánto—; ese conocimiento sobre las *proporciones de error* es lo que permite al estudiante aprender más rápido que entrenando solo con etiquetas. El factor T² compensa que los gradientes del término suavizado escalan como 1/T². DistilBERT aplica esta receta sobre las 12 capas de BERT para quedarse con 6, conservando alrededor del 97 % del rendimiento con un 40 % menos de parámetros, y es la razón de que este laboratorio pueda entrenarse sin GPU dedicada.

### Qué conviene graficar

Distribución de longitud, matriz de confusión, atención y comparación LoRA/full. Los mapas de atención revelan qué tokens influyen en la clasificación; la comparación LoRA vs. fine-tuning completo contrasta accuracy y macro_f1 frente al número de parámetros entrenables y la latencia, para juzgar el coste-beneficio de cada estrategia.

### Qué se mide y con qué se decide

El laboratorio reporta `accuracy`, `macro_f1`, `latency_ms`, `trainable_parameters`. De todas ellas, la que **decide** qué modelo se conserva es `accuracy`, y se mide siempre sobre `validation`: es la única forma de que `test` siga siendo una estimación honesta de lo que pasará con datos nuevos.

## 📓 Los tres cuadernos

El laboratorio se puede recorrer en Jupyter, y trae tres cuadernos con papeles distintos. Los tres siguen el mismo camino —descargar el dataset real, auditar la partición, entrenar, sellar el experimento y evaluar `test` una vez—; lo que cambia es qué te toca escribir a ti:

| Cuaderno | Qué trae | Cuándo usarlo |
|---|---|---|
| [📓 `notebook.ipynb`](notebook.ipynb) | El **recorrido de referencia**: 22 celdas (10 de código) con **todo el código escrito y ejecutable**, intercalado con las explicaciones. No trae ejercicios. | Para leer y ejecutar de principio a fin. |
| [✏️ `notebook_student.ipynb`](notebook_student.ipynb) | El mismo recorrido más **8 ejercicios evaluables** (37 celdas en total). Las celdas de ejercicio están marcadas con `# YOUR CODE HERE` y debajo de cada una hay una comprobación. | Para practicar. |
| [✅ `notebook_solution.ipynb`](notebook_solution.ipynb) | Los mismos ejercicios **resueltos**, marcados con `# SOLUCIÓN DE REFERENCIA`. Cada solución se ejecuta en la integración continua, así que se sabe que pasa. | Para contrastar después de intentarlo. |

### Qué se practica en los ejercicios

Cinco de ellos no son de arquitectura sino del **contrato experimental**, que es lo que distingue a estos laboratorios de un tutorial: auditar la partición, decidir con `validation`, compararse con la línea base, sellar antes de abrir `test` y dejar el plan por escrito. Se resuelven con Python estándar —**sin descargar el dataset ni entrenar**—, así que se corrigen en segundos y sin GPU, y cada uno está parametrizado con los valores de este laboratorio: su métrica de selección, su línea base y su experimento propio.

### Cómo abrirlos

Los cuadernos necesitan el extra `notebooks`, que instala Jupyter junto con el paquete:

```bash
pip install -e ".[dev,notebooks]"
jupyter lab advanced_labs/25_transformer_finetuning/notebook.ipynb
```

También se abren desde VS Code —con la extensión de Jupyter— haciendo doble clic en el archivo, o desde la interfaz clásica con `jupyter notebook`. El primer arranque descarga el dataset real desde su proveedor, así que la primera ejecución tarda más y **requiere conexión**.

Si prefieres ejecutar sin abrir un cuaderno, `train.py` hace exactamente lo mismo desde la terminal, y la sección de comandos de arriba explica cada opción.

## 🖥️ Los comandos, explicados

Todo el laboratorio se maneja con una sola herramienta de terminal, `neural-labs`, que se instala junto con el paquete (`pip install -e ".[dev,notebooks]"`). Cada subcomando hace **una** cosa del protocolo, y por eso se pueden ejecutar por separado: preparar datos, auditar la partición, entrenar, repetir con varias semillas.

La forma general es siempre la misma:

```bash
neural-labs <subcomando> --track <identificador> [opciones]
```

| Opción | Valor por defecto | Valores | Qué hace y cuándo cambiarla |
|---|---|---|---|
| `--track` | `25_transformer_finetuning` | obligatorio | Qué especialización se entrena. Solo acepta los seis identificadores existentes. |
| `--quick` | desactivado | — | Reduce datos y épocas para comprobar que la ruta corre de extremo a extremo. |
| `--split-seed N` | `42` | entero | Semilla que decide **qué ejemplo cae en qué partición**. Se mantiene fija al comparar. |
| `--training-seed N` | `42` | entero | Semilla de la inicialización de pesos y del barajado. Es la que se varía para medir dispersión. |
| `--device` | `auto` | `auto` · `cpu` · `cuda` · `mps` | Dónde entrenar. `auto` elige GPU si la hay. |
| `--output-dir` | `runs-advanced` | ruta | Dónde se escribe el directorio de la ejecución. |
| `--lora` / `--no-lora` | `--no-lora` | — | Con LoRA se entrenan unas pocas matrices de bajo rango en vez de todos los pesos: el objetivo del laboratorio es comparar ambas. |

### Lo mismo desde Python

```python
from neural_labs.advanced.training import train_advanced

resultado = train_advanced(
    "25_transformer_finetuning",
    quick=True,
    split_seed=42,
    training_seed=43,
)

print(resultado["run_dir"])
print(resultado["metrics"])
```

## 🪜 Paso a paso

Cada paso dice qué ocurre por dentro, por qué se hace en ese orden y cómo comprobar que salió bien. El orden no es una convención de estilo: es el que ejecuta el código, y alterarlo invalida el resultado.

### Paso 1 — Estudiar la teoría antes de ejecutar nada

**Qué ocurre.** Leer [`theory.md`](theory.md), que desarrolla Tokenización subword, atención preentrenada, fine-tuning completo y adaptación eficiente LoRA. y cita las obras y papers de los que procede.

**Por qué.** Estas rutas usan arquitecturas donde un error de comprensión no se manifiesta como un fallo, sino como un número plausible pero equivocado.

**Cómo sabes que salió bien.** Puedes explicar qué mide `accuracy` y por qué es la métrica de selección aquí.

### Paso 2 — Ejecutar la versión rápida

**Qué ocurre.** Descarga el dataset y los pesos preentrenados desde su proveedor, entrena una versión reducida y escribe la ejecución en `runs-advanced/`.

**Por qué.** Antes de gastar horas de cómputo conviene comprobar que la descarga, el entorno y la ruta completa funcionan de extremo a extremo.

```bash
neural-labs train-advanced --track 25_transformer_finetuning --quick --lora
```

**Cómo sabes que salió bien.** Termina sin error y deja `metrics.json`, `history.json` y `best_model.pt` en el directorio de la ejecución.

### Paso 3 — Entrenar en serio y seleccionar con `validation`

**Qué ocurre.** Se entrena el modelo completo conservando el checkpoint con el mejor valor de `accuracy` en validación, y se sella el experimento antes de evaluar `test`.

**Por qué.** Igual que en las rutas centrales: `validation` decide, `test` solo confirma, y el sello deja por escrito qué se había decidido antes de mirar.

```bash
neural-labs train-advanced --track 25_transformer_finetuning --split-seed 42 --training-seed 43 --lora
```

**Cómo sabes que salió bien.** Existe `experiment.lock.json` y `metrics.json` incluye tanto el valor de validación como el de test.

### Paso 4 — Repetir con otra semilla de entrenamiento

**Qué ocurre.** Se repite el entrenamiento con la misma partición y distinta semilla de entrenamiento.

**Por qué.** Estas arquitecturas —adversariales, contrastivas, de difusión— son especialmente sensibles a la inicialización: una sola ejecución no permite distinguir una mejora de una casualidad.

```bash
neural-labs train-advanced --track 25_transformer_finetuning --split-seed 42 --training-seed 44 --lora
```

**Cómo sabes que salió bien.** Puedes reportar el rango entre ejecuciones, no un único número.

### Paso 5 — Documentar los límites

**Qué ocurre.** Registrar el resultado junto con la limitación declarada de la ruta y responder [`assessment.md`](assessment.md).

**Por qué.** En generación y aprendizaje autosupervisado las métricas son aproximaciones: sin declarar qué NO demuestran, invitan a conclusiones que los números no sostienen.

**Cómo sabes que salió bien.** Tu reporte dice qué mejoró, cuánto costó y en qué condiciones no esperarías el mismo resultado.

## 🔍 Cómo leer lo que produce la ejecución

Cada ejecución escribe su propio directorio con nombre único, de modo que dos corridas nunca se pisan. Esto es lo que encontrarás dentro:

| Archivo | Qué contiene y qué mirar |
|---|---|
| `config.json` | Track, semillas, dispositivo y opciones con las que se lanzó. |
| `dataset_manifest.json` | Fuente, licencia y número de ejemplos por partición. |
| `best_model.pt` | El checkpoint seleccionado por validación. |
| `experiment.lock.json` | El sello: qué se decidió antes de abrir `test`. |
| `history.json` | La métrica de validación época a época. |
| `metrics.json` | Resultado de validación y de test, ya con el modelo congelado. |

## ⚠️ Dónde suele perderse la gente

- **Cambiar algo después de ver `test` invalida la comparación.** Si al mirar el resultado final se te ocurre una mejora, la ruta correcta es volver a `validation`, decidir allí, y sellar de nuevo.
- **Las dos semillas no son intercambiables.** `--split-seed` cambia *qué datos* caen en cada partición; `--training-seed` cambia *cómo se inicializa y baraja* el entrenamiento. Para comparar modelos se fija la primera y se varía la segunda.
- **Límite declarado de este dataset.** El corpus contiene titulares históricos y sesgos editoriales; no representa todo el lenguaje contemporáneo.

### Riesgos al interpretar los resultados

El corpus contiene titulares históricos y sesgos editoriales; no representa todo el lenguaje contemporáneo. Además, un mapa de atención alto no implica causalidad ni "explicación" fiable de la decisión: la atención es una entre varias señales internas del modelo y debe interpretarse con cautela.

## ✅ Antes de darlo por terminado

Y cuando tienes estos entregables:

- [ ] notebook ejecutado
- [ ] reporte experimental
- [ ] model card

El plan experimental con la tabla que hay que completar está en `experiments.md`, y las preguntas con su rúbrica, en `assessment.md`. Ambos documentos se abren desde la barra de navegación de arriba.

### Para ir más lejos

- Cambia una decisión experimental y justifícala con el resultado en `validation`, no con la intuición.
- Analiza los errores por clase o por segmento: casi siempre se concentran en un subconjunto reconocible.
- Compara costo, precisión y latencia; el mejor modelo no siempre es el que gana por décimas.
- Documenta sesgos, limitaciones y usos para los que **no** recomendarías este modelo.

## 📚 Fuentes

La teoría de arriba no es original de este repositorio: se apoya en la literatura de referencia del área y en los papers originales de cada arquitectura. Estas son las obras concretas, y lo que aporta cada una:

> Las referencias apuntan a las obras; no se reproduce su contenido, la redacción es original.

- Vaswani et al. (2017), *Attention Is All You Need*, NeurIPS — define la autoatención escalada, el bloque residual con normalización de capa y la FFN posición a posición.
- Devlin et al. (2019), *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*, NAACL — introduce el preentrenamiento bidireccional con modelado de lenguaje enmascarado.
- Hinton, Vinyals & Dean (2015), *Distilling the Knowledge in a Neural Network*, NeurIPS Deep Learning Workshop — la destilación con temperatura de la que parte DistilBERT.
- Sanh et al. (2019), *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter* — destilación que reduce tamaño y latencia conservando casi todo el rendimiento.
- Houlsby et al. (2019), *Parameter-Efficient Transfer Learning for NLP*, ICML — módulos adapter entrenables entre capas congeladas.
- Hu et al. (2022), *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR — adaptación de bajo rango que congela los pesos base y aprende una corrección B·A.
- Géron — *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* (3.ª ed., O'Reilly 2022), cap. 16 — tratamiento didáctico de atención y transformers para NLP.

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

Y fuera de la carpeta, tres referencias que esta guía usa: el catálogo `configs/advanced_tracks.yaml` —de donde salen el objetivo, la línea base y las métricas—, el código `src/neural_labs/advanced/training.py` —que define el orden de los pasos y los archivos que escribe cada ejecución— y `docs/experiment-protocol.md`, con la regla general del protocolo.

Los datasets se descargan de su proveedor original y conservan su licencia; este repositorio no los redistribuye ni sustituye una descarga fallida por datos generados.

<!-- nav-bottom -->
## 🧭 Navegación del recorrido

| ⬅️ Laboratorio anterior | 🏠 Índice | Laboratorio siguiente ➡️ |
|---|:---:|---|
| [🏁 Proyecto final: churn de telecomunicaciones](../../labs/24_capstone_real_project/README.md) | [Las 31 rutas](../../parts/README.md) | [🧷 Segmentación semántica con U-Net](../../advanced_labs/26_segmentation_unet/README.md) |

**En este laboratorio:** **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🔬 [Parte 7 — Especializaciones avanzadas](../../parts/07-especializaciones-avanzadas.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/25_transformer_finetuning/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
