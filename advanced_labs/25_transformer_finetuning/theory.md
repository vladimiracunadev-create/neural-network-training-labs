# Teoría — Fine-tuning eficiente de transformer

<!-- nav-top -->
> 🧭 **Ruta 26 / 31** · 🔬 [Parte 7 — Especializaciones avanzadas](../../parts/07-especializaciones-avanzadas.md)
>
> [⬅️ 🏁 Proyecto final: churn de telecomunicaciones](../../labs/24_capstone_real_project/theory.md) · [🏠 Índice de rutas](../../parts/README.md) · [🧷 Segmentación semántica con U-Net ➡️](../../advanced_labs/26_segmentation_unet/theory.md)
>
> [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

Tokenización subword, atención preentrenada, fine-tuning completo y adaptación eficiente LoRA.

## Idea central

Hasta aquí el recorrido entrenaba modelos desde cero. Este laboratorio invierte el punto de partida: se toma un modelo que **ya sabe leer** —preentrenado sobre miles de millones de palabras— y se pregunta cuánto hay que moverlo para que resuelva una tarea concreta. Clasificar titulares en cuatro categorías no requiere volver a aprender qué es un sustantivo ni cómo se relacionan las palabras de una frase: eso ya está en los pesos. Requiere, como mucho, reorientar esa competencia.

La pregunta que organiza el laboratorio es **cuánto del modelo hay que tocar**. El fine-tuning completo mueve los 67 millones de parámetros de DistilBERT; funciona, pero obliga a almacenar una copia entera del modelo por cada tarea, y en modelos grandes el costo de memoria del optimizador multiplica el problema. La alternativa parte de una observación empírica: la *diferencia* entre los pesos preentrenados y los ajustados tiene rango efectivo bajo, es decir, cabe en muchas menos dimensiones de las que ocupa. Si eso es cierto, se puede congelar el modelo y aprender solo esa corrección de rango pequeño.

Ese es el contraste que se mide aquí: **fine-tuning completo frente a LoRA**, comparando no solo la calidad sino el número de parámetros entrenables y la latencia. Y por debajo de ambos, una línea base de TF-IDF con regresión logística, que representa el texto por frecuencias de palabras sin orden ni contexto. Si la línea base queda cerca, la conclusión honesta es que la tarea no necesitaba un transformer; si queda lejos, la distancia mide exactamente cuánto aporta la comprensión lingüística preentrenada.

## Fundamento matemático

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

## Visualización específica

Distribución de longitud, matriz de confusión, atención y comparación LoRA/full. Los mapas de atención revelan qué tokens influyen en la clasificación; la comparación LoRA vs. fine-tuning completo contrasta accuracy y macro_f1 frente al número de parámetros entrenables y la latencia, para juzgar el coste-beneficio de cada estrategia.

## Riesgo de interpretación

El corpus contiene titulares históricos y sesgos editoriales; no representa todo el lenguaje contemporáneo. Además, un mapa de atención alto no implica causalidad ni "explicación" fiable de la decisión: la atención es una entre varias señales internas del modelo y debe interpretarse con cautela.

## 🔗 Referencias

> Las referencias apuntan a las obras; no se reproduce su contenido, la redacción es original.

- Vaswani et al. (2017), *Attention Is All You Need*, NeurIPS — define la autoatención escalada, el bloque residual con normalización de capa y la FFN posición a posición.
- Devlin et al. (2019), *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*, NAACL — introduce el preentrenamiento bidireccional con modelado de lenguaje enmascarado.
- Hinton, Vinyals & Dean (2015), *Distilling the Knowledge in a Neural Network*, NeurIPS Deep Learning Workshop — la destilación con temperatura de la que parte DistilBERT.
- Sanh et al. (2019), *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter* — destilación que reduce tamaño y latencia conservando casi todo el rendimiento.
- Houlsby et al. (2019), *Parameter-Efficient Transfer Learning for NLP*, ICML — módulos adapter entrenables entre capas congeladas.
- Hu et al. (2022), *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR — adaptación de bajo rango que congela los pesos base y aprende una corrección B·A.
- Géron — *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* (3.ª ed., O'Reilly 2022), cap. 16 — tratamiento didáctico de atención y transformers para NLP.
- Fuente del dataset: https://huggingface.co/datasets/fancyzhx/ag_news — **AG News Topic Classification Dataset** (Distribuido por Hugging Face Datasets, La ficha de Hugging Face declara `unknown`); procedencia, versión y SHA-256 en el registro de fuentes, entrada `ag-news` — esta clase la usa para contrastar fine-tuning completo, LoRA y una línea base TF-IDF en la clasificación de titulares reales.

<!-- nav-bottom -->
## 🧭 Navegación del recorrido

| ⬅️ Laboratorio anterior | 🏠 Índice | Laboratorio siguiente ➡️ |
|---|:---:|---|
| [🏁 Proyecto final: churn de telecomunicaciones](../../labs/24_capstone_real_project/README.md) | [Las 31 rutas](../../parts/README.md) | [🧷 Segmentación semántica con U-Net](../../advanced_labs/26_segmentation_unet/README.md) |

**En este laboratorio:** [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🔬 [Parte 7 — Especializaciones avanzadas](../../parts/07-especializaciones-avanzadas.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/25_transformer_finetuning/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
