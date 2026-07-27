# Teoría — Fine-tuning eficiente de transformer

Tokenización subword, atención preentrenada, fine-tuning completo y adaptación eficiente LoRA.

## Fundamento matemático

Un transformer procesa el texto tras convertirlo en tokens subword (WordPiece en el caso de DistilBERT). Cada token se representa por un vector de embedding al que se le suma una codificación posicional, formando una matriz X ∈ ℝ^(n×d), donde n es la longitud de secuencia y d la dimensión oculta. El corazón del modelo es la **autoatención**: a partir de X se proyectan consultas, claves y valores mediante matrices aprendidas, Q = X·W_Q, K = X·W_K, V = X·W_V, y se calcula

Attention(Q, K, V) = softmax( (Q·Kᵀ) / √d_k ) · V.

La división por √d_k evita que los productos escalares crezcan con la dimensión y saturen el softmax; la matriz softmax(Q·Kᵀ/√d_k) es la que se visualiza como "mapa de atención", pues cada fila indica cuánto pesa cada token del contexto al construir la representación de un token dado. Con varias cabezas (multi-head) el modelo atiende simultáneamente a distintos subespacios de relación.

El preentrenamiento (BERT) optimiza un objetivo de **modelado de lenguaje enmascarado**: se ocultan tokens aleatorios y la red minimiza la entropía cruzada al predecirlos, ℒ = −Σᵢ log p(xᵢ | x_contexto). Así el modelo aprende representaciones lingüísticas generales antes de ver la tarea final. DistilBERT es una versión **destilada**: un modelo "estudiante" más pequeño se entrena para imitar al "profesor" BERT, combinando la pérdida supervisada con un término de destilación sobre las distribuciones suaves del profesor (softmax con temperatura), conservando ~97% del rendimiento con ~40% menos parámetros.

En el **fine-tuning completo** se añade una capa de clasificación sobre el embedding del token especial [CLS] y se actualizan *todos* los pesos θ del modelo mediante descenso de gradiente sobre la entropía cruzada de la tarea, θ ← θ − α·∇_θ ℒ. Es potente pero costoso: hay que almacenar y actualizar decenas de millones de parámetros por tarea. La **adaptación eficiente LoRA** parte de la observación de que la actualización necesaria ΔW tiene rango efectivo bajo. En lugar de modificar la matriz preentrenada W₀ ∈ ℝ^(d×k), LoRA la congela y aprende una corrección factorizada de rango r pequeño:

W = W₀ + ΔW = W₀ + (α/r)·B·A,   con B ∈ ℝ^(d×r), A ∈ ℝ^(r×k), r ≪ min(d, k).

Solo se entrenan A y B (más el factor de escala α/r), reduciendo los parámetros entrenables en órdenes de magnitud sin añadir latencia en inferencia (las matrices pueden fusionarse en W). Esta es la esencia del *parameter-efficient transfer learning*, emparentada con los **adapters** de Houlsby et al., que insertan pequeños módulos entrenables entre capas congeladas. La línea base TF-IDF + regresión logística sirve de contraste: representa el texto por frecuencias de término ponderadas, sin capturar orden ni contexto, y ayuda a medir cuánto aporta realmente la atención preentrenada.

## Visualización específica

Distribución de longitud, matriz de confusión, atención y comparación LoRA/full. Los mapas de atención revelan qué tokens influyen en la clasificación; la comparación LoRA vs. fine-tuning completo contrasta accuracy y macro_f1 frente al número de parámetros entrenables y la latencia, para juzgar el coste-beneficio de cada estrategia.

## Riesgo de interpretación

El corpus contiene titulares históricos y sesgos editoriales; no representa todo el lenguaje contemporáneo. Además, un mapa de atención alto no implica causalidad ni "explicación" fiable de la decisión: la atención es una entre varias señales internas del modelo y debe interpretarse con cautela.

## 🔗 Referencias

> Las referencias apuntan a las obras; no se reproduce su contenido, la redacción es original.

- Devlin et al. (2019), *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*, NAACL — introduce el preentrenamiento bidireccional con modelado de lenguaje enmascarado.
- Sanh et al. (2019), *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter* — destilación que reduce tamaño y latencia conservando casi todo el rendimiento.
- Houlsby et al. (2019), *Parameter-Efficient Transfer Learning for NLP*, ICML — módulos adapter entrenables entre capas congeladas.
- Hu et al. (2022), *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR — adaptación de bajo rango que congela los pesos base y aprende una corrección B·A.
- Géron — *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* (3.ª ed., O'Reilly 2022), cap. 16 — tratamiento didáctico de atención y transformers para NLP.
