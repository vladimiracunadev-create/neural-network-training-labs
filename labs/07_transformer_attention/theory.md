# Teoría — Transformer para noticias

<!-- nav-top -->
> 🧭 **Ruta 8 / 31** · 🔵 [Parte 2 — Arquitecturas según la forma del dato](../../parts/02-arquitecturas.md)
>
> [⬅️ 🧬 Autoencoder para fraude](../../labs/06_autoencoder_anomaly/theory.md) · [🏠 Índice de rutas](../../parts/README.md) · [🎨 GAN generativa ➡️](../../labs/08_gan_generation/theory.md)
>
> [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

## Propósito

Aplicar atención multi-cabeza a clasificación de noticias reales.

## Idea central

Este laboratorio estudia **autoatención para clasificación de texto** usando `ag_news`, un dataset público real procedente de Hugging Face.

La intuición central es que el significado de una palabra depende de su contexto, y ese contexto puede estar lejos en la secuencia. Los modelos recurrentes procesan el texto token a token y arrastran la información en un estado que se degrada con la distancia. La **autoatención** rompe con eso: permite que cada token mire a *todos* los demás en un solo paso y pondere cuánto atender a cada uno según su relevancia. Para clasificar una noticia como "Deportes", "Mundo", "Negocios" o "Ciencia/Tecnología", el modelo aprende a concentrar la atención en las palabras discriminantes (nombres de equipos, términos financieros, etc.) sin importar en qué posición aparezcan.

Cada token se proyecta en tres roles: una **consulta** (query) que expresa "qué busco", una **clave** (key) que expresa "qué ofrezco" y un **valor** (value) que es la información a transmitir. La compatibilidad entre la consulta de un token y las claves de los demás determina los pesos con que se combinan los valores. Al apilar varias "cabezas" de atención en paralelo, el modelo puede atender simultáneamente a distintos tipos de relación (sintáctica, semántica, de correferencia). Este laboratorio construye el transformer desde cero para ver cómo estas piezas producen una representación contextual de toda la noticia.

## Fundamento matemático

El bloque de atención escalada por producto punto se define sobre matrices Q ∈ ℝ^{n×d_k}, K ∈ ℝ^{n×d_k} y V ∈ ℝ^{n×d_v}, donde n es el número de tokens:

    Attention(Q, K, V) = softmax( Q Kᵀ / √d_k ) V

La matriz Q Kᵀ contiene, en su entrada (i, j), el producto punto entre la consulta del token i y la clave del token j: un puntaje de compatibilidad. La división por √d_k es esencial —no cosmética—: cuando d_k es grande, los productos punto crecen en magnitud proporcionalmente a √d_k, y sin escalar empujarían al softmax a regiones de gradiente casi nulo (saturación), frenando el aprendizaje. Dividir por √d_k mantiene la varianza de los puntajes controlada. El **softmax** por filas convierte cada fila en una distribución de pesos que suma 1, y al multiplicar por V se obtiene, para cada token, un promedio ponderado de los valores de toda la secuencia: su nueva representación contextual.

La **atención multi-cabeza** ejecuta h proyecciones distintas en paralelo. Con matrices aprendidas W_iᵠ, W_iᴷ, W_iⱽ para cada cabeza i, se calcula headᵢ = Attention(Q W_iᵠ, K W_iᴷ, V W_iⱽ), y luego se concatenan y se proyectan: MultiHead(Q,K,V) = Concat(head₁, …, head_h) Wᴼ. Cada cabeza opera en un subespacio de dimensión d_k = d_model / h, de modo que el coste total es comparable al de una sola cabeza pero con mayor capacidad de representar relaciones diversas. Como la atención es invariante al orden (permutar los tokens permuta las filas pero no cambia las relaciones), se suma una **codificación posicional** a los embeddings de entrada para inyectar la noción de posición; sin ella el modelo no distinguiría "el perro muerde al hombre" de "el hombre muerde al perro".

Conectando con los cuatro elementos del laboratorio: la **representación de entrada** es la secuencia de embeddings de tokens más su codificación posicional; la **función del modelo** apila bloques de atención multi-cabeza + red feed-forward con conexiones residuales y normalización de capa, y agrega la secuencia en un vector para clasificar; la **función de pérdida** es la entropía cruzada categórica sobre las cuatro clases, ℒ = −Σ_c y_c log ŷ_c, donde ŷ = softmax de los logits; y la **regla de actualización** es descenso de gradiente (Adam) con θ ← θ − η ∇_θ ℒ. El notebook muestra las dimensiones de los tensores (n, d_model, h, d_k) en cada etapa y conserva la misma implementación que el script de terminal.

### Por qué se divide por √d_k, con el cálculo delante

El factor 1/√d_k es la parte de la fórmula que más se copia sin entender, y tiene una justificación estadística exacta. Supóngase que las componentes de q y k son variables aleatorias independientes de media 0 y varianza 1. Su producto escalar es q·k = Σ_(i=1..d_k) qᵢ·kᵢ, una suma de d_k términos independientes de media 0 y varianza 1, de modo que

𝔼[q·k] = 0,   Var(q·k) = d_k,   desviación típica = √d_k.

Es decir, **la magnitud típica del producto escalar crece con √d_k**. Con d_k = 64, los logits de atención tendrían una desviación típica de 8: valores que el softmax convierte casi en un one-hot, concentrando toda la atención en un único token. Y un softmax saturado tiene gradiente prácticamente nulo —la misma patología de la sigmoide en la ruta 00—, así que la atención dejaría de aprender a quién mirar. Dividir por √d_k devuelve la varianza a 1 y mantiene el softmax en su zona sensible, independientemente de la dimensión elegida.

De ahí también se entiende para qué sirven varias **cabezas**. Con d_model = 256 se podría hacer una sola atención de d_k = 256, pero se prefieren, por ejemplo, 8 cabezas de d_k = 32 cada una. El costo en parámetros es idéntico —las proyecciones suman lo mismo—, y a cambio el modelo obtiene ocho relaciones distintas en subespacios distintos, que luego concatena y mezcla con W_O. Una cabeza sola tiene que comprometer una única distribución de atención para todos los tipos de relación; ocho cabezas pueden especializarse, y en la práctica se observa que unas siguen la posición contigua, otras enlazan sujeto y verbo, otras marcan tokens raros.

### Lo que el transformer gana y lo que paga frente a la recurrencia

La comparación con la ruta 04 se puede hacer con dos números, y explica el cambio de paradigma completo.

**Camino de información.** En una RNN, la señal entre las posiciones i y j debe atravesar |i − j| pasos recurrentes, multiplicándose por otras tantas matrices —de ahí el desvanecimiento—. En la atención, cualquier par de posiciones está conectado por **un solo** producto escalar: el camino máximo es O(1). Esa es la razón de fondo por la que los transformers capturan dependencias largas que a una RNN se le escapan, y no una cuestión de tamaño.

**Costo.** La contrapartida es que construir la matriz de atención exige comparar todos los pares: complejidad **O(n²·d)** en tiempo y en memoria, frente a la O(n·d²) de la recurrencia. Para los titulares cortos de este laboratorio n es pequeño y el cuadrado no duele; para documentos de miles de tokens, esa cuadrática es la razón de ser de toda una línea de investigación en atención eficiente.

**Paralelismo.** Una RNN debe calcular hₜ antes que hₜ₊₁: su cómputo es secuencial por construcción y no se puede repartir en el tiempo. La atención procesa todas las posiciones a la vez, y por eso aprovecha el hardware paralelo de forma que la recurrencia nunca podrá. Esta ventaja práctica —no una superioridad teórica— es la que decidió la adopción masiva de la arquitectura.

Hay un precio conceptual: al mirar todas las posiciones simultáneamente, **la atención no sabe nada del orden**. Es una operación permutación-equivariante, así que sin información posicional «el perro mordió al hombre» y «el hombre mordió al perro» tendrían representaciones idénticas para el conjunto de tokens. La codificación posicional que se suma a los embeddings existe únicamente para romper esa simetría, y por eso no es un adorno: sin ella el modelo pierde toda la sintaxis.

### Qué es y qué no es un mapa de atención

Los mapas de atención se visualizan en este laboratorio y conviene interpretarlos con precisión. Cada fila de softmax(Q·Kᵀ/√d_k) es una distribución de probabilidad: suma 1 y dice qué mezcla de valores V construye la representación de esa posición. Eso es todo lo que dice.

En particular, **no es una explicación de la decisión**. Un peso alto significa que ese token contribuyó a la mezcla en esa capa y esa cabeza, no que la predicción dependa causalmente de él: la información puede haber viajado por la conexión residual, haber sido reescrita por la FFN, o repartirse entre varias cabezas que se compensan. Se han construido modelos con mapas de atención muy distintos y predicciones idénticas, que es la prueba de que la atención no identifica de forma única la causa. Para afirmar dependencia causal hacen falta las técnicas de la ruta 21 —perturbar la entrada y medir el cambio en la salida—, no leer los pesos.

Históricamente la atención nació como mecanismo de *alineamiento* en traducción (Bahdanau et al., 2015), donde el decodificador aprendía a qué palabras de la frase origen mirar en cada paso. La contribución de Vaswani et al. (2017) fue mostrar que la atención por sí sola —sin recurrencia ni convolución— basta para modelar secuencias, lo que además desbloquea el paralelismo masivo que hizo posibles los modelos de lenguaje actuales.

## Protocolo científico

- Ajustar transformaciones, vocabulario, normalización y selección de variables solo con `train`.
- Usar `validation` para arquitectura, hiperparámetros, checkpoint y umbrales.
- Evaluar `test` una vez, después de congelar las decisiones.
- Comparar contra **TF-IDF + regresión logística**.
- Reportar variación entre semillas e intervalos de confianza; una métrica puntual no expresa toda la incertidumbre.

## Riesgos de interpretación

Noticias reales en cuatro categorías con particiones públicas.

El dataset refleja su proceso de recolección y no representa automáticamente otros períodos, países o poblaciones. Una asociación predictiva no demuestra causalidad.

## Pregunta crítica

> ¿La atención observada coincide con evidencia útil para la clase?

> Las referencias apuntan a las obras; no se reproduce su contenido, la redacción es original.

## 🔗 Referencias

- Goodfellow, Bengio & Courville — *Deep Learning* (MIT Press, 2016), cap. 10 — modelado de secuencias y mecanismos de atención sobre redes recurrentes.
- Géron — *Hands-On Machine Learning* (3.ª ed., O'Reilly), cap. 16 — procesamiento de secuencias con RNN y atención, transformer paso a paso.
- Prince — *Understanding Deep Learning* (MIT Press, 2024), cap. 12 — desarrollo moderno y didáctico de la autoatención y la arquitectura transformer.
- Bahdanau, Cho & Bengio (2015), *Neural Machine Translation by Jointly Learning to Align and Translate*, ICLR — introdujo la atención como alineamiento suave en secuencia-a-secuencia.
- Vaswani et al. (2017), *Attention Is All You Need*, NeurIPS — la arquitectura transformer basada íntegramente en autoatención multi-cabeza.
- Fuente del dataset: https://huggingface.co/datasets/fancyzhx/ag_news — **AG News Topic Classification Dataset** (Distribuido por Hugging Face Datasets, La ficha de Hugging Face declara `unknown`); procedencia, versión y SHA-256 en el registro de fuentes, entrada `ag-news` — esta clase la usa para aplicar atención multi-cabeza a la clasificación de titulares reales en cuatro categorías.
- Consulte `docs/experiment-protocol.md`, `docs/reproducibility.md` y `docs/ethics-and-licenses.md`.

<!-- nav-bottom -->
## 🧭 Navegación del recorrido

| ⬅️ Laboratorio anterior | 🏠 Índice | Laboratorio siguiente ➡️ |
|---|:---:|---|
| [🧬 Autoencoder para fraude](../../labs/06_autoencoder_anomaly/README.md) | [Las 31 rutas](../../parts/README.md) | [🎨 GAN generativa](../../labs/08_gan_generation/README.md) |

**En este laboratorio:** [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🔵 [Parte 2 — Arquitecturas según la forma del dato](../../parts/02-arquitecturas.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/07_transformer_attention/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
