# Teoría — RNN para texto

<!-- nav-top -->
> 🧭 **Ruta 5 / 31** · 🔵 [Parte 2 — Arquitecturas según la forma del dato](../../parts/02-arquitecturas.md)
>
> [⬅️ 🖼️ CNN para visión](../../labs/03_cnn_vision/theory.md) · [🏠 Índice de rutas](../../parts/README.md) · [📈 LSTM para series temporales ➡️](../../labs/05_lstm_time_series/theory.md)
>
> [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

## Propósito

Clasificar sentimiento en reseñas reales usando embeddings y recurrencia.

## Idea central

Este laboratorio estudia **recurrencia sobre secuencias tokenizadas** usando `imdb`, un dataset público real procedente de Hugging Face / Stanford.

El texto es una secuencia de longitud variable donde el orden importa: "no es buena" y "es buena, no" significan cosas distintas. Un MLP o una CNN de tamaño fijo no capturan bien esa dependencia temporal. La red recurrente (RNN) procesa la secuencia token a token manteniendo un **estado oculto** que resume todo lo visto hasta el momento, de modo que la predicción en cada paso depende del contexto acumulado. Es, en esencia, una memoria que se actualiza con cada palabra.

Dos piezas colaboran aquí. Primero, los **embeddings**: cada token del vocabulario se representa por un vector denso aprendible, de forma que palabras con uso similar acaban cerca en el espacio vectorial (la idea que popularizó word2vec). Segundo, la **recurrencia**, que integra esos vectores en el tiempo. El laboratorio clasifica el sentimiento de reseñas de cine reales y se contrasta contra un TF-IDF + regresión logística, que ignora el orden; la comparación revela cuándo la estructura secuencial realmente aporta.

## Fundamento matemático

Una RNN recibe la secuencia de vectores de embedding x₁, x₂, …, x_T (uno por token) y mantiene un estado oculto hₜ que se recalcula en cada paso combinando la entrada actual con el estado anterior:

hₜ = tanh(Wₓ·xₜ + W_h·hₜ₋₁ + b)

La intuición es una recursión: hₜ es una síntesis comprimida de todo el prefijo x₁…xₜ. La matriz Wₓ decide cómo entra la nueva palabra, W_h decide cómo se transforma la memoria previa, y la **tanh** (con rango en (−1, 1)) mantiene el estado acotado e introduce la no linealidad. Un detalle esencial es que Wₓ, W_h y b **se comparten en todos los pasos de tiempo**: es el mismo conjunto de pesos aplicado en cada instante, análogo al peso compartido de las CNN pero a lo largo del eje temporal. Esto permite procesar secuencias de cualquier longitud con un número fijo de parámetros. Para clasificar sentimiento se usa el último estado h_T (o un agregado de todos), que pasa por una capa densa y una sigmoide para dar la probabilidad de reseña positiva.

El entrenamiento usa **retropropagación en el tiempo** (BPTT): se "desenrolla" la red en T copias y el gradiente fluye hacia atrás desde h_T hasta h₁. Aquí surge el problema fundamental de las RNN simples. Al aplicar la regla de la cadena a lo largo de T pasos, el gradiente respecto a estados lejanos contiene un producto de T factores del tipo W_h⊤ diag(tanh′). Si los valores propios dominantes de ese producto son menores que 1, el gradiente **se desvanece** exponencialmente (∝ λᵀ con λ < 1) y la red no aprende dependencias largas; si son mayores que 1, **explota**.

∂h_T/∂h₁ = Πₜ (∂hₜ/∂hₜ₋₁)  →  se contrae o crece exponencialmente con T

Este es exactamente el motivo por el que las reseñas se truncan y por el que existen arquitecturas con puertas (LSTM/GRU), tema del laboratorio siguiente. La explosión se mitiga en la práctica con **recorte de gradiente** (gradient clipping), que limita la norma del gradiente a un umbral antes de actualizar; el desvanecimiento, en cambio, exige cambiar la propia celda recurrente.

La formulación debe conectarse con cuatro elementos: representación de entrada, función del modelo, función de pérdida y regla de actualización. El notebook muestra las dimensiones de los tensores y conserva la misma implementación que el script de terminal.

## Protocolo científico

- Ajustar transformaciones, vocabulario, normalización y selección de variables solo con `train`.
- Usar `validation` para arquitectura, hiperparámetros, checkpoint y umbrales.
- Evaluar `test` una vez, después de congelar las decisiones.
- Comparar contra **TF-IDF + regresión logística**.
- Reportar variación entre semillas e intervalos de confianza; una métrica puntual no expresa toda la incertidumbre.

## Riesgos de interpretación

Reseñas cinematográficas reales con partición oficial.

El dataset refleja su proceso de recolección y no representa automáticamente otros períodos, países o poblaciones. Una asociación predictiva no demuestra causalidad.

## Pregunta crítica

> ¿Qué información se pierde al truncar las reseñas?

## 🔗 Referencias

> Las referencias apuntan a las obras; no se reproduce su contenido, la redacción es original.

- Goodfellow, Bengio & Courville — *Deep Learning* (MIT Press 2016), cap. 10 — redes recurrentes, BPTT y el problema del gradiente que se desvanece.
- Géron — *Hands-On Machine Learning* (3.ª ed., O'Reilly 2022), cap. 16 — procesamiento de lenguaje con RNN y embeddings.
- Prince — *Understanding Deep Learning* (MIT Press 2024), cap. 12 — modelado de secuencias y arquitecturas recurrentes.
- Elman (1990), *Finding Structure in Time*, Cognitive Science — RNN fundacional con estado oculto recurrente.
- Mikolov et al. (2013), *Efficient Estimation of Word Representations in Vector Space (word2vec)* — embeddings distribuidos de palabras.
- Fuente del dataset: https://huggingface.co/datasets/stanfordnlp/imdb
- Consulte `docs/experiment-protocol.md`, `docs/reproducibility.md` y `docs/ethics-and-licenses.md`.

<!-- nav-bottom -->
## 🧭 Navegación del recorrido

| ⬅️ Laboratorio anterior | 🏠 Índice | Laboratorio siguiente ➡️ |
|---|:---:|---|
| [🖼️ CNN para visión](../../labs/03_cnn_vision/README.md) | [Las 31 rutas](../../parts/README.md) | [📈 LSTM para series temporales](../../labs/05_lstm_time_series/README.md) |

**En este laboratorio:** [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🔵 [Parte 2 — Arquitecturas según la forma del dato](../../parts/02-arquitecturas.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/04_rnn_sequences/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
