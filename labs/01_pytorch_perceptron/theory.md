# Teoría — Perceptrón con PyTorch

<!-- nav-top -->
> 🧭 **Ruta 2 / 31** · 🟢 [Parte 1 — Fundamentos: de la derivada a la primera red](../../parts/01-fundamentos.md)
>
> [⬅️ 🔢 Neurona con NumPy](../../labs/00_numpy_neuron/theory.md) · [🏠 Índice de rutas](../../parts/README.md) · [🌀 MLP multiclase ➡️](../../labs/02_mlp_nonlinear/theory.md)
>
> [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

## Propósito

Aprender tensores, autograd, optimizadores y un clasificador lineal.

## Idea central

Este laboratorio estudia **clasificador lineal con autograd** usando `banknote_authentication`, un dataset público real procedente de UCI.

El salto respecto al laboratorio anterior no está en las matemáticas —seguimos entrenando esencialmente una neurona logística— sino en la ingeniería: en lugar de derivar los gradientes a mano, dejamos que PyTorch los construya automáticamente. Cada operación sobre un tensor con `requires_grad=True` se registra en un grafo de cómputo dinámico; al invocar `loss.backward()`, el motor `autograd` recorre ese grafo en sentido inverso aplicando la regla de la cadena y deposita en `.grad` exactamente las mismas derivadas que en el lab 00 escribimos manualmente.

El problema —distinguir billetes auténticos de falsos a partir de cuatro estadísticos extraídos por transformada wavelet de imágenes reales— es casi linealmente separable, lo que lo hace ideal para verificar que la maquinaria (tensores, `Dataset`/`DataLoader`, optimizador, bucle de entrenamiento) funciona antes de abordar problemas donde un solo hiperplano ya no basta. La pregunta crítica del laboratorio anticipa precisamente esa limitación.

## Fundamento matemático

El modelo calcula un **logit** —una puntuación real sin normalizar— mediante una transformación afín de las entradas:

z = x·W + b

Nótese que ahora trabajamos con lotes (batches): x es una matriz de forma (N, d) y la multiplicación x·W produce un vector de N logits en paralelo, aprovechando el álgebra matricial vectorizada. La probabilidad se obtiene, igual que antes, con la sigmoide p = σ(z) = 1 / (1 + e⁻ᶻ), pero aquí introducimos una diferencia importante de estabilidad numérica.

En lugar de calcular σ(z) y luego la entropía cruzada por separado, PyTorch ofrece `BCEWithLogitsLoss`, que **fusiona la sigmoide y la log-verosimilitud en una sola operación numéricamente estable**. La razón es que combinar exponencial y logaritmo por separado desborda con logits grandes; la forma fusionada aplica el truco log-sum-exp:

L = (1/N) Σᵢ [ max(zᵢ, 0) − zᵢ·yᵢ + ln(1 + e^(−|zᵢ|)) ]

que es algebraicamente idéntica a −(1/N) Σᵢ [ yᵢ·ln σ(zᵢ) + (1−yᵢ)·ln(1−σ(zᵢ)) ] pero no produce ni ∞ ni NaN en los extremos. Por eso la buena práctica es que la última capa devuelva **logits crudos** y la pérdida se encargue internamente de la sigmoide.

La magia de `autograd` es que, definida L, no necesitamos escribir ∂L/∂W. El grafo sabe que ∂L/∂z = (σ(z) − y)/N y propaga hacia atrás por la regla de la cadena hasta ∂L/∂W = xᵀ·(σ(z) − y)/N. El optimizador (por ejemplo SGD o Adam) consume esos gradientes y actualiza los parámetros θ ← θ − η·∇_θ L, encapsulando la regla de actualización que en el lab 00 escribíamos línea por línea. El ciclo canónico es: `optimizer.zero_grad()` → `loss.backward()` → `optimizer.step()`; olvidar el `zero_grad` acumula gradientes de iteraciones previas, un error clásico que el laboratorio permite observar.

La formulación debe conectarse con cuatro elementos: representación de entrada, función del modelo, función de pérdida y regla de actualización. El notebook muestra las dimensiones de los tensores y conserva la misma implementación que el script de terminal.

## Protocolo científico

- Ajustar transformaciones, vocabulario, normalización y selección de variables solo con `train`.
- Usar `validation` para arquitectura, hiperparámetros, checkpoint y umbrales.
- Evaluar `test` una vez, después de congelar las decisiones.
- Comparar contra **Regresión logística**.
- Reportar variación entre semillas e intervalos de confianza; una métrica puntual no expresa toda la incertidumbre.

## Riesgos de interpretación

Características extraídas de imágenes reales de billetes.

El dataset refleja su proceso de recolección y no representa automáticamente otros períodos, países o poblaciones. Una asociación predictiva no demuestra causalidad.

## Pregunta crítica

> ¿Qué ejemplos no puede separar un único hiperplano?

## 🔗 Referencias

> Las referencias apuntan a las obras; no se reproduce su contenido, la redacción es original.

- Géron — *Hands-On Machine Learning* (3.ª ed., O'Reilly 2022), cap. 10 — introducción a redes con Keras/PyTorch, logits y funciones de pérdida.
- Goodfellow, Bengio & Courville — *Deep Learning* (MIT Press 2016), cap. 6 — redes hacia adelante, entropía cruzada y descenso de gradiente basado en grafos.
- Zhang et al. — *Dive into Deep Learning* (d2l.ai, 2023), cap. 3–5 — regresión lineal/softmax en frameworks modernos y mecánica de entrenamiento.
- Paszke et al. (2019), *PyTorch: An Imperative Style, High-Performance Deep Learning Library*, NeurIPS — diseño del framework y del motor de diferenciación automática.
- Documentación oficial de PyTorch (autograd) — https://pytorch.org/docs/stable/notes/autograd.html — grafo dinámico y semántica de `backward()`.
- Fuente del dataset: https://archive.ics.uci.edu/dataset/267/banknote+authentication
- Consulte `docs/experiment-protocol.md`, `docs/reproducibility.md` y `docs/ethics-and-licenses.md`.

<!-- nav-bottom -->
## 🧭 Navegación del recorrido

| ⬅️ Laboratorio anterior | 🏠 Índice | Laboratorio siguiente ➡️ |
|---|:---:|---|
| [🔢 Neurona con NumPy](../../labs/00_numpy_neuron/README.md) | [Las 31 rutas](../../parts/README.md) | [🌀 MLP multiclase](../../labs/02_mlp_nonlinear/README.md) |

**En este laboratorio:** [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🟢 [Parte 1 — Fundamentos: de la derivada a la primera red](../../parts/01-fundamentos.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/01_pytorch_perceptron/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
