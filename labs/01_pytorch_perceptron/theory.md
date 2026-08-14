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

### Qué es realmente `autograd`: diferenciación en modo inverso

Decir que «el grafo calcula la derivada sola» esconde el mecanismo, y conviene abrirlo porque explica el costo de todo entrenamiento posterior.

Al ejecutar el paso hacia adelante, PyTorch construye un **grafo acíclico dirigido** donde cada nodo es una operación y guarda lo necesario para derivarse. No es simbólico —no manipula fórmulas— ni numérico —no usa diferencias finitas—: es **diferenciación automática**, que aplica la regla de la cadena sobre operaciones elementales cuyas derivadas están programadas exactamente.

La regla de la cadena se puede recorrer en dos sentidos, y la elección importa. Para una función f: ℝⁿ → ℝᵐ, el modo **directo** propaga derivadas desde las entradas y calcula una columna del jacobiano por pasada, así que cuesta n pasadas. El modo **inverso** propaga desde la salida y calcula una fila por pasada: cuesta m pasadas. En una red neuronal, n son los parámetros —millones— y m es 1, porque la pérdida es un escalar. De ahí que el modo inverso sea el correcto: **una sola** pasada hacia atrás produce el gradiente respecto de todos los parámetros a la vez.

El resultado práctico es la regla de oro del costo: el paso hacia atrás cuesta aproximadamente **el doble** que el paso hacia adelante, sin importar cuántos parámetros haya. Lo que sí crece con la profundidad es la memoria, porque hay que conservar las activaciones intermedias hasta que el gradiente pase por ellas. Esa es la razón de que el consumo de memoria escale con el tamaño de lote y con el número de capas, y de que existan técnicas como el *gradient checkpointing*, que recalcula activaciones en vez de guardarlas.

Cada nodo no almacena una matriz jacobiana, que sería inmanejable, sino la operación **producto vector-jacobiano**: dado el gradiente que llega desde arriba, v, devuelve vᵀ·J sin construir J. Para la capa lineal z = x·W + b eso se traduce en las tres expresiones que el laboratorio puede verificar a mano:

∂L/∂W = xᵀ·(∂L/∂z),   ∂L/∂b = Σ_filas (∂L/∂z),   ∂L/∂x = (∂L/∂z)·Wᵀ.

La tercera es la que permite encadenar capas: es el gradiente que esta capa entrega a la anterior.

### Por qué hay que llamar a `zero_grad`, y qué dice el minilote

El detalle que más errores causa tiene una explicación de diseño. PyTorch **acumula** gradientes en el atributo `.grad` en lugar de sobrescribirlos, es decir, `backward()` hace `p.grad += nuevo` y no `p.grad = nuevo`. Eso es deliberado: permite sumar gradientes de varios pasos hacia atrás antes de actualizar, que es exactamente lo que se necesita para simular un lote grande sin memoria para él (*gradient accumulation*), o para redes con varias salidas. El precio es que, en el bucle normal, olvidar `optimizer.zero_grad()` hace que la actualización de la iteración k use la suma de los gradientes de las iteraciones 1..k, un error que no lanza excepción y solo se manifiesta como un entrenamiento que no converge.

El otro concepto que aparece aquí por primera vez es el **minilote**. El gradiente sobre un lote de tamaño B es un estimador **insesgado** del gradiente sobre todo el conjunto: si se muestrea uniformemente, 𝔼[∇L_B] = ∇L. Su varianza, en cambio, decrece como σ²/B, así que la desviación típica del estimador baja con **√B**. De ahí se sigue el compromiso que gobierna la elección del tamaño de lote: cuadruplicar B cuesta cuatro veces más cómputo por paso y solo reduce el ruido a la mitad. Lotes pequeños dan pasos ruidosos —y ese ruido, lejos de ser solo un defecto, ayuda a escapar de mínimos estrechos—; lotes grandes dan direcciones precisas pero aprovechan peor el cómputo y suelen necesitar una tasa de aprendizaje mayor para avanzar lo mismo.

Ese es también el motivo por el que el orden en que se barajan los ejemplos forma parte de `training_seed` y no de `split_seed`: no cambia qué datos hay en cada partición, cambia la trayectoria del entrenamiento.

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
