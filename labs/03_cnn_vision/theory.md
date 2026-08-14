# Teoría — CNN para visión

<!-- nav-top -->
> 🧭 **Ruta 4 / 31** · [⬅️ 🌀 MLP multiclase](../../labs/02_mlp_nonlinear/theory.md) · [🏠 Índice](../../README.md#laboratorios) · [🔁 RNN para texto ➡️](../../labs/04_rnn_sequences/theory.md)
>
> [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

## Propósito

Entrenar una CNN y analizar errores sobre fotografías reales de diez clases.

## Idea central

Este laboratorio estudia **convoluciones para patrones espaciales** usando `cifar10`, un dataset público real procedente de Torchvision / University of Toronto.

Aplanar una imagen de 32×32×3 y pasarla por un MLP funciona, pero desperdicia la estructura del problema: ignora que los píxeles vecinos están correlacionados y que un objeto es el mismo aunque se desplace unos píxeles. La red convolucional (CNN) incorpora dos sesgos inductivos que encajan con las imágenes: **localidad** (cada neurona mira solo una región pequeña) e **invariancia por traslación** (el mismo filtro se aplica en toda la imagen, compartiendo parámetros). Esto reduce drásticamente el número de pesos y obliga al modelo a aprender detectores de patrones reutilizables.

La consecuencia es una jerarquía de representaciones: las primeras capas aprenden bordes y colores, las intermedias combinan esos bordes en texturas y partes, y las profundas responden a objetos completos. El laboratorio entrena esta jerarquía sobre CIFAR-10 (60.000 fotografías reales de 10 clases) y contrasta contra un clasificador lineal sobre píxeles para hacer patente cuánto aporta explotar la estructura espacial.

## Fundamento matemático

La operación central es la **convolución**: un filtro (kernel) de pesos K de tamaño pequeño se desliza sobre la imagen de entrada X y, en cada posición, calcula un producto punto local. Para un filtro de tamaño F×F sobre un mapa de entrada:

Y(i, j) = Σₘ Σₙ X(i+m, j+n)·K(m, n) + b

El mismo K se reutiliza en todas las posiciones (i, j): eso es el **peso compartido** que da la invariancia por traslación y reduce los parámetros de millones (como en una capa densa) a apenas F×F por canal de filtro. Cada filtro produce un **mapa de activación** que señala dónde aparece el patrón que ese filtro detecta. Tras la convolución se aplica una no linealidad (ReLU), sin la cual toda la pila colapsaría a una única convolución lineal.

El tamaño del mapa de salida depende del *stride* s (paso del deslizamiento) y del *padding* p (relleno de bordes): dimensión_salida = ⌊(dimensión_entrada − F + 2p)/s⌋ + 1. El **pooling** (típicamente max-pooling) submuestrea cada región tomando su máximo, lo que reduce la resolución espacial, aporta cierta invariancia a pequeñas traslaciones y amplía el campo receptivo de las capas posteriores sin añadir parámetros.

Un ingrediente decisivo para entrenar redes profundas es la **normalización por lotes** (batch normalization), que estandariza las activaciones de cada canal dentro del minilote:

x̂ = (x − μ_B) / √(σ²_B + ε),    y = γ·x̂ + β

donde μ_B y σ²_B son la media y varianza del lote, ε evita la división por cero, y γ, β son parámetros aprendibles que restauran la capacidad expresiva. Esto estabiliza y acelera el entrenamiento al mantener las activaciones en un rango controlado, reduciendo la sensibilidad a la inicialización.

Tras varias etapas de convolución + ReLU + pooling, los mapas finales se aplanan (o se promedian con global average pooling) y pasan a una cabeza densa que produce los 10 logits, entrenados con entropía cruzada categórica y softmax igual que en el MLP. Una idea arquitectónica que el laboratorio deja como horizonte es la **conexión residual** (ResNet): sumar la entrada de un bloque a su salida, y = F(x) + x, lo que crea un atajo por el que el gradiente fluye sin atenuarse y permite entrenar redes de cientos de capas.

La formulación debe conectarse con cuatro elementos: representación de entrada, función del modelo, función de pérdida y regla de actualización. El notebook muestra las dimensiones de los tensores y conserva la misma implementación que el script de terminal.

## Protocolo científico

- Ajustar transformaciones, vocabulario, normalización y selección de variables solo con `train`.
- Usar `validation` para arquitectura, hiperparámetros, checkpoint y umbrales.
- Evaluar `test` una vez, después de congelar las decisiones.
- Comparar contra **Clasificador lineal sobre píxeles**.
- Reportar variación entre semillas e intervalos de confianza; una métrica puntual no expresa toda la incertidumbre.

## Riesgos de interpretación

CIFAR-10 contiene 60.000 imágenes reales de 32×32.

El dataset refleja su proceso de recolección y no representa automáticamente otros períodos, países o poblaciones. Una asociación predictiva no demuestra causalidad.

## Pregunta crítica

> ¿Qué clases concentran errores visualmente plausibles?

## 🔗 Referencias

> Las referencias apuntan a las obras; no se reproduce su contenido, la redacción es original.

- Goodfellow, Bengio & Courville — *Deep Learning* (MIT Press 2016), cap. 9 — redes convolucionales, peso compartido y pooling.
- Géron — *Hands-On Machine Learning* (3.ª ed., O'Reilly 2022), cap. 14 — CNN modernas y visión por computador con frameworks.
- Zhang et al. — *Dive into Deep Learning* (d2l.ai, 2023), cap. 7–8 — convoluciones, arquitecturas clásicas y batch normalization.
- LeCun et al. (1998), *Gradient-based learning applied to document recognition (LeNet)*, Proc. IEEE — primera CNN entrenada de extremo a extremo.
- Krizhevsky, Sutskever & Hinton (2012), *ImageNet Classification with Deep Convolutional Neural Networks (AlexNet)*, NeurIPS — hito que popularizó el aprendizaje profundo en visión.
- He et al. (2016), *Deep Residual Learning for Image Recognition (ResNet)*, CVPR — conexiones residuales para redes muy profundas.
- Fuente del dataset: https://www.cs.toronto.edu/~kriz/cifar.html
- Consulte `docs/experiment-protocol.md`, `docs/reproducibility.md` y `docs/ethics-and-licenses.md`.

<!-- nav-bottom -->
## 🧭 Navegación del recorrido

| ⬅️ Laboratorio anterior | 🏠 Índice | Laboratorio siguiente ➡️ |
|---|:---:|---|
| [🌀 MLP multiclase](../../labs/02_mlp_nonlinear/README.md) | [Las 31 rutas](../../README.md#laboratorios) | [🔁 RNN para texto](../../labs/04_rnn_sequences/README.md) |

**En este laboratorio:** [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

[🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/03_cnn_vision/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
