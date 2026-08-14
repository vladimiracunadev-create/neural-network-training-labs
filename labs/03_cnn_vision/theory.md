# Teoría — CNN para visión

<!-- nav-top -->
> 🧭 **Ruta 4 / 31** · 🔵 [Parte 2 — Arquitecturas según la forma del dato](../../parts/02-arquitecturas.md)
>
> [⬅️ 🌀 MLP multiclase](../../labs/02_mlp_nonlinear/theory.md) · [🏠 Índice de rutas](../../parts/README.md) · [🔁 RNN para texto ➡️](../../labs/04_rnn_sequences/theory.md)
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

### Cuánto ahorra realmente el peso compartido

La frase «reduce drásticamente los parámetros» merece números, porque la magnitud sorprende. Una capa convolucional con C_in canales de entrada, C_out filtros y kernel F×F tiene

|θ|_conv = F²·C_in·C_out + C_out

parámetros, y —esto es lo decisivo— **ese número no depende del tamaño de la imagen**. Una primera capa con 32 filtros de 3×3 sobre las tres bandas de color de CIFAR-10 son 3²·3·32 + 32 = 896 pesos, y serían los mismos 896 sobre una imagen de 1024×1024.

La comparación con una capa densa equivalente es brutal. Aplanar una imagen de 32×32×3 da 3 072 entradas; conectarlas a 32 unidades cuesta 3 072·32 + 32 = 98 336 parámetros, más de cien veces más, y encima esa capa habría perdido toda noción de vecindad: para ella, dos píxeles contiguos y dos píxeles opuestos de la imagen son igual de ajenos.

El costo de cómputo sigue otra ley. Una capa convolucional realiza aproximadamente

FLOPs ≈ H_out · W_out · C_out · F² · C_in

multiplicaciones-acumulaciones, así que **sí** escala con el tamaño de la imagen aunque los parámetros no lo hagan. De ahí una asimetría que conviene tener presente al leer los resultados: en una CNN, la mayor parte de los *parámetros* suele estar en las capas densas finales, mientras que la mayor parte del *tiempo* se va en las capas convolucionales tempranas, que operan sobre mapas grandes. Optimizar el tamaño del modelo y optimizar su latencia no son la misma tarea.

### El campo receptivo: qué ve realmente cada neurona

Una neurona de la primera capa ve una ventana de 3×3 píxeles. La pregunta interesante es cuánto ve una neurona de la quinta capa, y la respuesta la da la recurrencia

R_ℓ = R_(ℓ−1) + (F_ℓ − 1) · Π_(i<ℓ) s_i,

donde F_ℓ es el tamaño de filtro de la capa ℓ y s_i los strides (incluido el del pooling) de las capas anteriores. El producto acumulado es la clave: **cada submuestreo duplica el efecto de todas las convoluciones posteriores**. Con capas 3×3 y un pooling de 2 intercalado cada dos convoluciones, el campo receptivo pasa de 3 a 7, luego a 15, luego a 31 píxeles: en cuatro bloques ya cubre la imagen entera de 32×32.

Esto explica la forma canónica de una CNN. Las primeras capas, con campo receptivo pequeño, solo pueden detectar bordes y colores; las profundas, con campo receptivo comparable a la imagen, pueden responder a objetos completos. Y explica también un error de diseño frecuente: si el campo receptivo final es mucho menor que el objeto que hay que reconocer, ninguna capa llega a «ver» el objeto entero, y añadir filtros no lo arregla —hay que añadir profundidad o submuestreo—.

Sobre el submuestreo hay dos opciones y conviene distinguirlas. El **max-pooling** no tiene parámetros y selecciona el máximo de cada región, quedándose con la evidencia más fuerte de que el patrón está presente e ignorando su posición exacta dentro de la ventana. La **convolución con stride 2** submuestrea aprendiendo cómo combinar la región en vez de imponer el máximo; cuesta parámetros y es la elección de las arquitecturas modernas. Y al final, el **global average pooling** promedia cada mapa completo a un único número, reduciendo un tensor de H×W×C a C valores: elimina de golpe casi todos los parámetros de la cabeza densa y hace la red independiente del tamaño de entrada.

### Normalización por lotes: dos modos y un error clásico

La fórmula de la normalización por lotes esconde una asimetría que causa uno de los fallos más difíciles de diagnosticar. Durante el **entrenamiento**, μ_B y σ²_B se calculan sobre el minilote actual, así que la salida de un ejemplo depende de con qué otros ejemplos comparta lote. Durante la **inferencia** eso sería inaceptable —la predicción no puede depender de quién más esté en el lote— y por eso la capa usa estadísticas acumuladas durante el entrenamiento mediante una media móvil,

μ̂ ← (1 − m)·μ̂ + m·μ_B,   σ̂² ← (1 − m)·σ̂² + m·σ²_B,

con un momento m típico de 0,1. Son parámetros del modelo que no se aprenden por gradiente: se estiman por acumulación.

De ahí el error clásico: evaluar sin poner el modelo en modo evaluación. Si se olvida, la capa sigue normalizando con el lote de test —y si además el lote de test es pequeño o está ordenado por clase, las estadísticas son malísimas— y las métricas salen peores sin que nada falle visiblemente. El mismo interruptor gobierna el dropout, que debe desactivarse en inferencia. Es la razón de que el protocolo de este repositorio evalúe siempre con el modelo congelado y en modo evaluación explícito.

Un efecto secundario que conviene conocer: como las estadísticas del lote introducen ruido en cada ejemplo, la normalización por lotes **regulariza**, y ese efecto se debilita con lotes grandes. Por eso a veces subir el tamaño de lote empeora la generalización aunque el entrenamiento sea más estable, y por eso existen alternativas —normalización de grupo o de capa— para escenarios con lotes muy pequeños.

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
| [🌀 MLP multiclase](../../labs/02_mlp_nonlinear/README.md) | [Las 31 rutas](../../parts/README.md) | [🔁 RNN para texto](../../labs/04_rnn_sequences/README.md) |

**En este laboratorio:** [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🔵 [Parte 2 — Arquitecturas según la forma del dato](../../parts/02-arquitecturas.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/03_cnn_vision/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
