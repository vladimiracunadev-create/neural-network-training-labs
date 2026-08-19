# Teoría — Activaciones y funciones de pérdida

<!-- nav-top -->
> 🧭 **Ruta 18 / 31** · 🔴 [Parte 5 — La mecánica fina, ahora en profundidad](../../parts/05-mecanica-fina.md)
>
> [⬅️ ∂ Backpropagation manual](../../labs/16_backpropagation_manual/theory.md) · [🏠 Índice de rutas](../../parts/README.md) · [⚙️ Optimizadores y schedulers ➡️](../../labs/18_optimizers_and_schedulers/theory.md)
>
> [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

## Propósito

Comparar ReLU, GELU, Tanh y pérdidas apropiadas en clases desbalanceadas.

## Idea central

Este laboratorio estudia **comparación controlada de activaciones y pérdidas** usando `wine_quality`, un dataset público real procedente de UCI. Dos decisiones de diseño gobiernan cómo aprende una red: qué **función de activación** introduce la no linealidad entre capas y qué **función de pérdida** define qué significa equivocarse. El laboratorio las aísla y las compara de forma controlada, cambiando una variable a la vez para atribuir con honestidad las diferencias de desempeño.

Sobre las **activaciones**: sin no linealidad, apilar capas lineales colapsa en una sola transformación lineal, incapaz de modelar fronteras complejas. Tanh satura en ambos extremos (su derivada tiende a 0 para entradas grandes), lo que frena el aprendizaje en redes profundas por gradientes que se desvanecen. ReLU evita esa saturación en el lado positivo manteniendo la derivada en 1, lo que acelera el entrenamiento y favorece representaciones dispersas, aunque puede "morir" si una neurona queda siempre en la zona negativa. GELU es una alternativa suave que pondera la entrada por su probabilidad bajo una gaussiana, combinando parte de la no saturación de ReLU con una transición diferenciable.

Sobre las **pérdidas**: el dataset de calidad de vino está desbalanceado (hay muchas más muestras de calidad media que de los extremos). La entropía cruzada estándar trata todos los ejemplos por igual y tiende a optimizar la clase mayoritaria, ignorando las minoritarias. La **Focal Loss** reescala la pérdida para bajar el peso de los ejemplos ya bien clasificados y concentrar el aprendizaje en los difíciles. La pregunta crítica —si la conclusión se mantiene en varias semillas— recuerda que en comparaciones finas la diferencia entre dos activaciones puede ser menor que el ruido de entrenamiento.

## Fundamento matemático

Una activación transforma cada preactivación z de forma no lineal. Sus definiciones y derivadas explican su comportamiento:

  Tanh:  σ(z) = (eᶻ − e⁻ᶻ)/(eᶻ + e⁻ᶻ),   σ′(z) = 1 − σ(z)²

  ReLU:  σ(z) = max(0, z),   σ′(z) = 1 si z > 0, 0 si z < 0

  GELU:  σ(z) = z · Φ(z),   con Φ la función de distribución acumulada de la normal estándar

La clave está en la derivada, porque es el factor por el que backpropagation multiplica el gradiente al atravesar la capa. Para Tanh, σ′(z) = 1 − σ(z)² tiende a 0 cuando |z| es grande: la neurona **satura** y el gradiente se desvanece. Para ReLU, σ′(z) = 1 en toda la región activa: el gradiente pasa sin atenuarse, lo que combate el desvanecimiento pero deja gradiente nulo (neuronas muertas) cuando z < 0. GELU suaviza esa transición, evitando el corte brusco en z = 0.

Para la salida de clasificación se usa softmax, ŷₖ = e^{zₖ} / Σⱼ e^{zⱼ}, y sobre él se define la pérdida. La **entropía cruzada** para la clase verdadera es:

  CE = −Σₖ yₖ · log ŷₖ

La **Focal Loss** añade un factor modulador (1 − p_t)^γ, donde p_t es la probabilidad asignada a la clase correcta y γ ≥ 0 controla cuánto se atenúan los ejemplos fáciles:

  FL = −α_t · (1 − p_t)^γ · log(p_t)

Cuando el modelo ya acierta con confianza, p_t → 1, el factor (1 − p_t)^γ → 0 y ese ejemplo casi no contribuye al gradiente; los ejemplos difíciles (p_t bajo) conservan casi toda su pérdida. Con γ = 0 la Focal Loss se reduce a la entropía cruzada ponderada. Por eso ayuda en clases desbalanceadas: reorienta la señal de aprendizaje ∇ hacia las clases minoritarias mal clasificadas en vez de reforzar la mayoría ya resuelta. Todo se optimiza con descenso de gradiente, θ ← θ − η · ∇_θ ℒ. La formulación conecta cuatro elementos: representación de entrada, función del modelo (con su activación), función de pérdida (CE o Focal) y regla de actualización (SGD con ∇). El notebook muestra las dimensiones de los tensores y conserva la misma implementación que el script de terminal.

### Las activaciones, comparadas por lo que le hacen al gradiente

Una activación se elige por cómo se comporta su **derivada**, no por la forma de su curva. Puestas una al lado de otra, las diferencias son concretas.

La **sigmoide** tiene derivada σ′(z) = σ(z)·(1 − σ(z)), cuyo máximo es 0,25 en z = 0. Encadenar L capas multiplica L de esos factores, así que el gradiente se atenúa como máximo por 0,25^L: con diez capas, un factor 10⁻⁶ en el mejor de los casos. Además su salida no está centrada en cero —siempre positiva—, lo que hace que todos los gradientes de una misma neurona compartan signo y la optimización avance en zigzag. La **tanh** corrige lo segundo: está centrada en cero y su derivada llega a 1 en el origen, por lo que atenúa menos, pero sigue saturando en ambos extremos.

La **ReLU** cambia el juego porque su derivada es exactamente 1 en toda la región positiva: no atenúa. A cambio es exactamente 0 en la negativa, y de ahí el fallo de la neurona muerta —si la preactivación queda negativa para todos los ejemplos, no vuelve a recibir gradiente nunca—. La **Leaky ReLU** deja pasar una pendiente pequeña α ≈ 0,01 en la zona negativa precisamente para que ese gradiente nunca sea nulo.

La **GELU** sustituye el corte duro por una compuerta suave: GELU(z) = z·Φ(z), donde Φ es la función de distribución acumulada de la normal estándar. En vez de decidir con un umbral, pondera la entrada por la probabilidad de que sea mayor que una variable normal. Su derivada es continua en todo punto —a diferencia de la ReLU, discontinua en 0— y no se anula en la zona negativa cercana al origen, lo que le da gradiente donde la ReLU ya no lo tiene. Es la activación por defecto en los transformers, y en este laboratorio se compara con las anteriores manteniendo todo lo demás fijo.

Una observación que ordena la comparación: lo que se está eligiendo no es «la mejor función», sino **el perfil de gradiente** que la red recibirá en cada capa. Por eso el efecto de la activación depende de la profundidad, y una comparación hecha con una sola capa oculta puede no extrapolarse a una red profunda.

### Qué pérdida usar cuando las clases están desbalanceadas

La segunda mitad del laboratorio trata la otra elección, y aquí el problema es que la entropía cruzada **trata todos los ejemplos por igual** en una situación en que no lo son.

Con una clase mayoritaria que domina el conjunto, la mayor parte del gradiente proviene de ejemplos fáciles y ya bien clasificados de esa clase. El modelo aprende rápido a predecir la mayoría y se estanca en la minoría, no por falta de capacidad sino porque la señal de la minoría queda ahogada. Hay tres respuestas, y conviene entender qué modifica cada una.

Los **pesos por clase** multiplican la contribución de cada ejemplo por un factor w_c inverso a la frecuencia:

ℒ = −(1/N) · Σᵢ w_(yᵢ) · log p_(i,yᵢ).

Es simple y directo, pero trata por igual a todos los ejemplos de la clase minoritaria, incluidos los que ya se clasifican perfectamente.

La **Focal Loss** cambia el eje: en vez de ponderar por clase, pondera por **dificultad**.

ℒ_focal = −(1 − p_t)^γ · log p_t,

donde p_t es la probabilidad asignada a la clase verdadera. El factor (1 − p_t)^γ vale casi 0 cuando el ejemplo ya está bien clasificado —p_t ≈ 1— y casi 1 cuando está mal. Con γ = 2, un ejemplo con p_t = 0,9 ve su pérdida reducida cien veces, mientras que uno con p_t = 0,1 apenas se toca. El efecto es que el gradiente se concentra en lo que el modelo aún no domina, sin necesidad de conocer las frecuencias de clase.

El **remuestreo** actúa antes de la pérdida, replicando ejemplos de la minoría o descartando de la mayoría. Cambia la distribución que ve el modelo, lo que tiene una consecuencia que se olvida a menudo: las probabilidades que produzca quedarán **descalibradas** respecto de la distribución real, y habrá que corregirlas si se van a interpretar como probabilidades. Es justo el problema que aborda la ruta 22.

Y la elección de la pérdida arrastra la elección de la métrica. Optimizar con pesos de clase y reportar exactitud es incoherente: se está pidiendo al modelo que priorice la minoría y midiéndolo con una cifra que premia la mayoría. Por eso este laboratorio decide con `macro_f1`, que promedia por clase y da a la minoritaria el mismo peso que a la mayoritaria.

## Protocolo científico

- Ajustar transformaciones, vocabulario, normalización y selección de variables solo con `train`.
- Usar `validation` para arquitectura, hiperparámetros, checkpoint y umbrales.
- Evaluar `test` una vez, después de congelar las decisiones.
- Comparar contra **Regresión ordinal y Random Forest**.
- Reportar variación entre semillas e intervalos de confianza; una métrica puntual no expresa toda la incertidumbre.

## Riesgos de interpretación

Muestras reales de vinho verde con análisis fisicoquímico y evaluación sensorial.

El dataset refleja su proceso de recolección y no representa automáticamente otros períodos, países o poblaciones. Una asociación predictiva no demuestra causalidad.

## Pregunta crítica

> ¿La conclusión se mantiene en varias semillas?

## 🔗 Referencias

> Las referencias apuntan a las obras; no se reproduce su contenido, la redacción es original.

- Goodfellow, Bengio & Courville — *Deep Learning* (MIT Press, 2016), cap. 6 — unidades de activación, no linealidades y funciones de salida con sus pérdidas asociadas.
- Géron — *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* (3.ª ed., O'Reilly 2022), cap. 10–11 — activaciones en la práctica y entrenamiento de redes profundas.
- Bishop — *Pattern Recognition and Machine Learning* (Springer, 2006), cap. 5 — funciones de error y su relación con la interpretación probabilística de la salida.
- Nair & Hinton (2010), *Rectified Linear Units Improve Restricted Boltzmann Machines (ReLU)*, ICML — introducción de la unidad ReLU.
- Glorot, Bordes & Bengio (2011), *Deep Sparse Rectifier Neural Networks*, AISTATS — evidencia de que los rectificadores facilitan el entrenamiento de redes profundas.
- Fuente del dataset: https://archive.ics.uci.edu/dataset/186/wine+quality — **Wine Quality** (UCI Machine Learning Repository, CC BY 4.0); procedencia, versión y SHA-256 en el registro de fuentes, entrada `uci-wine-quality` — esta clase la usa para comparar ReLU, GELU y Tanh, y las pérdidas apropiadas, sobre clases sensoriales desbalanceadas.
- Consulte `docs/experiment-protocol.md`, `docs/reproducibility.md` y `docs/ethics-and-licenses.md`.

<!-- nav-bottom -->
## 🧭 Navegación del recorrido

| ⬅️ Laboratorio anterior | 🏠 Índice | Laboratorio siguiente ➡️ |
|---|:---:|---|
| [∂ Backpropagation manual](../../labs/16_backpropagation_manual/README.md) | [Las 31 rutas](../../parts/README.md) | [⚙️ Optimizadores y schedulers](../../labs/18_optimizers_and_schedulers/README.md) |

**En este laboratorio:** [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🔴 [Parte 5 — La mecánica fina, ahora en profundidad](../../parts/05-mecanica-fina.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/17_activations_and_losses/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
