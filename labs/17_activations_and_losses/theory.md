# Teoría — Activaciones y funciones de pérdida

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
- Fuente del dataset: https://archive.ics.uci.edu/dataset/186/wine+quality
- Consulte `docs/experiment-protocol.md`, `docs/reproducibility.md` y `docs/ethics-and-licenses.md`.
