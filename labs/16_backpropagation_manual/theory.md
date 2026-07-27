# Teoría — Backpropagation manual

## Propósito

Derivar y programar backpropagation en una MLP pequeña.

## Idea central

Este laboratorio estudia **backpropagation manual** usando `iris`, un dataset público real procedente de UCI. El objetivo es abrir la caja negra: en lugar de llamar a `loss.backward()` y confiar en el autodiferenciador, se derivan a mano los gradientes de una perceptrón multicapa (MLP) de dos capas y se programan paso a paso. Entender este mecanismo es entender *cómo aprenden* de verdad las redes neuronales.

La retropropagación no es más que la **regla de la cadena** del cálculo aplicada con orden. Una red es una composición de funciones: entrada → capa 1 → activación → capa 2 → softmax → pérdida. Para saber cómo cambiar cada peso y reducir la pérdida, necesitamos la derivada de la pérdida respecto a ese peso. La regla de la cadena nos dice que esa derivada es un producto de derivadas locales encadenadas desde la salida hacia atrás. La idea brillante de backprop es reutilizar cálculos: se calcula una vez el "error" en cada capa y se propaga hacia la capa anterior, evitando recomputar el mismo camino muchas veces.

El flujo tiene dos fases. En el **paso hacia adelante** (forward) se calculan y se guardan las activaciones de cada capa. En el **paso hacia atrás** (backward) se parte del error en la salida y se lo empuja capa por capa hacia la entrada, acumulando en el camino los gradientes de pesos y sesgos. La pregunta crítica del laboratorio —dónde aparecen gradientes que explotan o desaparecen— se vuelve tangible al ver cómo cada capa multiplica el gradiente por factores que pueden encogerlo o amplificarlo.

## Fundamento matemático

Consideremos una MLP con una capa oculta. Con entrada x, la propagación hacia adelante es:

  z₁ = W₁ x + b₁,  a₁ = σ(z₁),  z₂ = W₂ a₁ + b₂,  ŷ = softmax(z₂)

y la pérdida de entropía cruzada para la etiqueta one-hot y es ℒ = −Σₖ yₖ · log ŷₖ.

La retropropagación calcula ∂ℒ/∂W₂, ∂ℒ/∂b₂, ∂ℒ/∂W₁ y ∂ℒ/∂b₁ aplicando la regla de la cadena desde la salida. Definimos el **error de la capa de salida**; con softmax + entropía cruzada este error se simplifica de forma notable:

  δ₂ = ∂ℒ/∂z₂ = ŷ − y

De ahí bajan directamente los gradientes de la segunda capa:

  ∂ℒ/∂W₂ = δ₂ · a₁ᵀ,   ∂ℒ/∂b₂ = δ₂

El error se propaga a la capa oculta multiplicando por la matriz de pesos transpuesta y por la derivada de la activación, usando el producto de Hadamard ⊙ (elemento a elemento):

  δ₁ = (W₂ᵀ δ₂) ⊙ σ′(z₁)

  ∂ℒ/∂W₁ = δ₁ · xᵀ,   ∂ℒ/∂b₁ = δ₁

Finalmente, todos los parámetros se actualizan con descenso de gradiente: W ← W − η · ∂ℒ/∂W y b ← b − η · ∂ℒ/∂b, con η la tasa de aprendizaje.

Aquí se ven los **gradientes que se desvanecen o explotan**. El término δ₁ contiene el producto W₂ᵀ δ₂ ⊙ σ′(z₁): si σ es una sigmoide o tanh saturada, σ′(z₁) ≈ 0 y el gradiente se apaga (vanishing); si los pesos son grandes, los factores se acumulan y el gradiente crece sin control (exploding). En una red de L capas, este patrón se repite L veces, así que el gradiente en las capas iniciales es un producto de L factores y su magnitud depende críticamente de que esos factores ronden 1. Comprobar los gradientes analíticos contra una estimación numérica (∂ℒ/∂θ ≈ [ℒ(θ+ε) − ℒ(θ−ε)] / 2ε) es la prueba de que la derivación es correcta. La formulación conecta cuatro elementos: representación de entrada x, función del modelo (MLP), función de pérdida (entropía cruzada) y regla de actualización (SGD con ∇). El notebook muestra las dimensiones de los tensores y conserva la misma implementación que el script de terminal.

## Protocolo científico

- Ajustar transformaciones, vocabulario, normalización y selección de variables solo con `train`.
- Usar `validation` para arquitectura, hiperparámetros, checkpoint y umbrales.
- Evaluar `test` una vez, después de congelar las decisiones.
- Comparar contra **Regresión logística multinomial**.
- Reportar variación entre semillas e intervalos de confianza; una métrica puntual no expresa toda la incertidumbre.

## Riesgos de interpretación

150 mediciones botánicas reales de tres especies de Iris.

El dataset refleja su proceso de recolección y no representa automáticamente otros períodos, países o poblaciones. Una asociación predictiva no demuestra causalidad.

## Pregunta crítica

> ¿Dónde aparecen gradientes que explotan o desaparecen?

## 🔗 Referencias

> Las referencias apuntan a las obras; no se reproduce su contenido, la redacción es original.

- Goodfellow, Bengio & Courville — *Deep Learning* (MIT Press, 2016), cap. 6 — redes feedforward y el algoritmo de retropropagación como aplicación de la regla de la cadena.
- Nielsen — *Neural Networks and Deep Learning* (online, 2015), cap. 2 — derivación paso a paso de backpropagation con las cuatro ecuaciones fundamentales.
- Bishop — *Pattern Recognition and Machine Learning* (Springer, 2006), cap. 5 — redes neuronales, propagación de errores y verificación numérica de gradientes.
- Rumelhart, Hinton & Williams (1986), *Learning representations by back-propagating errors*, Nature — artículo que popularizó la retropropagación para entrenar redes multicapa.
- Baydin et al. (2018), *Automatic Differentiation in Machine Learning: a Survey*, JMLR — panorama de la diferenciación automática que generaliza el backprop manual de este laboratorio.
- Fuente del dataset: https://archive.ics.uci.edu/dataset/53/iris
- Consulte `docs/experiment-protocol.md`, `docs/reproducibility.md` y `docs/ethics-and-licenses.md`.
