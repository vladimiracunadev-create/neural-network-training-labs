# Teoría — Optimizadores y schedulers

## Propósito

Comparar SGD, Momentum, Adam y reducción de tasa de aprendizaje.

## Idea central

Este laboratorio estudia **comparación controlada de optimizadores y schedulers** usando `california_housing`, un dataset público real procedente de scikit-learn / StatLib.

Entrenar una red neuronal es resolver un problema de **optimización**: buscar los parámetros θ que minimizan una función de pérdida ℒ(θ) sobre datos reales. Como no podemos evaluar el gradiente exacto sobre todo el conjunto (sería costoso y redundante), estimamos ∇ℒ con **mini-lotes** aleatorios. El optimizador es la regla que traduce ese gradiente ruidoso en un paso de actualización, y el *scheduler* es la política que hace variar la tasa de aprendizaje η a lo largo del entrenamiento. Dos decisiones aparentemente técnicas —qué optimizador y qué programación de η— determinan si el modelo converge rápido, se estanca en una meseta o diverge.

La comparación es "controlada" porque solo cambiamos el optimizador/scheduler mientras mantenemos fijos arquitectura, partición, semillas y presupuesto de cómputo. Así, cualquier diferencia observada en velocidad de convergencia o generalización se atribuye a la regla de actualización y no a factores de confusión. Sobre `california_housing` —una regresión del valor mediano de vivienda a partir de variables socioeconómicas— medimos cuán rápido baja la pérdida de entrenamiento y cuán bien se comporta el modelo en validación.

## Fundamento matemático

Actualizaciones de parámetros y programación de learning rate.

El **descenso de gradiente estocástico (SGD)** actualiza cada parámetro moviéndose en dirección opuesta al gradiente del mini-lote: θ ← θ − η · ∇θ ℒ(θ). El signo negativo es la intuición central: −∇ℒ apunta hacia donde la pérdida decrece más rápido, y η controla la longitud del paso. Un η demasiado grande hace que el paso "sobrepase" el mínimo y oscile o diverja; uno demasiado pequeño convierte el entrenamiento en un avance lentísimo. Como el gradiente proviene de un mini-lote, es un estimador *ruidoso* del gradiente verdadero, y ese ruido es a la vez un obstáculo (trayectoria zigzagueante) y una ayuda (permite escapar de mínimos pobres).

El **momentum** acumula una media exponencial de los gradientes recientes para suavizar la trayectoria: vₜ ← β·vₜ₋₁ + (1−β)·∇θ ℒ, luego θ ← θ − η·vₜ. La velocidad v actúa como inercia física: en direcciones donde el gradiente es consistente, los pasos se suman y el avance se acelera; en direcciones donde oscila, las contribuciones se cancelan y el zigzag se amortigua. El factor β ≈ 0.9 fija cuánta "memoria" se conserva.

**Adam** combina momentum con una normalización por la magnitud reciente de cada gradiente. Mantiene dos medias exponenciales, la del gradiente (primer momento mₜ) y la de su cuadrado (segundo momento vₜ), aplica una corrección de sesgo m̂ₜ, v̂ₜ (necesaria porque ambas medias arrancan en cero) y actualiza θ ← θ − η · m̂ₜ / (√v̂ₜ + ε). Dividir por √v̂ₜ da a cada parámetro una tasa de aprendizaje *efectiva* propia: los parámetros con gradientes grandes reciben pasos más cortos y los de gradientes pequeños pasos más largos, lo que hace a Adam robusto a la escala y suele acelerar las primeras épocas. **AdamW** corrige un detalle sutil: desacopla el *weight decay* (λ·θ) de la actualización adaptativa, aplicándolo directamente sobre θ en vez de mezclarlo con el gradiente, lo que restaura la interpretación de regularización L2 que Adam distorsiona.

El **scheduler** hace evolucionar η con el tiempo. La motivación es que conviene un η grande al principio, para avanzar deprisa por regiones lejanas del mínimo, y un η pequeño al final, para asentarse con precisión sin oscilar. Programaciones típicas son el decaimiento por pasos (η se divide por un factor cada cierto número de épocas), el decaimiento coseno η(t) = η_min + ½(η_max − η_min)(1 + cos(π·t/T)), o la reducción en meseta cuando la métrica de validación deja de mejorar. La regla práctica: el optimizador decide *la dirección y forma* del paso; el scheduler decide *su tamaño a lo largo del tiempo*.

## Protocolo científico

- Ajustar transformaciones, vocabulario, normalización y selección de variables solo con `train`.
- Usar `validation` para arquitectura, hiperparámetros, checkpoint y umbrales.
- Evaluar `test` una vez, después de congelar las decisiones.
- Comparar contra **Media y Ridge**.
- Reportar variación entre semillas e intervalos de confianza; una métrica puntual no expresa toda la incertidumbre.

## Riesgos de interpretación

Datos reales del censo de California de 1990.

El dataset refleja su proceso de recolección y no representa automáticamente otros períodos, países o poblaciones. Una asociación predictiva no demuestra causalidad.

## Pregunta crítica

> ¿Cuál mejora más rápido y cuál generaliza mejor?

## 🔗 Referencias

> Las referencias apuntan a las obras; no se reproduce su contenido, la redacción es original.

- Goodfellow, Bengio y Courville — *Deep Learning* (MIT Press, 2016), cap. 8 — tratamiento formal de la optimización para entrenamiento de redes profundas: SGD, momentum y métodos adaptativos.
- Ruder (2016), *An overview of gradient descent optimization algorithms*, arXiv — panorámica comparada de SGD, momentum, Adagrad, RMSProp y Adam con intuición geométrica.
- Robbins y Monro (1951), *A Stochastic Approximation Method*, Annals of Mathematical Statistics — origen teórico del descenso estocástico y las condiciones de convergencia.
- Kingma y Ba (2015), *Adam: A Method for Stochastic Optimization*, ICLR — definición del optimizador Adam y su corrección de sesgo de momentos.
- Loshchilov y Hutter (2019), *Decoupled Weight Decay Regularization (AdamW)*, ICLR — desacoplamiento del weight decay respecto de la actualización adaptativa.
- Fuente del dataset: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html
- Consulte `docs/experiment-protocol.md`, `docs/reproducibility.md` y `docs/ethics-and-licenses.md`.
