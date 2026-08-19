# Teoría — Optimizadores y schedulers

<!-- nav-top -->
> 🧭 **Ruta 19 / 31** · 🔴 [Parte 5 — La mecánica fina, ahora en profundidad](../../parts/05-mecanica-fina.md)
>
> [⬅️ 📐 Activaciones y funciones de pérdida](../../labs/17_activations_and_losses/theory.md) · [🏠 Índice de rutas](../../parts/README.md) · [🛡️ Regularización ➡️](../../labs/19_regularization_dropout_batchnorm/theory.md)
>
> [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

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

### El momentum, leído como una media móvil

La actualización con momentum se escribe v ← β·v + ∇L, θ ← θ − η·v, y su efecto se entiende mejor desarrollando la recurrencia:

v_t = Σ_(k=0..t) β^k · ∇L_(t−k),

es decir, una **media móvil exponencial** de todos los gradientes pasados, con peso decreciente. La suma de los coeficientes tiende a 1/(1 − β), así que con β = 0,9 el paso efectivo es del orden de **diez veces** el de un gradiente aislado: por eso al activar momentum suele haber que reducir la tasa de aprendizaje.

Su utilidad se ve en un valle alargado, que es la forma típica de una superficie de pérdida mal condicionada. En las direcciones donde el gradiente oscila de signo, los términos se cancelan al promediarse; en la dirección donde el gradiente es consistente, se acumulan. El resultado es que el momentum **amortigua el zigzag y acelera el avance por el fondo del valle**, que es exactamente lo que el descenso de gradiente simple hace mal.

### Adam: por qué necesita corrección de sesgo

Adam mantiene dos medias móviles, la del gradiente y la de su cuadrado:

m_t = β₁·m_(t−1) + (1 − β₁)·g_t,   v_t = β₂·v_(t−1) + (1 − β₂)·g_t².

Ambas se inicializan en cero, y ahí está el problema. En el primer paso, m₁ = (1 − β₁)·g₁ = 0,1·g₁ con β₁ = 0,9: la estimación vale **una décima** del valor real. Con β₂ = 0,999 la distorsión de v es aún peor, un factor 0,001. Tomando esperanza se comprueba que 𝔼[m_t] = (1 − β₁ᵗ)·𝔼[g], de donde sale la corrección exacta:

m̂_t = m_t / (1 − β₁ᵗ),   v̂_t = v_t / (1 − β₂ᵗ),   θ ← θ − η · m̂_t / (√v̂_t + ε).

El divisor tiende a 1 conforme t crece, así que la corrección solo actúa en las primeras iteraciones —justo donde el sesgo es grande—. Sin ella, los primeros pasos serían minúsculos y el entrenamiento arrancaría con retraso.

La división por √v̂ es lo que hace de Adam un método **adaptativo por parámetro**: cada peso recibe un paso inversamente proporcional a la magnitud típica de su gradiente. Los parámetros con gradientes grandes avanzan poco a poco; los de gradientes pequeños —capas iniciales, características raras— avanzan más. Esa normalización es lo que le da robustez frente a la elección de η y lo que explica su popularidad. El precio es que la varianza de los primeros pasos, cuando v̂ se ha estimado con pocas muestras, puede ser alta: es la motivación del **calentamiento** que se describe abajo.

### AdamW: el weight decay que Adam rompía

Regularizar con L2 y aplicar weight decay son la misma cosa en SGD, y no lo son en Adam. Merece verse porque el error estuvo presente en implementaciones muy usadas durante años.

En SGD, añadir (λ/2)·‖θ‖² a la pérdida aporta un término λ·θ al gradiente, y la actualización queda θ ← θ − η·(g + λ·θ) = (1 − η·λ)·θ − η·g: un encogimiento proporcional al propio peso. En Adam, ese mismo término λ·θ entra en g **antes** de dividirse por √v̂, así que el encogimiento efectivo de cada parámetro acaba siendo λ·θ/√v̂: los pesos con gradientes históricamente grandes se regularizan **menos** que los de gradientes pequeños. La regularización deja de ser uniforme y pasa a depender de la historia del gradiente, que no es lo que nadie quería.

**AdamW** lo corrige desacoplando: aplica el decaimiento fuera del mecanismo adaptativo,

θ ← θ − η · m̂/(√v̂ + ε) − η·λ·θ,

restituyendo un encogimiento uniforme. La diferencia se nota sobre todo en generalización, y es la razón de que AdamW sea hoy el estándar en visión y en transformers.

### Schedulers: bajar la tasa y por qué calentar

Una tasa fija es un compromiso permanente: alta para avanzar rápido al principio, baja para afinar al final, y no puede ser ambas. De ahí los **schedulers**.

El **recocido coseno** es el más usado y su forma es explícita:

η_t = η_min + ½·(η_max − η_min)·(1 + cos(π·t/T)),

que baja suavemente de η_max a η_min a lo largo de T pasos. Frente a la reducción escalonada, evita los saltos bruscos que desestabilizan el entrenamiento justo después de cada bajada. El **`ReduceLROnPlateau`** sigue otra lógica: en vez de un calendario fijo, reduce la tasa cuando la métrica de validación deja de mejorar, adaptándose al problema a costa de introducir una dependencia de la señal de validación en el propio entrenamiento.

El **calentamiento** hace lo contrario al principio: sube la tasa linealmente desde casi cero durante las primeras iteraciones. Su justificación es la de arriba —con pocos pasos acumulados, las estimaciones de m̂ y v̂ son ruidosas y un paso grande basado en ellas puede desplazar los pesos a una región mala— y se vuelve casi obligatorio con lotes grandes y en transformers.

Una precisión sobre el laboratorio: como aquí la tarea es de **regresión** y se decide con `rmse`, conviene recordar qué implica esa elección. El RMSE, al elevar al cuadrado, penaliza desproporcionadamente los errores grandes y es sensible a valores atípicos; el MAE los trata linealmente. Optimizar error cuadrático y reportar RMSE es coherente, pero significa que el modelo dedicará capacidad a no equivocarse mucho en pocos casos extremos antes que a acertar un poco mejor en la mayoría. Si eso no es lo que el problema pide, la pérdida —y no solo el optimizador— es lo que hay que cambiar.

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
- Fuente del dataset: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html — **California Housing (censo de 1990)** (StatLib (Carnegie Mellon University), StatLib no declara una licencia formal); procedencia, versión y SHA-256 en el registro de fuentes, entrada `california-housing-statlib` — esta clase la usa para comparar SGD, Momentum, Adam y la reducción de la tasa de aprendizaje en una regresión sobre datos censales reales.
- Consulte `docs/experiment-protocol.md`, `docs/reproducibility.md` y `docs/ethics-and-licenses.md`.

<!-- nav-bottom -->
## 🧭 Navegación del recorrido

| ⬅️ Laboratorio anterior | 🏠 Índice | Laboratorio siguiente ➡️ |
|---|:---:|---|
| [📐 Activaciones y funciones de pérdida](../../labs/17_activations_and_losses/README.md) | [Las 31 rutas](../../parts/README.md) | [🛡️ Regularización](../../labs/19_regularization_dropout_batchnorm/README.md) |

**En este laboratorio:** [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🔴 [Parte 5 — La mecánica fina, ahora en profundidad](../../parts/05-mecanica-fina.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/18_optimizers_and_schedulers/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
