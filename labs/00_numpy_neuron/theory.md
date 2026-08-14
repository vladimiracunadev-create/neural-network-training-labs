# Teoría — Neurona con NumPy

<!-- nav-top -->
> 🧭 **Ruta 1 / 31** · 🟢 [Parte 1 — Fundamentos: de la derivada a la primera red](../../parts/01-fundamentos.md)
>
> ⬅️ *inicio del recorrido* · [🏠 Índice de rutas](../../parts/README.md) · [🧩 Perceptrón con PyTorch ➡️](../../labs/01_pytorch_perceptron/theory.md)
>
> [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

## Propósito

Implementar propagación, entropía cruzada y descenso de gradiente sin autograd.

## Idea central

Este laboratorio estudia **regresión logística implementada sin autograd** usando `breast_cancer_wisconsin`, un dataset público real procedente de UCI.

La regresión logística es la unidad de construcción más simple del aprendizaje profundo: una sola neurona que combina linealmente sus entradas y las pasa por una no linealidad suave. Aquí no delegamos nada en un motor de diferenciación automática; escribimos a mano la propagación hacia adelante, la pérdida y las derivadas. El objetivo pedagógico es doble: entender de dónde salen los gradientes (no aparecen por magia) y comprobar que una neurona bien planteada resuelve un problema clínico real de diagnóstico binario (tumor benigno frente a maligno) a partir de 30 medidas morfológicas del núcleo celular.

Al forzar la derivación explícita, el laboratorio hace visible la cadena completa: cada peso wⱼ tiene una responsabilidad concreta sobre el error, y esa responsabilidad es exactamente lo que el gradiente cuantifica. Cuando en los laboratorios siguientes deleguemos esto en `autograd`, sabremos qué está calculando la máquina por debajo.

## Fundamento matemático

El modelo predice la probabilidad de que la clase sea positiva combinando las entradas de forma lineal y aplastando el resultado al intervalo (0, 1) con la función logística (sigmoide):

p(y=1 | x) = σ(z),  con  z = x·w + b = Σⱼ xⱼwⱼ + b,  y  σ(z) = 1 / (1 + e⁻ᶻ)

La sigmoide convierte una puntuación real ilimitada z en una probabilidad. Su forma en "S" comprime valores muy negativos hacia 0 y muy positivos hacia 1, dejando la mayor sensibilidad alrededor de z = 0, donde σ(0) = 0.5 marca la frontera de decisión. El sesgo b desplaza esa frontera y los pesos w orientan el hiperplano separador en el espacio de las 30 características.

Para ajustar los parámetros medimos el desacuerdo con la **entropía cruzada binaria** (equivalente a la log-verosimilitud negativa de un modelo Bernoulli). Para un conjunto de N ejemplos:

L = −(1/N) Σᵢ [ yᵢ·ln(pᵢ) + (1 − yᵢ)·ln(1 − pᵢ) ]

Esta pérdida penaliza con fuerza creciente la confianza equivocada: si el modelo asigna pᵢ ≈ 0 a un caso realmente positivo, ln(pᵢ) → −∞. Elegir entropía cruzada en lugar del error cuadrático no es arbitrario: al combinarla con la sigmoide, el gradiente se simplifica de forma notable y evita las mesetas de aprendizaje que produciría σ′(z) elevada al cuadrado.

El resultado clave, que este laboratorio deriva a mano, es que el gradiente de la pérdida respecto a los parámetros depende solo del **error de predicción** (pᵢ − yᵢ):

∂L/∂wⱼ = (1/N) Σᵢ (pᵢ − yᵢ)·xᵢⱼ    y    ∂L/∂b = (1/N) Σᵢ (pᵢ − yᵢ)

La intuición es transparente: si el modelo predice de más (pᵢ > yᵢ), el gradiente empuja los pesos en dirección contraria a las entradas activas; si predice de menos, los empuja a favor. La magnitud del ajuste es proporcional tanto al error como al valor de la característica, por eso la **escala de las variables importa**: una variable con valores muy grandes domina el gradiente y desestabiliza la convergencia si no se normaliza.

Finalmente, el descenso de gradiente actualiza los parámetros iterativamente con una tasa de aprendizaje η:

w ← w − η·∇_w L    ;    b ← b − η·∂L/∂b

La formulación debe conectarse con cuatro elementos: representación de entrada, función del modelo, función de pérdida y regla de actualización. El notebook muestra las dimensiones de los tensores y conserva la misma implementación que el script de terminal.

### La derivación completa, sin saltarse pasos

El resultado ∂L/∂wⱼ = (1/N)·Σᵢ (pᵢ − yᵢ)·xᵢⱼ es tan limpio que parece una coincidencia. No lo es, y verlo salir paso a paso es el objetivo de este laboratorio. Se aplica la regla de la cadena en tres tramos, ∂L/∂w = (∂L/∂p)·(∂p/∂z)·(∂z/∂w).

Primero, la derivada de la pérdida respecto de la probabilidad predicha. Derivando L = −[y·ln p + (1 − y)·ln(1 − p)]:

∂L/∂p = −y/p + (1 − y)/(1 − p) = (p − y) / (p·(1 − p)).

Segundo, la derivada de la sigmoide, que tiene una forma notable: como σ(z) = 1/(1 + e^(−z)),

σ′(z) = e^(−z) / (1 + e^(−z))² = σ(z)·(1 − σ(z)) = p·(1 − p).

Tercero, la parte lineal: ∂z/∂wⱼ = xⱼ.

Al multiplicar los tres, el factor p·(1 − p) del denominador de la primera derivada **se cancela exactamente** con el mismo factor que aporta σ′(z):

∂L/∂wⱼ = [ (p − y) / (p·(1 − p)) ] · [ p·(1 − p) ] · xⱼ = (p − y)·xⱼ.

Esa cancelación es la razón de fondo por la que la entropía cruzada es la pérdida correcta para la sigmoide, y no una preferencia estética. Si se usara error cuadrático, L = ½(p − y)², la derivada sería ∂L/∂z = (p − y)·σ′(z) = (p − y)·p·(1 − p), y el factor p·(1 − p) **sobreviviría**. Ese factor vale como máximo 0,25 en z = 0 y tiende a cero cuando el modelo está muy seguro: exactamente en los casos donde el modelo se equivoca con confianza —p ≈ 0 con y = 1— el gradiente se anularía y el aprendizaje se detendría justo donde más falta hace. Es el fenómeno de **saturación**, y la entropía cruzada lo evita por construcción.

### Por qué aquí hay una única solución, y después no

Esta pérdida tiene una propiedad que ningún laboratorio posterior volverá a tener: es **convexa** en los parámetros. Su matriz hessiana es

H = (1/N)·Xᵀ·S·X,   con S = diag(pᵢ·(1 − pᵢ)),

y como cada pᵢ·(1 − pᵢ) > 0, la matriz S es definida positiva y H resulta semidefinida positiva para cualquier X. Una función convexa no tiene mínimos locales distintos del global: cualquier punto donde el gradiente se anule es la solución óptima. Por eso aquí el descenso de gradiente converge al mismo sitio venga de donde venga la inicialización, y la única semilla que importa es la de la partición de datos.

Conviene guardar esa observación, porque explica un contraste que se vuelve central a partir de la ruta 02: en cuanto se añade una capa oculta con no linealidad, la superficie de pérdida deja de ser convexa, aparecen múltiples mínimos y puntos de silla, y **la inicialización empieza a cambiar el resultado**. Ese es el momento exacto en que `training_seed` se convierte en una variable experimental que hay que controlar y reportar, y no en un detalle.

Un caso límite conviene conocerlo: si las clases son **linealmente separables**, la verosimilitud no tiene máximo finito —los pesos crecen sin cota empujando las probabilidades hacia 0 y 1— y el entrenamiento diverge lentamente. La regularización L2 lo resuelve añadiendo (λ/2)·‖w‖², que vuelve la pérdida estrictamente convexa y garantiza un óptimo finito.

### Estabilidad numérica y comprobación del gradiente

Implementar estas fórmulas en punto flotante exige dos cuidados que el laboratorio hace visibles.

El primero es que ln(0) es −∞. Con z ≈ −40, σ(z) se redondea a 0,0 en float64 y la pérdida se vuelve infinita o NaN. La solución robusta no es recortar p a [ε, 1−ε], que sesga el resultado, sino no calcular σ por separado: se usa la forma estable

L = mean( max(z, 0) − z·y + ln(1 + e^(−|z|)) ),

algebraicamente idéntica a la entropía cruzada pero cuyo exponente nunca es positivo, de modo que e^(−|z|) ∈ (0, 1] y no desborda. Es exactamente lo que hace `BCEWithLogitsLoss` en la ruta siguiente, y aquí se escribe a mano para saber qué hay dentro.

El segundo es cómo saber que la derivada está bien programada. La comprobación estándar es contrastarla contra una **diferencia finita central**:

∂L/∂θ ≈ ( L(θ + ε) − L(θ − ε) ) / (2ε),

cuyo error es O(ε²) frente al O(ε) de la diferencia hacia adelante. Con ε ≈ 10⁻⁵ en float64, el error relativo entre el gradiente analítico y el numérico debería quedar por debajo de 10⁻⁷; por encima de 10⁻⁴ hay un fallo real en la derivación. Esta técnica es la que la ruta 16 aplica capa por capa a una red completa.

Sobre la escala de las variables: como el gradiente es proporcional a xᵢⱼ, una característica medida en miles produce gradientes miles de veces mayores que una medida en unidades. Con una tasa de aprendizaje única, la dirección de descenso queda dominada por la variable de mayor escala y el resto avanza a paso de tortuga. En términos de la hessiana, la relación entre su mayor y su menor autovalor —el **número de condición**— se dispara, y la convergencia del descenso de gradiente se degrada en la misma proporción. Estandarizar las 30 características de este dataset no es cosmética: es lo que hace que el problema sea resoluble en un número razonable de épocas. Y se ajusta **solo con `train`**, porque usar la media y la desviación del conjunto completo filtraría información de `test` al preprocesamiento.

## Protocolo científico

- Ajustar transformaciones, vocabulario, normalización y selección de variables solo con `train`.
- Usar `validation` para arquitectura, hiperparámetros, checkpoint y umbrales.
- Evaluar `test` una vez, después de congelar las decisiones.
- Comparar contra **DummyClassifier y regresión logística de scikit-learn**.
- Reportar variación entre semillas e intervalos de confianza; una métrica puntual no expresa toda la incertidumbre.

## Riesgos de interpretación

Datos clínicos reales derivados de imágenes digitalizadas de aspirados de masas mamarias.

El dataset refleja su proceso de recolección y no representa automáticamente otros períodos, países o poblaciones. Una asociación predictiva no demuestra causalidad.

## Pregunta crítica

> ¿Cómo cambia la convergencia al modificar la escala de las variables?

## 🔗 Referencias

> Las referencias apuntan a las obras; no se reproduce su contenido, la redacción es original.

- Bishop — *Pattern Recognition and Machine Learning* (1.ª ed., Springer 2006), cap. 4 (modelos lineales para clasificación) — deriva la regresión logística y su verosimilitud.
- Goodfellow, Bengio & Courville — *Deep Learning* (MIT Press 2016), cap. 5–6 — fundamentos de aprendizaje y redes hacia adelante, entropía cruzada y gradientes.
- Géron — *Hands-On Machine Learning* (3.ª ed., O'Reilly 2022), cap. 4 y 10 — regresión logística práctica y la neurona como base de las redes.
- Nielsen — *Neural Networks and Deep Learning* (online, 2015), cap. 1–2 — intuición de la neurona sigmoide y la retropropagación derivada a mano.
- Rosenblatt (1958), *The perceptron: a probabilistic model for information storage and organization in the brain*, Psychological Review — origen histórico de la neurona artificial entrenable.
- Fuente del dataset: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- Consulte `docs/experiment-protocol.md`, `docs/reproducibility.md` y `docs/ethics-and-licenses.md`.

<!-- nav-bottom -->
## 🧭 Navegación del recorrido

| ⬅️ Laboratorio anterior | 🏠 Índice | Laboratorio siguiente ➡️ |
|---|:---:|---|
| *— inicio del recorrido* | [Las 31 rutas](../../parts/README.md) | [🧩 Perceptrón con PyTorch](../../labs/01_pytorch_perceptron/README.md) |

**En este laboratorio:** [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🟢 [Parte 1 — Fundamentos: de la derivada a la primera red](../../parts/01-fundamentos.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/00_numpy_neuron/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
