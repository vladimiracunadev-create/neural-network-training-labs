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
