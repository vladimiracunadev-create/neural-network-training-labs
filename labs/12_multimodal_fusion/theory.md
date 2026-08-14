# Teoría — Fusión de sensores

<!-- nav-top -->
> 🧭 **Ruta 13 / 31** · [⬅️ ♻️ Transfer learning con mascotas](../../labs/11_transfer_learning/theory.md) · [🏠 Índice](../../README.md#laboratorios) · [🎛️ Búsqueda de hiperparámetros ➡️](../../labs/13_hyperparameter_search/theory.md)
>
> [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

## Propósito

Fusionar acelerómetro y giroscopio de smartphones para reconocer actividades.

## Idea central

Este laboratorio estudia **fusión de ramas de sensores** usando `uci_har`, un dataset público real procedente de UCI. La intuición es que cada sensor aporta una vista parcial y complementaria del mismo fenómeno físico: el acelerómetro captura la aceleración lineal (útil para distinguir estar de pie de caminar), mientras que el giroscopio mide la velocidad angular (útil para detectar giros y cambios de orientación). Ninguna modalidad basta por sí sola para separar todas las actividades, pero combinadas reducen la ambigüedad.

La **fusión tardía** (late fusion) que se practica aquí procesa cada modalidad con su propia rama (una red que aprende una representación específica del sensor) y luego concatena esas representaciones antes de la cabeza de clasificación. La alternativa de **fusión temprana** (early fusion) concatenaría las señales crudas desde el inicio. La fusión tardía suele ser más robusta cuando las modalidades tienen escalas, ruidos y estructuras temporales distintas, porque permite que cada rama normalice y abstraiga su señal antes de mezclarlas. El aprendizaje de estas representaciones intermedias es el mecanismo central del deep learning aplicado a datos heterogéneos.

El laboratorio también invita a comparar frente a líneas base de una sola modalidad y frente a un modelo lineal. Esto responde la pregunta de si la fusión realmente agrega valor o si una sola rama ya resuelve la tarea. Medir la ganancia marginal de cada sensor es tan importante como alcanzar buena exactitud global.

## Fundamento matemático

La entrada se divide en dos vistas de la misma ventana temporal: x_acc (canales del acelerómetro) y x_gyro (canales del giroscopio). Cada rama aplica una función parametrizada que produce un vector de características (embedding):

  h_acc = f_acc(x_acc; θ_acc),  h_gyro = f_gyro(x_gyro; θ_gyro)

La fusión concatena ambas representaciones y la cabeza produce los logits de las K actividades:

  f = [h_acc ; h_gyro],  z = head(f; θ_head),  ŷ = softmax(z)

donde la probabilidad de la clase k es ŷ_k = e^{z_k} / Σⱼ e^{z_j}. El entrenamiento minimiza la entropía cruzada sobre N ejemplos:

  ℒ(θ) = −(1/N) Σᵢ Σₖ y_{i,k} · log ŷ_{i,k}

con y_{i,k} la codificación one-hot de la etiqueta verdadera. La actualización de todos los parámetros θ = {θ_acc, θ_gyro, θ_head} sigue el descenso de gradiente estocástico:

  θ ← θ − η · ∇_θ ℒ

El gradiente ∇_θ ℒ se propaga por retropropagación a través de la cabeza y luego se **reparte** por las dos ramas. Aquí aparece la clave de la fusión: el error retrocede por ambos caminos simultáneamente, de modo que cada rama recibe una señal de aprendizaje condicionada por lo que la otra ya aporta. Por eso el modelo puede aprender a que el giroscopio se especialice en los patrones que el acelerómetro no discrimina bien. La formulación conecta cuatro elementos: representación de entrada (las dos vistas), función del modelo (ramas + cabeza), función de pérdida (entropía cruzada) y regla de actualización (SGD con ∇). El notebook muestra las dimensiones de los tensores y conserva la misma implementación que el script de terminal.

## Protocolo científico

- Ajustar transformaciones, vocabulario, normalización y selección de variables solo con `train`.
- Usar `validation` para arquitectura, hiperparámetros, checkpoint y umbrales.
- Evaluar `test` una vez, después de congelar las decisiones.
- Comparar contra **Acelerómetro solo, giroscopio solo y regresión logística**.
- Reportar variación entre semillas e intervalos de confianza; una métrica puntual no expresa toda la incertidumbre.

## Riesgos de interpretación

Señales inerciales reales de 30 participantes.

El dataset refleja su proceso de recolección y no representa automáticamente otros períodos, países o poblaciones. Una asociación predictiva no demuestra causalidad.

## Pregunta crítica

> ¿Qué modalidad explica cada actividad?

## 🔗 Referencias

> Las referencias apuntan a las obras; no se reproduce su contenido, la redacción es original.

- Goodfellow, Bengio & Courville — *Deep Learning* (MIT Press, 2016) — marco general sobre aprendizaje de representaciones y por qué las capas intermedias abstraen mejor los datos heterogéneos.
- Baltrušaitis, Ahuja & Morency (2019), *Multimodal Machine Learning: A Survey and Taxonomy*, IEEE TPAMI — taxonomía de estrategias de fusión (temprana, tardía, híbrida) y de los desafíos de alinear modalidades.
- Radford et al. (2021), *Learning Transferable Visual Models from Natural Language Supervision (CLIP)*, ICML — ejemplo influyente de aprender un espacio compartido entre modalidades distintas.
- Fuente del dataset: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
- Consulte `docs/experiment-protocol.md`, `docs/reproducibility.md` y `docs/ethics-and-licenses.md`.

<!-- nav-bottom -->
## 🧭 Navegación del recorrido

| ⬅️ Laboratorio anterior | 🏠 Índice | Laboratorio siguiente ➡️ |
|---|:---:|---|
| [♻️ Transfer learning con mascotas](../../labs/11_transfer_learning/README.md) | [Las 31 rutas](../../README.md#laboratorios) | [🎛️ Búsqueda de hiperparámetros](../../labs/13_hyperparameter_search/README.md) |

**En este laboratorio:** [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

[🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/12_multimodal_fusion/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
