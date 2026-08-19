# Teoría — Transfer learning con mascotas

<!-- nav-top -->
> 🧭 **Ruta 12 / 31** · 🟣 [Parte 3 — Familias especializadas: generar, decidir, relacionar](../../parts/03-familias-especializadas.md)
>
> [⬅️ 🕹️ DQN para inventario con demanda real](../../labs/10_dqn_reinforcement/theory.md) · [🏠 Índice de rutas](../../parts/README.md) · [🔀 Fusión de sensores ➡️](../../labs/12_multimodal_fusion/theory.md)
>
> [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

## Propósito

Comparar extracción de características, fine-tuning y entrenamiento desde cero.

## Idea central

Este laboratorio estudia **reutilización de representaciones preentrenadas** usando `oxford_iiit_pet`, un dataset público real procedente de Torchvision / Oxford.

La idea del **aprendizaje por transferencia** es que una red entrenada sobre un problema grande (clasificar millones de imágenes de ImageNet en 1000 categorías) aprende representaciones visuales genéricas —bordes, texturas, formas, partes de objetos— que sirven para *otras* tareas de visión. En vez de partir de pesos aleatorios y aprender todo desde cero con nuestras 7.349 imágenes de 37 razas de perros y gatos (un dataset pequeño), reutilizamos esa red preentrenada como punto de partida. La intuición es que las primeras capas de una CNN capturan características de bajo nivel casi universales, y solo las últimas capas son específicas de la tarea original; por eso conviene conservar lo genérico y readaptar lo específico.

El laboratorio contrasta tres estrategias sobre el mismo backbone. En la **extracción de características** se congela toda la red preentrenada y se entrena únicamente una nueva cabeza de clasificación: la red actúa como un extractor fijo de embeddings. En el **fine-tuning** se descongelan algunas (o todas) las capas y se las reajusta con una tasa de aprendizaje pequeña, adaptando las representaciones a las mascotas. Y el **entrenamiento desde cero** parte de pesos aleatorios como línea base. Comparar los tres responde una pregunta práctica central: ¿cuándo aporta el preentrenamiento y cuándo deja de compensar?

## Fundamento matemático

Sea un modelo dividido en un cuerpo de extracción de características f(·; θ_f) y una cabeza de clasificación h(·; θ_h), de modo que la predicción es ŷ = h(f(x; θ_f); θ_h). En un entrenamiento desde cero, θ_f y θ_h se inicializan al azar y se optimizan ambos minimizando la entropía cruzada ℒ = −Σ_c y_c log ŷ_c mediante θ ← θ − η ∇_θ ℒ. La transferencia cambia la **inicialización** y qué parámetros reciben gradiente: se parte de θ_f = θ_f^{ImageNet}, pesos ya optimizados sobre un problema masivo, en lugar de ruido aleatorio. Conectando con los cuatro elementos: la **representación de entrada** es la imagen normalizada con las mismas estadísticas de ImageNet; la **función del modelo** es h∘f con backbone preentrenado; la **función de pérdida** es la entropía cruzada sobre las 37 clases; y la **regla de actualización** es descenso de gradiente, aplicado a distintos subconjuntos de parámetros según la estrategia. El notebook muestra las dimensiones de los tensores y conserva la misma implementación que el script de terminal.

La diferencia operativa entre estrategias es qué gradientes se propagan. En **extracción de características** se fija ∇_{θ_f} ℒ = 0 (capas congeladas) y solo se actualiza θ_h; esto es rápido, usa poca memoria y resiste el sobreajuste cuando hay pocos datos, porque hay muchos menos parámetros que ajustar. En **fine-tuning** se permite ∇_{θ_f} ℒ ≠ 0 en las capas descongeladas, pero con una tasa η pequeña: la razón es que los pesos preentrenados ya están cerca de una buena solución, y un η grande los destruiría (fenómeno de *olvido catastrófico*) antes de que la cabeza —inicialmente aleatoria— produzca gradientes sensatos. Una práctica común es entrenar primero solo la cabeza y luego descongelar gradualmente el cuerpo con learning rate reducido.

¿Por qué transfieren las características? Yosinski et al. mostraron empíricamente que la transferibilidad **decae con la profundidad**: las capas iniciales aprenden filtros genéricos (parecidos a detectores de bordes y color, casi idénticos entre tareas) mientras que las capas finales se especializan en las clases de la tarea original. Cuanto más se parezcan la tarea origen y la destino, más capas conviene reutilizar; cuanto más difieran, más capas hay que readaptar. Formalmente, esto encaja en el marco de Pan & Yang: transferimos conocimiento de un dominio origen 𝒟_S (ImageNet) a un dominio destino 𝒟_T (mascotas) que comparten el espacio de características pero difieren en la distribución de entrada y en la tarea. El beneficio del preentrenamiento es mayor cuanto menor es el dataset destino y mayor la afinidad entre dominios; con suficientes datos destino, entrenar desde cero puede alcanzar —o superar— a la transferencia, y ahí es donde el preentrenamiento deja de aportar.

### Los tres regímenes, en parámetros y en gradiente

Las tres estrategias que compara el laboratorio se distinguen por qué subconjunto de θ recibe gradiente, y esa diferencia se puede contar.

En **extracción de características** se congela el cuerpo entero: `requires_grad = False` para todos sus pesos, y solo se entrena la cabeza. Con una ResNet18 —unos 11,2 millones de parámetros en el cuerpo— y una cabeza lineal de 512 entradas a las clases del problema, los parámetros entrenables bajan a unos pocos miles. Como el cuerpo no cambia, sus salidas para cada imagen son **constantes durante todo el entrenamiento**, lo que permite un truco muy rentable: calcularlas una sola vez, guardarlas en caché, y entrenar la cabeza sobre esos vectores. El entrenamiento pasa a ser una regresión logística sobre 512 dimensiones y corre en segundos.

En **fine-tuning completo** todo el cuerpo recibe gradiente. Cuesta memoria —hay que guardar activaciones y estados del optimizador de 11 millones de parámetros— y exige una tasa de aprendizaje pequeña, típicamente uno o dos órdenes de magnitud menor que la del entrenamiento desde cero. La razón es la misma que en la ruta 25: la cabeza está inicializada al azar y sus primeros gradientes son grandes; si se propagan con una tasa alta, destruyen las representaciones preentrenadas antes de que la cabeza haya aprendido nada útil. El remedio habitual es un **calentamiento**: entrenar unas épocas solo la cabeza y descongelar después, o usar tasas discriminativas —más pequeñas en las capas iniciales, más grandes en las finales—, que es la traducción directa de que las capas iniciales necesitan cambiar menos.

En **entrenamiento desde cero** no hay transferencia: es la referencia que dice cuánto aportó realmente el preentrenamiento. Compararla con las otras dos con el mismo presupuesto de épocas es lo que convierte el laboratorio en un experimento y no en una demostración.

### El detalle que arruina un fine-tuning sin dar error

Hay un fallo específico de este régimen que no produce excepción, no aparece en las curvas de entrenamiento y degrada el resultado: las estadísticas de la **normalización por lotes**.

Como se vio en la ruta 03, esas capas guardan una media y una varianza acumuladas que no se aprenden por gradiente, sino que se actualizan en cada paso hacia adelante mientras el modelo esté en modo entrenamiento. Congelar los pesos con `requires_grad = False` **no congela esas estadísticas**. El resultado es que, en un supuesto «cuerpo congelado», las capas de normalización siguen adaptándose a los nuevos datos y la representación se mueve, aunque ningún peso reciba gradiente. Si además el lote es pequeño, las estadísticas del nuevo dominio son ruidosas y la degradación puede ser notable. Congelar de verdad exige poner esas capas en modo evaluación explícitamente.

Un segundo detalle de la misma familia: la normalización de entrada debe ser **la del preentrenamiento**. Un modelo entrenado con las medias y desviaciones de ImageNet espera recibir imágenes normalizadas con esos mismos valores; alimentarlo con otra normalización lo sitúa fuera de la distribución para la que se calibraron sus filtros, y el rendimiento cae sin que nada lo señale.

### Cuándo la transferencia deja de ayudar

La transferencia no es gratis ni siempre positiva, y conocer sus límites forma parte del laboratorio.

El factor que más pesa es la **distancia entre dominios**. Las capas iniciales de una red entrenada con fotografías naturales detectan bordes y texturas, y eso es útil para casi cualquier imagen natural —de ahí que funcione tan bien con mascotas—. Cuando el dominio destino es muy distinto —imágenes médicas monocromas, radar, microscopía— la ventaja se reduce a las primeras capas y puede desaparecer; se habla de **transferencia negativa** cuando el modelo preentrenado rinde por debajo de uno entrenado desde cero, algo que ocurre si el preentrenamiento induce invariancias contraproducentes para la tarea nueva.

El segundo factor es el **tamaño del conjunto destino**, y determina qué régimen conviene. Con pocos datos, el fine-tuning completo tiene demasiados grados de libertad y sobreajusta: la extracción de características suele ganar, precisamente porque su capacidad está limitada por construcción. A medida que crecen los datos la relación se invierte, y con un conjunto grande el entrenamiento desde cero puede alcanzar a ambos. Ese cruce de curvas es el resultado más interesante que este laboratorio puede producir, y por eso conviene reportar los tres regímenes juntos y no solo el ganador.

## Protocolo científico

- Ajustar transformaciones, vocabulario, normalización y selección de variables solo con `train`.
- Usar `validation` para arquitectura, hiperparámetros, checkpoint y umbrales.
- Evaluar `test` una vez, después de congelar las decisiones.
- Comparar contra **CNN pequeña entrenada desde cero**.
- Reportar variación entre semillas e intervalos de confianza; una métrica puntual no expresa toda la incertidumbre.

## Riesgos de interpretación

7.349 imágenes reales de 37 razas de perros y gatos.

El dataset refleja su proceso de recolección y no representa automáticamente otros períodos, países o poblaciones. Una asociación predictiva no demuestra causalidad.

## Pregunta crítica

> ¿Cuándo el preentrenamiento deja de aportar?

> Las referencias apuntan a las obras; no se reproduce su contenido, la redacción es original.

## 🔗 Referencias

- Géron — *Hands-On Machine Learning* (3.ª ed., O'Reilly), cap. 14 — CNN profundas y reutilización de modelos preentrenados para visión.
- Goodfellow, Bengio & Courville — *Deep Learning* (MIT Press, 2016), cap. 15 — aprendizaje de representaciones y transferencia entre tareas.
- Yosinski et al. (2014), *How transferable are features in deep neural networks?*, NeurIPS — estudio empírico de cómo la transferibilidad decae con la profundidad de la capa.
- Pan & Yang (2010), *A Survey on Transfer Learning*, IEEE Transactions on Knowledge and Data Engineering — marco conceptual de dominios y tareas origen/destino.
- Fuente del dataset: https://www.robots.ox.ac.uk/~vgg/data/pets/ — **The Oxford-IIIT Pet Dataset** (Visual Geometry Group, Creative Commons Attribution-ShareAlike 4.0 International); procedencia, versión y SHA-256 en el registro de fuentes, entrada `oxford-iiit-pet` — esta clase la usa para comparar extracción de características, fine-tuning y entrenamiento desde cero sobre 37 razas de mascotas.
- Consulte `docs/experiment-protocol.md`, `docs/reproducibility.md` y `docs/ethics-and-licenses.md`.

<!-- nav-bottom -->
## 🧭 Navegación del recorrido

| ⬅️ Laboratorio anterior | 🏠 Índice | Laboratorio siguiente ➡️ |
|---|:---:|---|
| [🕹️ DQN para inventario con demanda real](../../labs/10_dqn_reinforcement/README.md) | [Las 31 rutas](../../parts/README.md) | [🔀 Fusión de sensores](../../labs/12_multimodal_fusion/README.md) |

**En este laboratorio:** [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🟣 [Parte 3 — Familias especializadas: generar, decidir, relacionar](../../parts/03-familias-especializadas.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/11_transfer_learning/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
