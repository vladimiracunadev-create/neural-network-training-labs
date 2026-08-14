# Teoría — Destilación de conocimiento

<!-- nav-top -->
> 🧭 **Ruta 15 / 31** · 🟠 [Parte 4 — Entrenar mejor, más barato y sin centralizar datos](../../parts/04-entrenamiento-eficiente.md)
>
> [⬅️ 🎛️ Búsqueda de hiperparámetros](../../labs/13_hyperparameter_search/theory.md) · [🏠 Índice de rutas](../../parts/README.md) · [🌐 Aprendizaje federado por participante ➡️](../../labs/15_federated_learning/theory.md)
>
> [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

## Propósito

Transferir conocimiento de una CNN profesora a una estudiante compacta.

## Idea central

Este laboratorio estudia **transferencia de conocimiento profesor-estudiante** usando `cifar10`, un dataset público real procedente de Torchvision / University of Toronto. La observación de partida es que un modelo grande y preciso (el profesor) no solo predice la clase correcta: en su distribución de salida codifica *cómo de parecidas* considera a las clases entre sí. Por ejemplo, una imagen de gato puede recibir alta probabilidad en "gato", algo en "perro" y casi nada en "camión". Esas probabilidades relativas —las **etiquetas blandas** (soft labels)— son información rica que la etiqueta dura (solo "gato") descarta.

La destilación entrena a un modelo pequeño (el estudiante) para que imite esa distribución blanda del profesor, además de acertar la etiqueta verdadera. El estudiante recibe así una señal de aprendizaje mucho más informativa por ejemplo: en lugar de un único bit correcto/incorrecto, aprende la estructura de similitud que el profesor descubrió con más capacidad y más cómputo. El resultado es un modelo compacto que se acerca a la exactitud del grande con una fracción de los parámetros y la latencia, útil para desplegar en dispositivos con recursos limitados.

La **temperatura** T es la palanca central. Al dividir los logits por T antes del softmax se suavizan las probabilidades: con T alto, las diferencias entre clases se atenúan y emergen las señales pequeñas (esa pizca de "perro" en la imagen de gato) que de otro modo quedarían aplastadas cerca de cero. La pregunta crítica del laboratorio es qué temperatura equilibra mejor la señal dura (la etiqueta verdadera) con la señal blanda (el conocimiento del profesor).

## Fundamento matemático

Sean z^t los logits del profesor y z^s los del estudiante para las K clases. El softmax con temperatura T produce distribuciones suavizadas:

  p_k(z; T) = e^{z_k / T} / Σⱼ e^{z_j / T}

Con T = 1 se recupera el softmax normal; con T > 1 la distribución se aplana y revela las probabilidades pequeñas. La pérdida de destilación combina dos términos:

  ℒ = α · CE(y, softmax(z^s)) + (1 − α) · T² · KL( softmax(z^t / T) ‖ softmax(z^s / T) )

El primer término es la **entropía cruzada** contra la etiqueta dura y (aprender lo correcto). El segundo es la **divergencia de Kullback–Leibler** entre la distribución blanda del profesor y la del estudiante, ambas a temperatura T (imitar al profesor). El coeficiente α ∈ [0,1] pondera cuánto pesa cada objetivo.

El factor T² tiene una justificación precisa. Los gradientes del término blando respecto a z^s escalan aproximadamente como 1/T² (porque tanto el softmax suavizado como su derivada introducen un factor 1/T). Multiplicar por T² **reescala** esos gradientes para que su magnitud sea comparable a la del término duro, de modo que al variar T no haya que reajustar α ni el learning rate. La divergencia KL entre profesor p y estudiante q es:

  KL(p ‖ q) = Σₖ p_k · log( p_k / q_k )

y se minimiza cuando q iguala a p. El estudiante actualiza sus parámetros por descenso de gradiente, θ^s ← θ^s − η · ∇_{θ^s} ℒ, mientras el profesor permanece congelado. La formulación conecta cuatro elementos: representación de entrada (la imagen), función del modelo (estudiante), función de pérdida (CE + KL con temperatura) y regla de actualización (SGD con ∇). El notebook muestra las dimensiones de los tensores y conserva la misma implementación que el script de terminal.

### Qué información contienen las etiquetas suaves

La pregunta de fondo del método es por qué un estudiante aprende mejor imitando a un profesor que entrenando con las etiquetas verdaderas, si las etiquetas verdaderas son, por definición, correctas.

La respuesta está en lo que cada señal transporta. Una etiqueta dura sobre CIFAR-10 es un vector one-hot: dice «esto es un gato» y nada más. La salida del profesor para esa misma imagen podría ser p = (gato 0,85 · perro 0,10 · caballo 0,03 · …): dice «esto es un gato, se parece bastante a un perro y algo a un caballo, y no tiene nada que ver con un camión». Esa estructura de similitudes entre clases —lo que Hinton llamó **conocimiento oscuro**— es información que la etiqueta dura no contiene y que el estudiante recibe gratis en cada ejemplo.

Cuantifíquese: una etiqueta dura sobre C clases aporta como mucho log₂ C bits, aquí unos 3,3. Una distribución completa aporta C − 1 números reales, y sobre todo aporta **restricciones sobre la geometría** del espacio de salida. Por eso el estudiante converge con menos datos y menos épocas: cada ejemplo es más informativo.

La temperatura es lo que hace visible esa información. Con T = 1, un profesor bien entrenado produce distribuciones muy picudas —0,999 en la clase correcta— y las probabilidades del resto son tan pequeñas que su contribución al gradiente es despreciable. Elevar T aplana:

p_i^(T) = exp(z_i / T) / Σ_j exp(z_j / T),

y en el límite T → ∞ la distribución tiende a la uniforme, mientras que con T → 0 tiende al one-hot y la destilación degenera en entrenamiento normal. El valor útil está en el medio, típicamente entre 2 y 5, y es un hiperparámetro que se ajusta en `validation`.

El factor T² del término de destilación tiene una razón exacta. Al derivar el softmax con temperatura, el gradiente respecto de los logits escala como 1/T; como la pérdida combina dos términos y solo uno lleva temperatura, sin corrección el término destilado perdería peso relativo al subir T. Multiplicar por T² restablece la escala y permite variar T sin tener que reajustar λ ni la tasa de aprendizaje.

### Qué se gana y qué se paga al comprimir

La comparación que el laboratorio pide es tridimensional, y conviene tener claro qué mide cada eje.

Los **parámetros** miden el tamaño en disco y en memoria. Los **FLOPs** miden el trabajo aritmético. Y la **latencia** mide el tiempo real, que no es proporcional a ninguno de los dos: una red con menos operaciones puede ser más lenta si sus accesos a memoria son irregulares o si sus capas son demasiado pequeñas para saturar el hardware paralelo. Reportar solo la reducción de parámetros y llamarla «aceleración» es un error frecuente; por eso este laboratorio mide la latencia directamente.

Hay un supuesto sin el cual todo el método se cae, y merece enunciarse: la destilación presupone que **existe** una red pequeña capaz de resolver la tarea, y que el problema era encontrarla, no que faltara capacidad. Si la arquitectura del estudiante no tiene capacidad suficiente para representar la función, ninguna cantidad de destilación la creará; el profesor solo puede guiar la búsqueda hacia una buena solución dentro del espacio que el estudiante ya podía representar. De ahí que un estudiante demasiado pequeño no mejore con destilación, y que la elección de su arquitectura sea parte del experimento.

Conviene además situar la destilación entre sus alternativas, porque resuelven el mismo problema por vías distintas y son combinables. La **poda** elimina pesos o canales de la red grande según su importancia; la **cuantización** —que se estudia en la ruta 23— reduce la precisión numérica de 32 a 8 bits, con un factor 4 de reducción casi garantizado y soporte de hardware. Frente a ambas, la destilación tiene una ventaja específica: permite cambiar la **arquitectura** por completo, no solo encoger la existente.

### Cómo evaluar honestamente al estudiante

Tres reglas que este laboratorio hace explícitas y que se incumplen con facilidad.

La primera: la comparación correcta no es estudiante destilado frente a profesor, sino **estudiante destilado frente al mismo estudiante entrenado con etiquetas duras**. Esa es la única que aísla el aporte de la destilación; comparar contra el profesor solo mide cuánta calidad se perdió al comprimir, que es otra pregunta.

La segunda: el profesor debe estar **congelado** durante la destilación y no puede haberse entrenado con ninguna información de `validation` ni de `test`. Un profesor que vio esos datos filtra su conocimiento al estudiante a través de las etiquetas suaves, y la fuga es indirecta pero real.

La tercera: la latencia se mide en el **hardware objetivo**, con el modelo en modo evaluación, tras un calentamiento y promediando varias repeticiones. Una medición única incluye el costo de inicializar los núcleos de cómputo y puede sobreestimar el tiempo en un orden de magnitud.

## Protocolo científico

- Ajustar transformaciones, vocabulario, normalización y selección de variables solo con `train`.
- Usar `validation` para arquitectura, hiperparámetros, checkpoint y umbrales.
- Evaluar `test` una vez, después de congelar las decisiones.
- Comparar contra **Estudiante entrenado solo con etiquetas**.
- Reportar variación entre semillas e intervalos de confianza; una métrica puntual no expresa toda la incertidumbre.

## Riesgos de interpretación

Mismo test real para profesor, estudiante base y estudiante destilada.

El dataset refleja su proceso de recolección y no representa automáticamente otros períodos, países o poblaciones. Una asociación predictiva no demuestra causalidad.

## Pregunta crítica

> ¿Qué temperatura equilibra mejor señales duras y blandas?

## 🔗 Referencias

> Las referencias apuntan a las obras; no se reproduce su contenido, la redacción es original.

- Goodfellow, Bengio & Courville — *Deep Learning* (MIT Press, 2016) — fundamentos de softmax, entropía cruzada y compresión de modelos.
- Buciluă, Caruana & Niculescu-Mizil (2006), *Model Compression*, KDD — idea seminal de comprimir un conjunto grande en un modelo pequeño que imita sus salidas.
- Hinton, Vinyals & Dean (2015), *Distilling the Knowledge in a Neural Network*, NeurIPS Deep Learning Workshop — formulación de la destilación con temperatura y etiquetas blandas usada en este laboratorio.
- Sanh et al. (2019), *DistilBERT, a distilled version of BERT* — aplicación a gran escala que muestra estudiantes compactos cercanos al profesor.
- Fuente del dataset: https://www.cs.toronto.edu/~kriz/cifar.html
- Consulte `docs/experiment-protocol.md`, `docs/reproducibility.md` y `docs/ethics-and-licenses.md`.

<!-- nav-bottom -->
## 🧭 Navegación del recorrido

| ⬅️ Laboratorio anterior | 🏠 Índice | Laboratorio siguiente ➡️ |
|---|:---:|---|
| [🎛️ Búsqueda de hiperparámetros](../../labs/13_hyperparameter_search/README.md) | [Las 31 rutas](../../parts/README.md) | [🌐 Aprendizaje federado por participante](../../labs/15_federated_learning/README.md) |

**En este laboratorio:** [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🟠 [Parte 4 — Entrenar mejor, más barato y sin centralizar datos](../../parts/04-entrenamiento-eficiente.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/14_knowledge_distillation/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
