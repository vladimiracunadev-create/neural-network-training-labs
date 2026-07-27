# Teoría — Destilación de conocimiento

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
