# Teoría — Aumento de datos

## Propósito

Comparar recortes, volteos y perturbaciones sobre imágenes reales.

## Idea central

Este laboratorio estudia **aumento de datos seleccionado por validation** usando `cifar10`, un dataset público real procedente de Torchvision / University of Toronto.

El aumento de datos (*data augmentation*) genera, sobre la marcha, variantes transformadas de cada imagen de entrenamiento —recortes, volteos horizontales, cambios de brillo o color— manteniendo su etiqueta. La motivación es sencilla y profunda: si sabemos que la clase "gato" no cambia porque la imagen se desplace unos píxeles o se refleje en espejo, entonces exponer a la red a esas versiones le enseña una **invariancia** que de otro modo tendría que descubrir por sí sola (o no aprendería nunca). Efectivamente, ampliamos el conjunto de entrenamiento con ejemplos plausibles y así reducimos el sobreajuste sin recolectar más datos.

La clave metodológica es que las transformaciones codifican *conocimiento previo* sobre qué variaciones son irrelevantes para la tarea, y ese conocimiento debe ser correcto: un volteo horizontal es inocuo para reconocer animales, pero destruiría la etiqueta de un dígito o de un texto. Por eso el catálogo y la intensidad del aumento se eligen con `validation`, no con `test`, y la evaluación final se hace siempre sobre imágenes de test *sin* aumentar. Sobre `cifar10` (60.000 imágenes a color de 32×32 en 10 clases) comparamos una CNN con y sin aumento para aislar su contribución.

## Fundamento matemático

Invariancias y regularización por transformaciones.

Sea T una transformación (recorte, volteo, jitter de color) muestreada de una distribución p(T) que preserva la etiqueta: si (x, y) es un par imagen–clase, queremos que el modelo cumpla f(T(x)) ≈ f(x) para toda T. El aumento de datos convierte el objetivo de entrenamiento en una **esperanza sobre transformaciones**: en lugar de minimizar ℒ(f(x), y) minimizamos 𝔼_{T∼p(T)}[ ℒ(f(T(x)), y) ]. En la práctica esa esperanza se aproxima con Monte Carlo: cada época, cada imagen se ve bajo una T distinta muestreada al azar, de modo que el modelo nunca recibe exactamente el mismo ejemplo dos veces. El efecto es que la red aprende a asignar la misma etiqueta a toda una *órbita* de versiones de x, es decir, aprende invariancia (o al menos robustez) frente a esa familia de transformaciones.

Visto como regularización, el aumento suaviza la función aprendida: promediar la pérdida sobre pequeñas perturbaciones de la entrada penaliza que f cambie bruscamente ante variaciones que la etiqueta considera irrelevantes, lo que empuja hacia fronteras de decisión más estables. Frente a la regularización explícita (weight decay, que actúa sobre los pesos) o al dropout (que actúa sobre las activaciones), el aumento actúa sobre el **espacio de entrada** e inyecta el sesgo inductivo de forma directa e interpretable. Técnicas como Cutout borran una región rectangular de la imagen para forzar el uso de múltiples pistas, mientras que estrategias aprendidas como AutoAugment *buscan* la política de transformaciones p(T) que maximiza la exactitud de validación, en lugar de fijarla a mano.

El riesgo es que una transformación demasiado agresiva rompa la premisa de invariancia y cambie de hecho la etiqueta (un recorte que elimina el objeto, un giro que convierte un 6 en un 9): entonces se inyecta ruido de etiqueta y el rendimiento cae. La condición de validez es siempre la misma: T debe preservar la semántica de la clase. La medición sobre imágenes de test sin aumento garantiza que la mejora reportada refleje generalización real y no un artefacto del procedimiento de evaluación.

## Protocolo científico

- Ajustar transformaciones, vocabulario, normalización y selección de variables solo con `train`.
- Usar `validation` para arquitectura, hiperparámetros, checkpoint y umbrales.
- Evaluar `test` una vez, después de congelar las decisiones.
- Comparar contra **CNN sin aumento**.
- Reportar variación entre semillas e intervalos de confianza; una métrica puntual no expresa toda la incertidumbre.

## Riesgos de interpretación

La evaluación usa imágenes de test sin aumento.

El dataset refleja su proceso de recolección y no representa automáticamente otros períodos, países o poblaciones. Una asociación predictiva no demuestra causalidad.

## Pregunta crítica

> ¿La mejora proviene de invariancias coherentes?

## 🔗 Referencias

> Las referencias apuntan a las obras; no se reproduce su contenido, la redacción es original.

- Géron — *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* (3.ª ed., O'Reilly, 2022), cap. 14 — visión por computador con CNN y uso del aumento de datos para mejorar la generalización.
- Shorten y Khoshgoftaar (2019), *A survey on Image Data Augmentation for Deep Learning*, Journal of Big Data — panorámica sistemática de técnicas de aumento de imágenes.
- DeVries y Taylor (2017), *Improved Regularization of Convolutional Neural Networks with Cutout*, arXiv — borrado aleatorio de regiones como regularizador.
- Cubuk et al. (2019), *AutoAugment: Learning Augmentation Strategies from Data*, CVPR — búsqueda automática de políticas de aumento optimizadas por validación.
- Fuente del dataset: https://www.cs.toronto.edu/~kriz/cifar.html
- Consulte `docs/experiment-protocol.md`, `docs/reproducibility.md` y `docs/ethics-and-licenses.md`.
