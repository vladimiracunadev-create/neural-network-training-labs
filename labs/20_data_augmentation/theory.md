# Teoría — Aumento de datos

<!-- nav-top -->
> 🧭 **Ruta 21 / 31** · 🔴 [Parte 5 — La mecánica fina, ahora en profundidad](../../parts/05-mecanica-fina.md)
>
> [⬅️ 🛡️ Regularización](../../labs/19_regularization_dropout_batchnorm/theory.md) · [🏠 Índice de rutas](../../parts/README.md) · [🔍 Explicabilidad ➡️](../../labs/21_explainability/theory.md)
>
> [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

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

### Aumentar datos es declarar una invariancia

Aplicar un recorte aleatorio o un volteo horizontal parece un truco para «tener más datos». Es algo más preciso: es **afirmar** que la etiqueta no cambia bajo esa transformación. Cuando se entrena con pares (T(x), y) para T en un conjunto 𝒯, se le está diciendo al modelo que f(T(x)) debe valer lo mismo que f(x) para toda T de ese conjunto.

De ahí se sigue el criterio para elegir transformaciones, y también el error más caro: una transformación que **sí** cambia la etiqueta enseña algo falso. El volteo horizontal es seguro en CIFAR-10 —un avión volteado sigue siendo un avión— y sería destructivo en reconocimiento de dígitos o de texto, donde distinguir una `b` de una `d` depende justamente de la orientación. Rotar 180° un `6` produce un `9` con etiqueta equivocada. La lista de aumentaciones no es genérica: **depende del dominio**, y justificarla forma parte del diseño experimental.

El efecto sobre la función objetivo es explícito. En vez de minimizar la pérdida sobre los datos observados, se minimiza su esperanza sobre la distribución aumentada,

ℒ_aug(θ) = 𝔼_(x,y) 𝔼_(T∼𝒯) [ ℓ( f_θ(T(x)), y ) ],

y esa esperanza extra actúa como un regularizador: penaliza que la salida varíe cuando la entrada se mueve dentro de las transformaciones declaradas, es decir, **suaviza la función aprendida** en las direcciones que 𝒯 recorre. Por eso el aumento y el weight decay de la ruta 19 no son intercambiables: uno restringe la magnitud de los pesos, el otro restringe la forma de la función en direcciones concretas y elegidas.

Con transformaciones estocásticas aplicadas en cada época, el modelo prácticamente **nunca ve dos veces el mismo ejemplo**, lo que dificulta la memorización. Ese es el mecanismo por el que el aumento reduce la brecha entre entrenamiento y validación, y la razón de que su efecto sea mayor cuanto más pequeño es el conjunto de datos.

### Solo en `train`, y por qué es tan fácil equivocarse

Las transformaciones aleatorias se aplican **únicamente al conjunto de entrenamiento**. Aplicarlas a `validation` o a `test` introduce ruido aleatorio en la evaluación: dos ejecuciones sobre el mismo modelo darían métricas distintas, y la comparación entre configuraciones dejaría de ser válida. La evaluación debe ser determinista.

Esto obliga a separar dos cosas que suelen ir juntas en el mismo bloque de código: el **preprocesamiento** —redimensionar, convertir a tensor, normalizar con las estadísticas de `train`— se aplica a las tres particiones, y el **aumento** —recortes, volteos, perturbaciones de color— solo a una. Mezclarlos en una sola cadena aplicada a todo es el error de implementación más común de esta ruta, y su síntoma es un `validation` inexplicablemente ruidoso.

Existe una excepción deliberada y bien definida: el **aumento en inferencia** (TTA), que consiste en promediar las predicciones sobre varias versiones transformadas de la misma entrada. Suele mejorar algo la métrica a costa de multiplicar el tiempo de inferencia, pero es una **decisión de despliegue** que debe declararse, no un aumento accidental de la evaluación. Si se usa, se usa igual en todas las variantes comparadas.

### Las aumentaciones que mezclan ejemplos

Más allá de las transformaciones geométricas y de color, hay una familia que opera sobre pares de ejemplos y merece conocerse porque cambia también la etiqueta.

**Mixup** interpola linealmente dos ejemplos y sus etiquetas:

x̃ = λ·xᵢ + (1 − λ)·xⱼ,   ỹ = λ·yᵢ + (1 − λ)·yⱼ,   con λ ∼ Beta(α, α).

El modelo aprende así que entre dos clases la transición debe ser gradual, lo que suaviza la frontera de decisión y —efecto documentado— mejora la **calibración** de las probabilidades, que es justo lo que mide la ruta 22. **CutMix** hace lo mismo con parches: recorta una región de una imagen y la pega en otra, ponderando las etiquetas por el área ocupada, lo que preserva la estructura local que el mixup difumina.

Ambas rompen el supuesto de que la etiqueta es una clase única y exigen una pérdida que acepte objetivos blandos; la entropía cruzada la admite sin cambios, sumando los dos términos ponderados.

### Cómo se mide si el aumento aportó

La línea base del laboratorio —**la misma CNN sin aumento**— es la comparación que da sentido a todo, y debe correr con partición, semillas, arquitectura y presupuesto de épocas idénticos. Solo cambia el conjunto 𝒯.

Hay un detalle temporal que conviene anticipar al leer las curvas: el aumento **ralentiza el ajuste al entrenamiento**. Con el mismo número de épocas, la variante aumentada suele mostrar peor métrica de entrenamiento y tardar más en converger, porque cada época presenta un problema ligeramente distinto. Comparar a igual número de épocas puede por tanto **subestimar** su beneficio, y por eso conviene mirar también la curva completa y no solo el punto final.

Y hay dos resultados que reportar por separado. El primero es la métrica **limpia**, sobre datos sin transformar, que dice si el aumento mejoró la generalización ordinaria. El segundo es la **robustez**: evaluar el modelo sobre entradas perturbadas —ruido, desenfoque, cambios de brillo— y medir cuánto cae. Un modelo puede ganar poco en limpio y mucho en robustez, y ese es exactamente el caso en que el aumento vale la pena aunque la tabla principal apenas se mueva.

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

<!-- nav-bottom -->
## 🧭 Navegación del recorrido

| ⬅️ Laboratorio anterior | 🏠 Índice | Laboratorio siguiente ➡️ |
|---|:---:|---|
| [🛡️ Regularización](../../labs/19_regularization_dropout_batchnorm/README.md) | [Las 31 rutas](../../parts/README.md) | [🔍 Explicabilidad](../../labs/21_explainability/README.md) |

**En este laboratorio:** [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🔴 [Parte 5 — La mecánica fina, ahora en profundidad](../../parts/05-mecanica-fina.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/20_data_augmentation/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
