# Teoría — Regularización

<!-- nav-top -->
> 🧭 **Ruta 20 / 31** · [⬅️ ⚙️ Optimizadores y schedulers](../../labs/18_optimizers_and_schedulers/theory.md) · [🏠 Índice](../../README.md#laboratorios) · [🔄 Aumento de datos ➡️](../../labs/20_data_augmentation/theory.md)
>
> [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

## Propósito

Medir dropout, weight decay y batch normalization.

## Idea central

Este laboratorio estudia **dropout, batch normalization y weight decay** usando `fashion_mnist`, un dataset público real procedente de Torchvision / Zalando Research.

Una red con suficiente capacidad puede *memorizar* el conjunto de entrenamiento —incluido su ruido— sin aprender el patrón que generaliza. Esa brecha entre el rendimiento en `train` y en `validation` es la señal del **sobreajuste**. La regularización es el conjunto de técnicas que restringen o perturban el modelo para que prefiera soluciones más simples y estables, sacrificando algo de ajuste en `train` a cambio de un mejor comportamiento en datos no vistos. El laboratorio compara tres mecanismos complementarios sobre imágenes reales de prendas (`fashion_mnist`, 28×28 en escala de grises, 10 clases).

La idea que conecta las tres técnicas es que penalizar la complejidad o introducir ruido controlado durante el entrenamiento actúa como un *prior* hacia funciones suaves. **Weight decay** limita la magnitud de los pesos; **dropout** impide que las neuronas dependan de coadaptaciones frágiles; **batch normalization** estabiliza las distribuciones internas y añade un ruido de mini-lote con efecto regularizador. Medimos su impacto observando cómo cambia la brecha train–validation y la exactitud final.

## Fundamento matemático

Regularización explícita e implícita; brecha train-validation.

**Weight decay (regularización L2)** añade a la pérdida un término proporcional al cuadrado de la norma de los pesos: ℒ_total = ℒ_datos + (λ/2)·‖θ‖². Su gradiente es λ·θ, de modo que en cada paso los pesos se contraen ligeramente hacia cero: θ ← θ − η(∇ℒ_datos + λ·θ). La intuición es que pesos grandes producen funciones con curvaturas abruptas que se ajustan al ruido; penalizar ‖θ‖² empuja hacia funciones más planas y suaves. El hiperparámetro λ gradúa el compromiso entre ajuste y simplicidad.

**Dropout** apaga aleatoriamente una fracción p de las activaciones en cada paso de entrenamiento. Formalmente, cada activación se multiplica por una máscara Bernoulli: h̃ = h ⊙ m, con mᵢ ~ Bernoulli(1−p), y se reescala por 1/(1−p) para mantener la esperanza. Al forzar a la red a producir la salida correcta con subconjuntos distintos de neuronas, impide que unas pocas unidades formen "conspiraciones" (coadaptaciones) y reparte la representación de forma redundante. Puede leerse como un promedio implícito sobre un número exponencial de subredes que comparten pesos: en inferencia se usa la red completa sin máscara, aproximando ese ensamble.

**Batch normalization** normaliza cada activación dentro del mini-lote antes de la no linealidad. Para una activación x calcula la media μ_B y varianza σ²_B del lote, normaliza x̂ = (x − μ_B)/√(σ²_B + ε), y luego reescala y desplaza con parámetros aprendidos: y = γ·x̂ + β. Al mantener las distribuciones internas con media y varianza estables reduce el *internal covariate shift*, permite tasas de aprendizaje mayores y hace el entrenamiento menos sensible a la inicialización; los parámetros γ, β devuelven a la red la libertad de recuperar cualquier escala útil. Además, como μ_B y σ²_B dependen del mini-lote, inyectan un ruido estocástico que actúa como regularizador implícito. En inferencia se sustituyen por estadísticas acumuladas durante el entrenamiento, para que la predicción de un ejemplo no dependa de sus compañeros de lote.

La lectura conjunta: weight decay actúa sobre la *magnitud* de los pesos, dropout sobre la *estructura* de las representaciones y batch norm sobre la *escala de las activaciones*. Ninguno elimina el sobreajuste por decreto; cada uno desplaza el equilibrio sesgo–varianza, y el laboratorio mide empíricamente cuál reduce la brecha train–validation sin caer en el subajuste.

## Protocolo científico

- Ajustar transformaciones, vocabulario, normalización y selección de variables solo con `train`.
- Usar `validation` para arquitectura, hiperparámetros, checkpoint y umbrales.
- Evaluar `test` una vez, después de congelar las decisiones.
- Comparar contra **MLP sin regularización**.
- Reportar variación entre semillas e intervalos de confianza; una métrica puntual no expresa toda la incertidumbre.

## Riesgos de interpretación

Prendas reales normalizadas en 28×28.

El dataset refleja su proceso de recolección y no representa automáticamente otros períodos, países o poblaciones. Una asociación predictiva no demuestra causalidad.

## Pregunta crítica

> ¿Qué técnica reduce sobreajuste sin subajustar?

## 🔗 Referencias

> Las referencias apuntan a las obras; no se reproduce su contenido, la redacción es original.

- Goodfellow, Bengio y Courville — *Deep Learning* (MIT Press, 2016), cap. 7 — marco general de la regularización en aprendizaje profundo: penalizaciones de norma, dropout y estrategias de generalización.
- Géron — *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* (3.ª ed., O'Reilly, 2022), cap. 11 — técnicas prácticas para entrenar redes profundas, incluidas normalización por lotes y regularización.
- Srivastava et al. (2014), *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*, JMLR — formulación original de dropout y su interpretación como ensamble implícito.
- Ioffe y Szegedy (2015), *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*, ICML — definición de batch normalization y su efecto sobre la estabilidad del entrenamiento.
- Fuente del dataset: https://github.com/zalandoresearch/fashion-mnist
- Consulte `docs/experiment-protocol.md`, `docs/reproducibility.md` y `docs/ethics-and-licenses.md`.

<!-- nav-bottom -->
## 🧭 Navegación del recorrido

| ⬅️ Laboratorio anterior | 🏠 Índice | Laboratorio siguiente ➡️ |
|---|:---:|---|
| [⚙️ Optimizadores y schedulers](../../labs/18_optimizers_and_schedulers/README.md) | [Las 31 rutas](../../README.md#laboratorios) | [🔄 Aumento de datos](../../labs/20_data_augmentation/README.md) |

**En este laboratorio:** [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

[🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/19_regularization_dropout_batchnorm/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
