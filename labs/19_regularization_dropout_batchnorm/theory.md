# Teoría — Regularización

<!-- nav-top -->
> 🧭 **Ruta 20 / 31** · 🔴 [Parte 5 — La mecánica fina, ahora en profundidad](../../parts/05-mecanica-fina.md)
>
> [⬅️ ⚙️ Optimizadores y schedulers](../../labs/18_optimizers_and_schedulers/theory.md) · [🏠 Índice de rutas](../../parts/README.md) · [🔄 Aumento de datos ➡️](../../labs/20_data_augmentation/theory.md)
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

### Weight decay: qué supuesto está imponiendo

Añadir (λ/2)·‖θ‖² a la pérdida no es un truco: es exactamente lo que sale de hacer inferencia bayesiana con una **distribución previa gaussiana** sobre los pesos. Maximizando la probabilidad posterior,

log p(θ | D) = log p(D | θ) + log p(θ) + const,

y con p(θ) = 𝒩(0, σ²I), el segundo término es −‖θ‖²/(2σ²), es decir, el término L2 con λ = 1/σ². La lectura es que regularizar equivale a declarar una creencia previa: **los pesos pequeños son más probables que los grandes**, y λ mide cuánta evidencia hace falta para abandonar esa creencia.

Su efecto sobre el gradiente es un encogimiento multiplicativo, θ ← (1 − η·λ)·θ − η·g, que empuja continuamente hacia cero y solo se contrarresta donde los datos lo exigen. Dos consecuencias prácticas: los **sesgos no se regularizan** —desplazan la función, no controlan su complejidad, y encogerlos solo introduce error—, y en Adam hay que usar la forma desacoplada de AdamW por la razón que explica la ruta 18.

### Dropout: por qué se escala y qué apaga exactamente

Durante el entrenamiento, el dropout multiplica cada activación por una máscara de Bernoulli que la anula con probabilidad p. Eso cambia la magnitud esperada de la salida: si h tenía esperanza 𝔼[h], tras el apagado pasa a (1 − p)·𝔼[h]. Si en inferencia —donde no se apaga nada— no se corrigiera, la red recibiría activaciones sistemáticamente mayores que las vistas durante el entrenamiento.

La implementación estándar es el **dropout invertido**: se divide por (1 − p) ya en el entrenamiento,

h̃ = (h ⊙ m) / (1 − p),   con m ~ Bernoulli(1 − p),

de modo que 𝔼[h̃] = 𝔼[h] y la inferencia no necesita corrección alguna: basta con desactivar la capa. Es la razón, otra vez, de que el modo evaluación sea obligatorio al medir.

Conceptualmente, entrenar con dropout equivale a entrenar un **conjunto exponencial** de subredes que comparten pesos —2^n máscaras posibles para n unidades—, y evaluar sin dropout aproxima el promedio de todas ellas. De ahí que sea un método de conjunto barato. Su efecto concreto es impedir la **coadaptación**: como ninguna unidad puede contar con que otra esté presente, cada una debe aportar señal útil por sí sola, y la representación resultante es redundante y más robusta.

Dónde ponerlo importa. En capas densas, con p entre 0,2 y 0,5, funciona bien. En capas convolucionales el dropout puntual es poco efectivo, porque los píxeles vecinos de un mapa de activación están muy correlacionados y apagar uno no elimina la información —la aporta su vecino—: la variante útil es el dropout **por canal**, que apaga mapas de características completos.

### La interacción entre dropout y normalización por lotes

Combinar ambas cosas es tan habitual como problemático, y conviene saber por qué.

El dropout modifica la **varianza** de las activaciones, y la modifica de forma distinta en entrenamiento y en inferencia. La normalización por lotes, por su parte, acumula estadísticas durante el entrenamiento —cuando el dropout está activo— para usarlas en inferencia, cuando ya no lo está. Las estadísticas acumuladas no corresponden entonces a la distribución que la capa ve al evaluar, y esa discrepancia de varianza degrada el resultado. Es la razón de que muchas arquitecturas modernas prescindan del dropout en los bloques convolucionales normalizados y lo reserven para la cabeza densa, o de que se coloque siempre **después** de la normalización y no antes.

Conviene además recordar que la normalización por lotes ya regulariza por sí sola, porque el ruido de las estadísticas del minilote actúa como perturbación estocástica. Ese efecto **se debilita con lotes grandes**, así que la cantidad de regularización efectiva de un modelo depende del tamaño de lote: cambiarlo altera el equilibrio y puede exigir reajustar λ y p. Es una interacción que el diseño experimental debe controlar.

### Cómo se mide si la regularización funcionó

La cifra que hay que mirar no es la métrica de validación sino la **brecha** entre entrenamiento y validación. Un modelo que acierta el 99 % en entrenamiento y el 78 % en validación está memorizando; uno que acierta 84 % y 82 % ha generalizado, aunque su cifra de entrenamiento sea peor. Regularizar consiste precisamente en aceptar peor ajuste a cambio de menor brecha, y ese intercambio debe verse en los números.

De ahí que el experimento correcto no sea «activar dropout y comprobar que mejora», sino barrer p y λ observando **las dos curvas a la vez**. Hay tres desenlaces posibles y todos informan: si la brecha es grande, falta regularización; si ambas curvas son bajas y cercanas, sobra —el modelo está subajustado y la regularización le impide aprender—; y si la brecha ya era pequeña de entrada, el modelo no tenía capacidad excedente y la regularización solo puede empeorarlo.

La **parada temprana** merece contarse como parte del mismo conjunto de herramientas, porque es regularización implícita: detener el entrenamiento cuando la validación deja de mejorar limita cuánto pueden crecer los pesos y, en modelos lineales, se puede demostrar equivalente a una penalización L2 con λ dependiente del número de pasos. Es también la más barata, y por eso el repositorio la aplica por defecto mediante `patience`.

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
- Fuente del dataset: https://github.com/zalandoresearch/fashion-mnist — **Fashion-MNIST** (Zalando Research (Zalando SE), MIT License); procedencia, versión y SHA-256 en el registro de fuentes, entrada `fashion-mnist` — esta clase la usa para medir el efecto de dropout, weight decay y normalización por lotes sobre imágenes reales de prendas.
- Consulte `docs/experiment-protocol.md`, `docs/reproducibility.md` y `docs/ethics-and-licenses.md`.

<!-- nav-bottom -->
## 🧭 Navegación del recorrido

| ⬅️ Laboratorio anterior | 🏠 Índice | Laboratorio siguiente ➡️ |
|---|:---:|---|
| [⚙️ Optimizadores y schedulers](../../labs/18_optimizers_and_schedulers/README.md) | [Las 31 rutas](../../parts/README.md) | [🔄 Aumento de datos](../../labs/20_data_augmentation/README.md) |

**En este laboratorio:** [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🔴 [Parte 5 — La mecánica fina, ahora en profundidad](../../parts/05-mecanica-fina.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/19_regularization_dropout_batchnorm/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
