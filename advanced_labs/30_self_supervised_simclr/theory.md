# Teoría — Aprendizaje autosupervisado SimCLR

<!-- nav-top -->
> 🧭 **Ruta 31 / 31** · 🔬 [Parte 7 — Especializaciones avanzadas](../../parts/07-especializaciones-avanzadas.md)
>
> [⬅️ 🌫️ Difusión DDPM sobre Fashion-MNIST](../../advanced_labs/29_diffusion_ddpm/theory.md) · [🏠 Índice de rutas](../../parts/README.md) · *fin del recorrido* ➡️
>
> [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

Dos vistas, similitud coseno, pérdida NT-Xent y evaluación linear probe.

## Idea central

Todas las rutas anteriores necesitan etiquetas. Alguien tuvo que mirar 50 000 imágenes de CIFAR-10 y escribir «avión», «gato», «camión». Ese trabajo es caro, lento y, en muchos dominios reales —imágenes médicas, defectos industriales, sensores— directamente inviable a escala. La pregunta de este laboratorio es si se puede aprender una representación útil **sin ninguna etiqueta**, y cuánto se pierde por hacerlo.

La respuesta contrastiva parte de una intuición sencilla: aunque no sepamos *qué* hay en una imagen, sí sabemos algo con certeza absoluta —dos recortes distintos de la misma foto muestran la misma cosa, y un recorte de otra foto muestra algo distinto—. Eso basta para inventar una tarea de aprendizaje que no requiere anotador: acerca en el espacio de representación las dos vistas de una misma imagen, aleja las vistas de imágenes diferentes. La etiqueta la genera la propia estructura de los datos.

Lo delicado es qué invariancias se enseñan, porque **las decide el conjunto de aumentaciones**, no el algoritmo. Si se entrena con recortes agresivos y cambios de color, se le está diciendo al modelo que el color y el encuadre son irrelevantes. Eso es excelente para reconocer objetos y desastroso si la tarea posterior era distinguir un plátano maduro de uno verde. Las aumentaciones no son un detalle de implementación: son la definición operativa de «qué cosas deberían considerarse la misma».

También hay un fallo trivial que acecha. Una función que mapee **todas** las imágenes al mismo vector cumple perfectamente el objetivo de acercar las vistas positivas; es la solución degenerada, el colapso. Lo que lo evita es el término de negativos: al exigir simultáneamente que vistas de imágenes distintas queden lejos, la representación tiene que extenderse por el espacio en vez de contraerse a un punto.

Cómo se juzga el resultado es el otro aporte del laboratorio. Como no hay etiquetas durante el preentrenamiento, la calidad se mide después: se **congela** el encoder y se entrena únicamente un clasificador lineal encima. Si esa recta separa bien las clases, es porque la representación ya había organizado la información sin que nadie le dijera cuáles eran las categorías. Y la línea base es un encoder sin entrenar con la misma sonda lineal, que fija el piso: cuánto se obtiene por pura arquitectura, antes de aprender nada.

## Fundamento matemático

El aprendizaje autosupervisado busca aprender representaciones útiles **sin etiquetas**, inventando una tarea a partir de los propios datos. SimCLR (Chen et al.) lo hace con **aprendizaje contrastivo**: la idea es que dos vistas distorsionadas de la misma imagen deben quedar cerca en el espacio de representación, y vistas de imágenes diferentes, lejos. Para cada imagen del lote se generan **dos vistas** aplicando aumentaciones estocásticas (recorte aleatorio, cambio de color, desenfoque, escala de grises). Ambas pasan por un encoder f (aquí una ResNet18), que produce una representación h = f(x), y luego por una cabeza de proyección g (un MLP) que da z = g(h). El contraste se hace sobre z; la representación h es la que se conserva para tareas posteriores.

La medida de cercanía es la **similitud coseno**, que compara dirección ignorando magnitud:

sim(zᵢ, zⱼ) = (zᵢ · zⱼ) / (‖zᵢ‖ · ‖zⱼ‖).

Con un lote de N imágenes se obtienen 2N vistas. Para un par positivo (i, j) —las dos vistas de la misma imagen— las otras 2(N−1) vistas actúan como **negativos**. La pérdida es la **NT-Xent** (normalized temperature-scaled cross-entropy), una forma de InfoNCE:

ℒ_(i,j) = − log [ exp( sim(zᵢ, zⱼ) / τ ) / Σ_(k=1..2N, k≠i) exp( sim(zᵢ, z_k) / τ ) ].

El numerador premia la similitud del par positivo; el denominador suma sobre todos los negativos, empujándolos a ser disímiles. Es, en esencia, un softmax de "clasificación": entre todas las vistas del lote, identificar cuál es la pareja correcta. El **parámetro de temperatura** τ > 0 escala las similitudes: valores pequeños agudizan las diferencias y penalizan con fuerza los negativos difíciles, controlando la concentración del espacio aprendido. La pérdida total promedia ℒ_(i,j) sobre todos los pares positivos del lote, por lo que **lotes grandes** aportan más negativos y suelen mejorar la representación.

Otras familias resuelven de distinto modo la necesidad de negativos y estabilidad. **MoCo** (He et al.) mantiene un banco/cola de negativos y un *encoder de momento* actualizado como θ_k ← m·θ_k + (1 − m)·θ_q, desacoplando el número de negativos del tamaño de lote. **BYOL** (Grill et al.) prescinde por completo de negativos: usa una red *online* y una *target* (esta última actualizada por media móvil exponencial) y evita el colapso trivial mediante un predictor asimétrico y el gradiente detenido en la rama target. Comparar estas estrategias aclara qué componentes son realmente imprescindibles.

La calidad de lo aprendido se juzga con **linear probe**: se **congela** el encoder f y se entrena únicamente un clasificador lineal (softmax) sobre las representaciones h con las etiquetas reales. Como el encoder no se ajusta, la accuracy resultante mide directamente cuánta información linealmente separable capturaron las representaciones autosupervisadas. La línea base "ResNet18 aleatoria + linear probe" fija el piso: cuánto se logra con un encoder sin entrenar, para aislar el aporte real del preentrenamiento contrastivo. Métricas complementarias como knn_accuracy y la uniformidad del embedding evalúan la estructura del espacio sin entrenar clasificador alguno.

### Qué hace realmente el gradiente de la NT-Xent

Escribir la pérdida no basta para entender qué presión ejerce. Si se define sᵢₖ = sim(zᵢ, z_k)/τ y se llama pᵢₖ = exp(sᵢₖ) / Σ_(m≠i) exp(sᵢₘ) a la distribución softmax sobre los candidatos, la pérdida del par (i, j) es simplemente ℒ = −log p_ij, y su gradiente respecto de las similitudes es

∂ℒ/∂s_ij = −(1 − p_ij),   ∂ℒ/∂s_ik = p_ik   para k ≠ j.

La lectura es directa. El par positivo recibe un empuje proporcional a **1 − p_ij**, es decir, cuanto peor lo está haciendo la red, más fuerte el tirón; una vez que el positivo ya es el más similar, deja de aportar gradiente. Cada negativo recibe un empuje proporcional a **p_ik**, su propia probabilidad: los negativos que el modelo confunde con el positivo son los que más contribuyen, y los obviamente distintos casi no aportan. La NT-Xent, sin ningún mecanismo adicional, **se concentra automáticamente en los negativos difíciles**.

Esto explica el papel de la temperatura τ, que no es un hiperparámetro menor. Como divide las similitudes antes del softmax, τ pequeña agranda las diferencias y concentra casi todo el gradiente en el negativo más parecido —aprendizaje agresivo, riesgo de que un negativo que en realidad es de la misma clase (un falso negativo) domine la actualización—. τ grande aplana el softmax y reparte el gradiente entre todos los negativos por igual, lo que suaviza el aprendizaje pero desdibuja la estructura fina. Valores típicos están entre 0,07 y 0,5.

También explica por qué el tamaño del lote importa tanto aquí y no en un entrenamiento supervisado. Con N imágenes hay 2(N − 1) negativos por vista: el lote **es** el conjunto de negativos. Duplicarlo no solo estabiliza el gradiente, cambia la tarea, porque identificar la pareja correcta entre 511 candidatos es un problema más exigente —y por tanto más informativo— que entre 63.

### Los dos ejes que miden la calidad de una representación

Wang e Isola mostraron que optimizar una pérdida contrastiva equivale, en el límite de muchos negativos, a optimizar dos propiedades que se pueden medir por separado sobre la hiperesfera unidad. Son exactamente las dos métricas que este laboratorio reporta además de la exactitud.

La **alineación** mide cuánto se acercan las vistas positivas:

ℒ_align = 𝔼_((x,x⁺)) ‖ f(x) − f(x⁺) ‖²,

y baja cuando el encoder es invariante a las aumentaciones aplicadas. La **uniformidad** mide cuán repartida está la representación sobre la esfera, mediante un potencial gaussiano:

ℒ_uniform = log 𝔼_(x,y) [ e^(−t·‖f(x) − f(y)‖²) ],   con t > 0 (habitualmente 2),

y baja cuando los puntos se distribuyen sin apelotonarse. Las dos están en tensión: el colapso total tiene alineación perfecta y uniformidad pésima; un encoder aleatorio puede tener uniformidad razonable y alineación nula. Una buena representación necesita ambas, y por eso mirar solo una de las dos cifras induce a error. Esto da sentido a la métrica `embedding_uniformity` del laboratorio: es el detector de colapso que la exactitud del linear probe tardaría en revelar.

La **exactitud kNN** completa el cuadro sin entrenar nada. Se calcula asignando a cada imagen de test la clase mayoritaria entre sus k vecinos más próximos por similitud coseno en el espacio de representaciones. A diferencia del linear probe, no ajusta ningún parámetro, así que no puede compensar una representación mediocre con una frontera bien colocada: mide si las clases **ya están agrupadas** localmente. Cuando la exactitud lineal es alta y la kNN baja, suele significar que las clases son separables globalmente pero están entremezcladas a escala fina.

### Por qué la cabeza de proyección se descarta

Un detalle que parece arbitrario y no lo es: el contraste se hace sobre z = g(h), pero lo que se conserva para las tareas posteriores es **h**, no z. Medido empíricamente, la sonda lineal sobre h supera con claridad a la sonda sobre z.

La explicación es que la pérdida contrastiva empuja a z a ser invariante a las aumentaciones, y ser invariante significa **descartar** la información que las aumentaciones alteran: color, orientación, escala. Esa información suele ser inútil para la tarea contrastiva y útil para la tarea final. La cabeza g actúa como amortiguador: absorbe en sus propias capas la pérdida de información que el objetivo exige, y deja que h la conserve. Se entrena con g y se usa sin g, que es una de las decisiones de diseño menos intuitivas y más reproducibles de SimCLR.

Conviene cerrar con el límite del método. La representación aprendida es tan buena como las invariancias que se le impusieron: si las aumentaciones incluyen conversión a escala de grises, el encoder aprende a ignorar el color, y ninguna sonda posterior podrá recuperar lo que h ya no contiene. Elegir el conjunto de aumentaciones **es** elegir qué información se conserva.

## Visualización específica

Pares aumentados, proyección 2D, vecinos y curva de linear probe. Ver los pares aumentados aclara qué invariancias se están imponiendo; la proyección 2D y los vecinos más cercanos muestran si imágenes semánticamente similares se agrupan; la curva de linear probe cuantifica la utilidad de las representaciones frente a la línea base aleatoria.

## Riesgo de interpretación

La elección de aumentos define invariancias y puede borrar información relevante para tareas posteriores. Por ejemplo, forzar invariancia al color ayuda en unas tareas pero perjudica otras donde el color es discriminante; una buena accuracy en linear probe para una tarea no garantiza transferencia a otra con necesidades distintas.

## 🔗 Referencias

> Las referencias apuntan a las obras; no se reproduce su contenido, la redacción es original.

- Chen et al. (2020), *A Simple Framework for Contrastive Learning of Visual Representations* (SimCLR), ICML — define la pérdida NT-Xent, el rol de las aumentaciones y la cabeza de proyección.
- He et al. (2020), *Momentum Contrast for Unsupervised Visual Representation Learning* (MoCo), CVPR — cola de negativos y encoder de momento para escalar el contraste.
- Grill et al. (2020), *Bootstrap Your Own Latent* (BYOL), NeurIPS — aprendizaje sin negativos mediante redes online/target y predictor asimétrico.
- Wang & Isola (2020), *Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere*, ICML — descompone la pérdida contrastiva en alineación y uniformidad, y define las métricas correspondientes.
- Oord, Li & Vinyals (2018), *Representation Learning with Contrastive Predictive Coding*, arXiv — introduce InfoNCE, la pérdida de la que NT-Xent es un caso particular.
- Fuente del dataset: https://www.cs.toronto.edu/~kriz/cifar.html — **CIFAR-10** (University of Toronto, La fuente no declara una licencia); procedencia, versión y SHA-256 en el registro de fuentes, entrada `cifar-10` — esta clase la usa sin etiquetas durante el preentrenamiento contrastivo, y solo las recupera para la sonda lineal que evalúa la representación.

<!-- nav-bottom -->
## 🧭 Navegación del recorrido

| ⬅️ Laboratorio anterior | 🏠 Índice | Laboratorio siguiente ➡️ |
|---|:---:|---|
| [🌫️ Difusión DDPM sobre Fashion-MNIST](../../advanced_labs/29_diffusion_ddpm/README.md) | [Las 31 rutas](../../parts/README.md) | *— fin del recorrido* |

**En este laboratorio:** [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🔬 [Parte 7 — Especializaciones avanzadas](../../parts/07-especializaciones-avanzadas.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/30_self_supervised_simclr/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
