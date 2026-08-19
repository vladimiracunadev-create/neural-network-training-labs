# Teoría — GAN generativa

<!-- nav-top -->
> 🧭 **Ruta 9 / 31** · 🟣 [Parte 3 — Familias especializadas: generar, decidir, relacionar](../../parts/03-familias-especializadas.md)
>
> [⬅️ 🔭 Transformer para noticias](../../labs/07_transformer_attention/theory.md) · [🏠 Índice de rutas](../../parts/README.md) · [🕸️ GNN sobre red de citas ➡️](../../labs/09_gnn_graphs/theory.md)
>
> [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

## Propósito

Generar prendas a partir de imágenes reales de Fashion-MNIST.

## Idea central

Este laboratorio estudia **aprendizaje adversarial generativo** usando `fashion_mnist`, un dataset público real procedente de Torchvision / Zalando Research.

Una **red generativa adversarial (GAN)** plantea el aprendizaje como un juego entre dos redes con objetivos opuestos. El **generador** G toma ruido aleatorio z y trata de producir imágenes que parezcan prendas reales. El **discriminador** D es un clasificador que recibe una imagen y estima la probabilidad de que sea real (proveniente del dataset) y no falsa (generada por G). Ambos se entrenan a la vez: D mejora en distinguir real de falso, y G mejora en engañar a D. La metáfora habitual es la del falsificador (G) y el detective (D): cada uno fuerza al otro a mejorar, y en el equilibrio ideal el falsificador produce prendas indistinguibles de las auténticas.

Lo elegante es que G nunca ve las imágenes reales directamente ni recibe una pérdida de reconstrucción píxel a píxel; aprende **solo a través del gradiente que le pasa D**. En vez de decirle a G "copia esta imagen", D le dice "esto todavía se nota falso por aquí", y ese señal guía a G hacia la variedad de imágenes plausibles. Este laboratorio usa una **DCGAN** (GAN convolucional profunda), donde G usa convoluciones transpuestas para expandir el ruido hasta una imagen de 28×28 y D usa convoluciones para clasificarla; esta receta convolucional es la que estabilizó el entrenamiento de GANs sobre imágenes.

## Fundamento matemático

El objetivo original es un juego minimax de suma cero sobre el valor V(D, G):

    min_G max_D  V(D, G) = 𝔼_{x∼p_data}[ log D(x) ] + 𝔼_{z∼p_z}[ log(1 − D(G(z))) ]

Leámoslo por partes. El discriminador D quiere **maximizar** V: para muestras reales x quiere D(x) → 1 (así log D(x) → 0, su máximo), y para muestras falsas G(z) quiere D(G(z)) → 0 (así log(1 − D(G(z))) → 0). El generador G quiere **minimizar** V respecto al segundo término: busca que D(G(z)) → 1, es decir, engañar a D. z se muestrea de una distribución simple p_z (típicamente 𝒩(0, I)) y G la transforma en la distribución generada p_g. El entrenamiento alterna pasos: se congela G y se da un paso de ascenso de gradiente en θ_D, luego se congela D y se da un paso de descenso en θ_G.

¿Por qué este juego produce imágenes realistas? Goodfellow et al. probaron que, para un G fijo, el discriminador óptimo es D*(x) = p_data(x) / (p_data(x) + p_g(x)). Sustituyendo D* en V, el objetivo de G se vuelve equivalente a minimizar la **divergencia de Jensen–Shannon** entre la distribución real p_data y la generada p_g (salvo constantes): min_G V = 2·D_JS(p_data ‖ p_g) − log 4. El mínimo global se alcanza cuando p_g = p_data, es decir, cuando el generador reproduce exactamente la distribución de las prendas reales y D no puede hacer mejor que responder ½ en todo. Ese es el sentido preciso de "generar imágenes indistinguibles".

En la práctica, el término log(1 − D(G(z))) tiene gradiente casi nulo justo cuando G es malo (al inicio, D lo detecta con facilidad), así que se suele entrenar G maximizando 𝔼_z[ log D(G(z)) ] —el truco del "gradiente no saturante"— que apunta al mismo óptimo pero da señal fuerte desde el principio. Conectando con los cuatro elementos: la **representación de entrada** es el vector de ruido z para G y la imagen (28×28) para D; la **función del modelo** son las dos redes convolucionales G y D; la **función de pérdida** es la entropía cruzada binaria derivada de V (una para D, otra para G); y la **regla de actualización** son los dos pasos de gradiente alternados θ_D ← θ_D + η ∇_{θ_D} V y θ_G ← θ_G − η ∇_{θ_G} V. El notebook muestra las dimensiones de los tensores en cada capa y conserva la misma implementación que el script de terminal.

### El discriminador óptimo y de dónde sale la divergencia

Que el juego minimax equivalga a minimizar una divergencia de Jensen-Shannon no es una afirmación suelta: se deriva en dos pasos y merece verse, porque explica el fallo del método.

Primero se fija G y se busca el D óptimo. El objetivo, escrito como integral sobre x, es ∫ [ p_r(x)·log D(x) + p_g(x)·log(1 − D(x)) ] dx. El integrando se maximiza punto a punto, y derivando respecto de D(x) e igualando a cero:

D*(x) = p_r(x) / ( p_r(x) + p_g(x) ).

Es un resultado con una lectura clara: el discriminador perfecto no memoriza ejemplos, estima la **razón de densidades** en cada punto. Donde solo hay datos reales vale 1; donde solo hay generados, 0; y donde ambas distribuciones coinciden, exactamente ½ —el punto de máxima confusión—.

Segundo, sustituyendo D* en el objetivo y reordenando, aparece

V(G, D*) = 2·JS(p_r ‖ p_g) − 2·log 2,

de modo que minimizar en G equivale a minimizar la divergencia de Jensen-Shannon. El óptimo global se alcanza cuando p_g = p_r, y entonces D* ≡ ½ y el valor del juego es −2·log 2 ≈ −1,386.

Aquí está el problema que la ruta 28 resolverá. La JS entre dos distribuciones con soportes **disjuntos** vale log 2 sea cual sea la distancia entre ellas: es constante, y su gradiente es cero. Y los soportes son disjuntos casi siempre al principio, porque las imágenes reales viven en una variedad de dimensión bajísima dentro del espacio de píxeles y las generadas, otra. La consecuencia es la paradoja característica de las GAN: **cuanto mejor es el discriminador, menos aprende el generador**, porque un D casi perfecto satura y deja de transmitir dirección. Toda la dificultad práctica de entrenar una GAN —equilibrar los dos jugadores, no dejar que ninguno gane— nace de ahí.

### Colapso de modos, y por qué la pérdida no sirve para decidir

El colapso de modos tiene una explicación exacta en el objetivo. El generador no está obligado a **cubrir** p_r; está obligado a producir muestras que D no distinga. Si encuentra una prenda concreta que engaña al discriminador, generarla siempre es una estrategia óptima desde su punto de vista: la pérdida no contiene ningún término que premie la diversidad. El resultado es un modelo que produce imágenes convincentes de dos o tres tipos de prenda y ninguna del resto, con un valor de pérdida perfectamente razonable.

De ahí se sigue lo que más desconcierta al entrenar la primera GAN: **las curvas de pérdida no son criterio de selección**. En un entrenamiento supervisado, la pérdida de validación baja y elegir el mínimo es lo correcto. Aquí, ambos jugadores optimizan objetivos opuestos, así que las pérdidas oscilan alrededor de un equilibrio y su valor no indica calidad: pueden bajar mientras las muestras empeoran. Un discriminador que gana produce pérdida baja para él y muestras malas; un generador que gana puede estar colapsado. La única evaluación sensata es externa al juego —mirar las muestras, y medir cobertura de clases con un clasificador entrenado aparte—, y es exactamente lo que hace este laboratorio.

Sobre la selección del checkpoint conviene ser explícito: como no hay una pérdida monótona que minimizar, el criterio debe fijarse **antes** de mirar los resultados y dejarse escrito en `experiment.lock.json`. Elegir a posteriori la época cuyas muestras se ven mejor es seleccionar sobre el conjunto de evaluación, y ese resultado no es reproducible.

### Cómo se mide algo que no tiene etiqueta correcta

Un clasificador se evalúa contra la verdad; un generador, no: no existe «la imagen correcta». Por eso las métricas generativas son todas indirectas, y conviene saber qué mide cada familia.

La vía que usa este laboratorio es un **clasificador externo** entrenado sobre datos reales. Aplicado a las muestras generadas ofrece dos señales distintas: si sus predicciones son **confiadas** —distribución p(y|x) concentrada— las imágenes son reconocibles; y si el promedio de esas predicciones sobre muchas muestras, p(y), está **repartido** entre las diez clases, hay diversidad. Ambas cosas a la vez son lo que se busca, y la segunda es la que detecta el colapso: un generador colapsado puede producir imágenes nítidas y perfectamente clasificables, pero su p(y) marginal se concentra en una o dos clases.

Comparar directamente las distribuciones de rasgos —los vectores intermedios del clasificador para muestras reales frente a generadas— es la idea que subyace a métricas como la distancia de Fréchet, y tiene la ventaja de penalizar tanto la baja fidelidad como la baja cobertura. Todas ellas, sin excepción, son **aproximaciones**: dependen del clasificador elegido, no capturan errores semánticos sutiles y no sustituyen la inspección humana. Reportarlas como si fueran una medida objetiva de calidad es el error de interpretación más común en la literatura generativa.

La línea base **PCA generativa** es especialmente instructiva aquí: muestrear en el subespacio de las componentes principales y reconstruir produce prendas borrosas pero **diversas**, justo el defecto contrario al del colapso. Contrastar ambos fallos —nitidez sin cobertura frente a cobertura sin nitidez— es lo que enseña que en generación no hay una única métrica que ordene los modelos.

El riesgo técnico característico es el **colapso de modos** (mode collapse): G descubre unas pocas imágenes que engañan a D y las produce siempre, perdiendo diversidad aunque la pérdida parezca buena. Por eso este laboratorio no se conforma con las curvas de pérdida y mide diversidad, distancia al vecino real más cercano y discrepancia de momentos: distinguir *diversidad real* de *ruido visual* o de un puñado de prototipos repetidos es exactamente el reto de evaluar una GAN.

## Protocolo científico

- Ajustar transformaciones, vocabulario, normalización y selección de variables solo con `train`.
- Usar `validation` para arquitectura, hiperparámetros, checkpoint y umbrales.
- Evaluar `test` una vez, después de congelar las decisiones.
- Comparar contra **PCA generativa y distribución real de referencia**.
- Reportar variación entre semillas e intervalos de confianza; una métrica puntual no expresa toda la incertidumbre.

## Riesgos de interpretación

No usa anillos ni puntos inventados; entrena con prendas reales etiquetadas.

El dataset refleja su proceso de recolección y no representa automáticamente otros períodos, países o poblaciones. Una asociación predictiva no demuestra causalidad.

## Pregunta crítica

> ¿Cómo se distingue diversidad real de ruido visual?

> Las referencias apuntan a las obras; no se reproduce su contenido, la redacción es original.

## 🔗 Referencias

- Foster — *Generative Deep Learning* (2.ª ed., O'Reilly) — tratamiento práctico de GANs, DCGAN y evaluación de modelos generativos.
- Goodfellow, Bengio & Courville — *Deep Learning* (MIT Press, 2016), cap. 20 — modelos generativos profundos y fundamentos del marco adversarial.
- Goodfellow et al. (2014), *Generative Adversarial Nets*, NeurIPS — formulación original del juego minimax y prueba del óptimo p_g = p_data.
- Radford, Metz & Chintala (2016), *Unsupervised Representation Learning with Deep Convolutional GANs (DCGAN)*, ICLR — arquitectura convolucional que estabilizó el entrenamiento de GANs sobre imágenes.
- Fuente del dataset: https://github.com/zalandoresearch/fashion-mnist — **Fashion-MNIST** (Zalando Research (Zalando SE), MIT License); procedencia, versión y SHA-256 en el registro de fuentes, entrada `fashion-mnist` — esta clase la usa para entrenar una GAN que genera prendas a partir de imágenes reales etiquetadas, en lugar de figuras sintéticas.
- Consulte `docs/experiment-protocol.md`, `docs/reproducibility.md` y `docs/ethics-and-licenses.md`.

<!-- nav-bottom -->
## 🧭 Navegación del recorrido

| ⬅️ Laboratorio anterior | 🏠 Índice | Laboratorio siguiente ➡️ |
|---|:---:|---|
| [🔭 Transformer para noticias](../../labs/07_transformer_attention/README.md) | [Las 31 rutas](../../parts/README.md) | [🕸️ GNN sobre red de citas](../../labs/09_gnn_graphs/README.md) |

**En este laboratorio:** [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🟣 [Parte 3 — Familias especializadas: generar, decidir, relacionar](../../parts/03-familias-especializadas.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/08_gan_generation/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
