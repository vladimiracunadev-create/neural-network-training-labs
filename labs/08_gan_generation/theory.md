# Teoría — GAN generativa

<!-- nav-top -->
> 🧭 **Ruta 9 / 31** · [⬅️ 🔭 Transformer para noticias](../../labs/07_transformer_attention/theory.md) · [🏠 Índice](../../README.md#laboratorios) · [🕸️ GNN sobre red de citas ➡️](../../labs/09_gnn_graphs/theory.md)
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
- Fuente del dataset: https://github.com/zalandoresearch/fashion-mnist
- Consulte `docs/experiment-protocol.md`, `docs/reproducibility.md` y `docs/ethics-and-licenses.md`.

<!-- nav-bottom -->
## 🧭 Navegación del recorrido

| ⬅️ Laboratorio anterior | 🏠 Índice | Laboratorio siguiente ➡️ |
|---|:---:|---|
| [🔭 Transformer para noticias](../../labs/07_transformer_attention/README.md) | [Las 31 rutas](../../README.md#laboratorios) | [🕸️ GNN sobre red de citas](../../labs/09_gnn_graphs/README.md) |

**En este laboratorio:** [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

[🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/08_gan_generation/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
