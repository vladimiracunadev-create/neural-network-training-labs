# Teoría — Segmentación semántica con U-Net

<!-- nav-top -->
> 🧭 **Ruta 27 / 31** · 🔬 [Parte 7 — Especializaciones avanzadas](../../parts/07-especializaciones-avanzadas.md)
>
> [⬅️ 🔧 Fine-tuning eficiente de transformer](../../advanced_labs/25_transformer_finetuning/theory.md) · [🏠 Índice de rutas](../../parts/README.md) · [🎙️ Clasificación de audio con SpeechCommands ➡️](../../advanced_labs/27_audio_speechcommands/theory.md)
>
> [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

Arquitectura encoder-decoder, conexiones skip, pérdida por píxel e intersección sobre unión.

## Idea central

Una CNN de clasificación responde «hay un gato en esta foto». La segmentación responde algo mucho más exigente: «este píxel es gato, este es fondo, y este es el borde entre ambos». La salida deja de ser una etiqueta y pasa a ser **una imagen de etiquetas**, del mismo tamaño que la entrada. Ese cambio de forma es el que obliga a cambiar de arquitectura.

El problema de fondo es un conflicto entre dos necesidades opuestas. Para decidir *qué* hay en una región hace falta contexto amplio: un parche de 8×8 píxeles de pelaje no distingue un gato de una alfombra. Ganar contexto significa reducir la resolución con submuestreos sucesivos, y cada submuestreo destruye información sobre *dónde* estaba exactamente cada cosa. Al final de un encoder típico la red sabe muy bien qué hay en la imagen y muy mal en qué píxel empieza. La segmentación necesita las dos respuestas a la vez.

La U-Net resuelve ese conflicto sin renunciar a ninguna de las dos: baja la resolución para ganar contexto y luego la sube para recuperar el detalle, pero en cada nivel de subida **reinyecta** los mapas de alta resolución que el encoder había producido antes de perderlos. Esas son las conexiones skip, y son la idea entera de la arquitectura. Su efecto se ve mejor en la clase más difícil de este laboratorio, el contorno de la mascota: son franjas de pocos píxeles que el submuestreo borra por completo y que solo el camino directo desde el encoder puede restituir.

El desbalance es el segundo protagonista. En una foto de mascota, el fondo ocupa la mayoría de los píxeles y el contorno una fracción mínima. Un modelo que prediga «fondo» en todas partes obtiene una exactitud por píxel alta y es completamente inútil: por eso la métrica no es la exactitud, sino la intersección sobre unión desglosada por clase, y por eso la pérdida necesita un término que no se deje dominar por la clase mayoritaria.

## Fundamento matemático

La segmentación semántica asigna a *cada* píxel de la imagen una etiqueta de clase. Formalmente, dada una entrada X ∈ ℝ^(H×W×3) se busca una función que produzca un mapa de probabilidades ŷ ∈ ℝ^(H×W×C), donde C es el número de clases (aquí: mascota, fondo y contorno). Es una clasificación densa: en lugar de una etiqueta por imagen, se predice una por posición espacial. Las **redes totalmente convolucionales** (FCN) de Long, Shelhamer y Darrell hicieron esto viable al sustituir las capas densas finales de una CNN por convoluciones, permitiendo salidas del tamaño de la imagen mediante *upsampling* (convoluciones transpuestas).

La **U-Net** refina esta idea con una arquitectura simétrica en forma de U. El **encoder** (camino de contracción) aplica bloques de convolución seguidos de submuestreo (max-pooling), reduciendo la resolución espacial y aumentando la profundidad de canales: captura el *qué* (contexto semántico) pero pierde el *dónde* (detalle espacial). El **decoder** (camino de expansión) revierte el proceso con upsampling progresivo hasta recuperar la resolución original. La clave son las **conexiones skip**: en cada nivel, los mapas de características del encoder se concatenan con los del decoder de igual resolución. Así se reinyecta la información espacial de alta frecuencia (bordes, contornos finos) que el submuestreo había diluido, resolviendo el compromiso entre contexto y localización. Esto es decisivo para la clase "contorno", que ocupa franjas delgadas de pocos píxeles.

El entrenamiento minimiza una **pérdida por píxel**, típicamente la entropía cruzada promediada sobre todas las posiciones:

ℒ_CE = −(1 / (H·W)) · Σ_(i,j) Σ_(c=1..C) y_(i,j,c) · log ŷ_(i,j,c),

donde la probabilidad por clase se obtiene con un softmax sobre el eje de canales, ŷ_(i,j,c) = e^(z_(i,j,c)) / Σ_k e^(z_(i,j,k)). Como las clases suelen estar desbalanceadas (el fondo domina), se complementa con la **pérdida Dice**, ℒ_Dice = 1 − (2·Σ ŷ·y + ε) / (Σ ŷ + Σ y + ε), donde ε > 0 evita división por cero; Dice premia el solape directo y es más robusta al desbalance.

La métrica principal es la **intersección sobre unión** (IoU), o índice de Jaccard, definida por clase como

IoU = |A ∩ B| / |A ∪ B| = TP / (TP + FP + FN),

siendo A la máscara predicha y B la real. Vale 1 si coinciden perfectamente y 0 si no se solapan; el *mean IoU* promedia sobre clases y es el estándar de la segmentación semántica. La línea base "máscara de clase mayoritaria" predice siempre la clase más frecuente: fija un piso trivial que la U-Net debe superar ampliamente para demostrar que aprende estructura real y no solo la proporción de píxeles de fondo.

### La aritmética de la U: cuánto se pierde y cuánto se recupera

Cada nivel del encoder aplica convoluciones y un max-pooling de factor 2. Tras L niveles, un mapa de H×W queda en (H/2^L)×(W/2^L): con L = 4 y una entrada de 128×128, el cuello de botella mide 8×8. Ese es el precio explícito del contexto.

Lo que se gana a cambio se mide con el **campo receptivo**, la región de la entrada que influye en una sola activación. Para una pila de capas, crece según

R_ℓ = R_(ℓ−1) + (F_ℓ − 1) · Π_(i<ℓ) s_i,

donde F_ℓ es el tamaño del filtro de la capa ℓ y s_i los strides acumulados de las anteriores. La consecuencia es que cada submuestreo **duplica** el efecto de las convoluciones posteriores: una convolución 3×3 tras cuatro poolings ve una ventana de 48 píxeles de la imagen original, mientras que la misma convolución al principio ve solo 3. Por eso el fondo se clasifica bien en las capas profundas y el borde no.

La subida usa **convoluciones transpuestas**, cuya dimensión de salida invierte la fórmula de la convolución normal:

dimensión_salida = (dimensión_entrada − 1)·s − 2p + F.

Con s = 2, F = 2 y p = 0 se duplica exactamente la resolución. Tras cada subida se concatena el mapa del encoder de igual resolución —la conexión skip— de modo que el decoder recibe C_dec + C_enc canales: la información semántica que trae de abajo y la información espacial que nunca pasó por el cuello de botella. Es una concatenación, no una suma, y esa distinción importa: la red aprende con sus pesos cómo combinar ambas fuentes en vez de imponerles el mismo peso por construcción.

### Cómo se combate el desbalance en la pérdida

La entropía cruzada por píxel trata todas las posiciones por igual, así que en una imagen con 80 % de fondo el gradiente está dominado por píxeles fáciles. Hay dos correcciones habituales, y el laboratorio usa ambas.

La primera es **ponderar las clases** en la entropía cruzada, ℒ_CE^w = −(1/(H·W)) · Σ_(i,j) Σ_c w_c · y_(i,j,c) · log ŷ_(i,j,c), con pesos inversamente proporcionales a la frecuencia, típicamente w_c ∝ 1 / f_c o w_c ∝ 1 / √f_c, siendo f_c la fracción de píxeles de la clase c. La raíz cuadrada modera la corrección: la ponderación inversa pura suele desestabilizar el entrenamiento porque dispara el gradiente de clases con muy pocos píxeles.

La segunda es sumar el término **Dice**, que no mide aciertos por píxel sino solape entre conjuntos y por tanto es insensible al tamaño del fondo. La pérdida total queda

ℒ = ℒ_CE^w + λ · ℒ_Dice,   con   ℒ_Dice = 1 − (1/C) · Σ_c (2·Σ_(i,j) ŷ_(i,j,c)·y_(i,j,c) + ε) / (Σ_(i,j) ŷ_(i,j,c) + Σ_(i,j) y_(i,j,c) + ε).

El Dice se calcula sobre las probabilidades ŷ sin binarizar, lo que lo hace diferenciable —una versión "blanda" del coeficiente clásico— y permite optimizarlo por descenso de gradiente. Los dos términos se complementan: la entropía cruzada da gradiente denso y estable desde el primer paso; el Dice orienta el entrenamiento hacia la métrica que de verdad se reporta.

### Por qué Dice e IoU no son la misma cifra

Ambos miden solape y se confunden con frecuencia, pero no coinciden. Con TP, FP y FN contados sobre píxeles,

IoU = TP / (TP + FP + FN),   Dice = 2·TP / (2·TP + FP + FN),

y están ligados por una relación monótona exacta:

Dice = 2·IoU / (1 + IoU),   equivalentemente   IoU = Dice / (2 − Dice).

Como Dice ≥ IoU siempre (con igualdad solo en 0 y en 1), **el Dice siempre se ve mejor**. Un IoU de 0,50 es un Dice de 0,67; un IoU de 0,80, un Dice de 0,89. Reportar uno creyendo que es el otro infla el resultado sin que nada falle visiblemente, y es la razón de que este laboratorio fije el `mean_iou` como métrica de selección y exija además el desglose `iou_per_class`: un mean IoU alto puede convivir con un IoU de contorno cercano a cero, que es exactamente el fallo que la arquitectura pretendía evitar.

## Visualización específica

Imagen, máscara real, máscara predicha, IoU por clase y mapas intermedios. Los mapas intermedios muestran cómo el encoder abstrae el contexto y las conexiones skip recuperan el detalle; el IoU por clase expone en qué categoría (mascota, fondo o contorno) falla más el modelo.

## Riesgo de interpretación

Las imágenes se concentran en mascotas y fondos cotidianos; no generaliza a segmentación médica o industrial. Un IoU global alto puede ocultar mal desempeño en clases minoritarias como el contorno, por lo que conviene leer siempre el IoU desglosado por clase.

## 🔗 Referencias

> Las referencias apuntan a las obras; no se reproduce su contenido, la redacción es original.

- Long, Shelhamer & Darrell (2015), *Fully Convolutional Networks for Semantic Segmentation*, CVPR — funda la segmentación densa reemplazando capas densas por convoluciones y upsampling.
- Ronneberger, Fischer & Brox (2015), *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI — encoder-decoder simétrico con conexiones skip para localización precisa.
- Goodfellow, Bengio & Courville — *Deep Learning* (MIT Press, 2016), cap. 9 — fundamentos de las redes convolucionales que sustentan el encoder-decoder.
- Fuente del dataset: https://www.robots.ox.ac.uk/~vgg/data/pets/ — **The Oxford-IIIT Pet Dataset** (Visual Geometry Group, Creative Commons Attribution-ShareAlike 4.0 International); procedencia, versión y SHA-256 en el registro de fuentes, entrada `oxford-iiit-pet` — esta clase usa sus trimaps de segmentación para entrenar una U-Net y medir intersección sobre unión por clase, incluido el contorno.

<!-- nav-bottom -->
## 🧭 Navegación del recorrido

| ⬅️ Laboratorio anterior | 🏠 Índice | Laboratorio siguiente ➡️ |
|---|:---:|---|
| [🔧 Fine-tuning eficiente de transformer](../../advanced_labs/25_transformer_finetuning/README.md) | [Las 31 rutas](../../parts/README.md) | [🎙️ Clasificación de audio con SpeechCommands](../../advanced_labs/27_audio_speechcommands/README.md) |

**En este laboratorio:** [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🔬 [Parte 7 — Especializaciones avanzadas](../../parts/07-especializaciones-avanzadas.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/26_segmentation_unet/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
