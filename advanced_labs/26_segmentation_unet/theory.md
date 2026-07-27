# Teoría — Segmentación semántica con U-Net

Arquitectura encoder-decoder, conexiones skip, pérdida por píxel e intersección sobre unión.

## Fundamento matemático

La segmentación semántica asigna a *cada* píxel de la imagen una etiqueta de clase. Formalmente, dada una entrada X ∈ ℝ^(H×W×3) se busca una función que produzca un mapa de probabilidades ŷ ∈ ℝ^(H×W×C), donde C es el número de clases (aquí: mascota, fondo y contorno). Es una clasificación densa: en lugar de una etiqueta por imagen, se predice una por posición espacial. Las **redes totalmente convolucionales** (FCN) de Long, Shelhamer y Darrell hicieron esto viable al sustituir las capas densas finales de una CNN por convoluciones, permitiendo salidas del tamaño de la imagen mediante *upsampling* (convoluciones transpuestas).

La **U-Net** refina esta idea con una arquitectura simétrica en forma de U. El **encoder** (camino de contracción) aplica bloques de convolución seguidos de submuestreo (max-pooling), reduciendo la resolución espacial y aumentando la profundidad de canales: captura el *qué* (contexto semántico) pero pierde el *dónde* (detalle espacial). El **decoder** (camino de expansión) revierte el proceso con upsampling progresivo hasta recuperar la resolución original. La clave son las **conexiones skip**: en cada nivel, los mapas de características del encoder se concatenan con los del decoder de igual resolución. Así se reinyecta la información espacial de alta frecuencia (bordes, contornos finos) que el submuestreo había diluido, resolviendo el compromiso entre contexto y localización. Esto es decisivo para la clase "contorno", que ocupa franjas delgadas de pocos píxeles.

El entrenamiento minimiza una **pérdida por píxel**, típicamente la entropía cruzada promediada sobre todas las posiciones:

ℒ_CE = −(1 / (H·W)) · Σ_(i,j) Σ_(c=1..C) y_(i,j,c) · log ŷ_(i,j,c),

donde la probabilidad por clase se obtiene con un softmax sobre el eje de canales, ŷ_(i,j,c) = e^(z_(i,j,c)) / Σ_k e^(z_(i,j,k)). Como las clases suelen estar desbalanceadas (el fondo domina), se complementa con la **pérdida Dice**, ℒ_Dice = 1 − (2·Σ ŷ·y + ε) / (Σ ŷ + Σ y + ε), donde ε > 0 evita división por cero; Dice premia el solape directo y es más robusta al desbalance.

La métrica principal es la **intersección sobre unión** (IoU), o índice de Jaccard, definida por clase como

IoU = |A ∩ B| / |A ∪ B| = TP / (TP + FP + FN),

siendo A la máscara predicha y B la real. Vale 1 si coinciden perfectamente y 0 si no se solapan; el *mean IoU* promedia sobre clases y es el estándar de la segmentación semántica. La línea base "máscara de clase mayoritaria" predice siempre la clase más frecuente: fija un piso trivial que la U-Net debe superar ampliamente para demostrar que aprende estructura real y no solo la proporción de píxeles de fondo.

## Visualización específica

Imagen, máscara real, máscara predicha, IoU por clase y mapas intermedios. Los mapas intermedios muestran cómo el encoder abstrae el contexto y las conexiones skip recuperan el detalle; el IoU por clase expone en qué categoría (mascota, fondo o contorno) falla más el modelo.

## Riesgo de interpretación

Las imágenes se concentran en mascotas y fondos cotidianos; no generaliza a segmentación médica o industrial. Un IoU global alto puede ocultar mal desempeño en clases minoritarias como el contorno, por lo que conviene leer siempre el IoU desglosado por clase.

## 🔗 Referencias

> Las referencias apuntan a las obras; no se reproduce su contenido, la redacción es original.

- Long, Shelhamer & Darrell (2015), *Fully Convolutional Networks for Semantic Segmentation*, CVPR — funda la segmentación densa reemplazando capas densas por convoluciones y upsampling.
- Ronneberger, Fischer & Brox (2015), *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI — encoder-decoder simétrico con conexiones skip para localización precisa.
- Goodfellow, Bengio & Courville — *Deep Learning* (MIT Press, 2016), cap. 9 — fundamentos de las redes convolucionales que sustentan el encoder-decoder.
