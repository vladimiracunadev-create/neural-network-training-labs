# Teoría — Aprendizaje autosupervisado SimCLR

Dos vistas, similitud coseno, pérdida NT-Xent y evaluación linear probe.

## Fundamento matemático

El aprendizaje autosupervisado busca aprender representaciones útiles **sin etiquetas**, inventando una tarea a partir de los propios datos. SimCLR (Chen et al.) lo hace con **aprendizaje contrastivo**: la idea es que dos vistas distorsionadas de la misma imagen deben quedar cerca en el espacio de representación, y vistas de imágenes diferentes, lejos. Para cada imagen del lote se generan **dos vistas** aplicando aumentaciones estocásticas (recorte aleatorio, cambio de color, desenfoque, escala de grises). Ambas pasan por un encoder f (aquí una ResNet18), que produce una representación h = f(x), y luego por una cabeza de proyección g (un MLP) que da z = g(h). El contraste se hace sobre z; la representación h es la que se conserva para tareas posteriores.

La medida de cercanía es la **similitud coseno**, que compara dirección ignorando magnitud:

sim(zᵢ, zⱼ) = (zᵢ · zⱼ) / (‖zᵢ‖ · ‖zⱼ‖).

Con un lote de N imágenes se obtienen 2N vistas. Para un par positivo (i, j) —las dos vistas de la misma imagen— las otras 2(N−1) vistas actúan como **negativos**. La pérdida es la **NT-Xent** (normalized temperature-scaled cross-entropy), una forma de InfoNCE:

ℒ_(i,j) = − log [ exp( sim(zᵢ, zⱼ) / τ ) / Σ_(k=1..2N, k≠i) exp( sim(zᵢ, z_k) / τ ) ].

El numerador premia la similitud del par positivo; el denominador suma sobre todos los negativos, empujándolos a ser disímiles. Es, en esencia, un softmax de "clasificación": entre todas las vistas del lote, identificar cuál es la pareja correcta. El **parámetro de temperatura** τ > 0 escala las similitudes: valores pequeños agudizan las diferencias y penalizan con fuerza los negativos difíciles, controlando la concentración del espacio aprendido. La pérdida total promedia ℒ_(i,j) sobre todos los pares positivos del lote, por lo que **lotes grandes** aportan más negativos y suelen mejorar la representación.

Otras familias resuelven de distinto modo la necesidad de negativos y estabilidad. **MoCo** (He et al.) mantiene un banco/cola de negativos y un *encoder de momento* actualizado como θ_k ← m·θ_k + (1 − m)·θ_q, desacoplando el número de negativos del tamaño de lote. **BYOL** (Grill et al.) prescinde por completo de negativos: usa una red *online* y una *target* (esta última actualizada por media móvil exponencial) y evita el colapso trivial mediante un predictor asimétrico y el gradiente detenido en la rama target. Comparar estas estrategias aclara qué componentes son realmente imprescindibles.

La calidad de lo aprendido se juzga con **linear probe**: se **congela** el encoder f y se entrena únicamente un clasificador lineal (softmax) sobre las representaciones h con las etiquetas reales. Como el encoder no se ajusta, la accuracy resultante mide directamente cuánta información linealmente separable capturaron las representaciones autosupervisadas. La línea base "ResNet18 aleatoria + linear probe" fija el piso: cuánto se logra con un encoder sin entrenar, para aislar el aporte real del preentrenamiento contrastivo. Métricas complementarias como knn_accuracy y la uniformidad del embedding evalúan la estructura del espacio sin entrenar clasificador alguno.

## Visualización específica

Pares aumentados, proyección 2D, vecinos y curva de linear probe. Ver los pares aumentados aclara qué invariancias se están imponiendo; la proyección 2D y los vecinos más cercanos muestran si imágenes semánticamente similares se agrupan; la curva de linear probe cuantifica la utilidad de las representaciones frente a la línea base aleatoria.

## Riesgo de interpretación

La elección de aumentos define invariancias y puede borrar información relevante para tareas posteriores. Por ejemplo, forzar invariancia al color ayuda en unas tareas pero perjudica otras donde el color es discriminante; una buena accuracy en linear probe para una tarea no garantiza transferencia a otra con necesidades distintas.

## 🔗 Referencias

> Las referencias apuntan a las obras; no se reproduce su contenido, la redacción es original.

- Chen et al. (2020), *A Simple Framework for Contrastive Learning of Visual Representations* (SimCLR), ICML — define la pérdida NT-Xent, el rol de las aumentaciones y la cabeza de proyección.
- He et al. (2020), *Momentum Contrast for Unsupervised Visual Representation Learning* (MoCo), CVPR — cola de negativos y encoder de momento para escalar el contraste.
- Grill et al. (2020), *Bootstrap Your Own Latent* (BYOL), NeurIPS — aprendizaje sin negativos mediante redes online/target y predictor asimétrico.
