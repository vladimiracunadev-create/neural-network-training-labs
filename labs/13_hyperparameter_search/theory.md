# Teoría — Búsqueda de hiperparámetros

<!-- nav-top -->
> 🧭 **Ruta 14 / 31** · [⬅️ 🔀 Fusión de sensores](../../labs/12_multimodal_fusion/theory.md) · [🏠 Índice](../../README.md#laboratorios) · [⚗️ Destilación de conocimiento ➡️](../../labs/14_knowledge_distillation/theory.md)
>
> [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

## Propósito

Optimizar profundidad, ancho, dropout y learning rate sin tocar test.

## Idea central

Este laboratorio estudia **búsqueda de hiperparámetros sin tocar test** usando `adult_census`, un dataset público real procedente de UCI. Los hiperparámetros no se aprenden por descenso de gradiente: son decisiones de diseño (número de capas, neuronas por capa, tasa de dropout, learning rate) que gobiernan *cómo* se aprenden los parámetros. Ajustarlos bien es lo que separa un modelo que memoriza de uno que generaliza.

La idea central del protocolo es tratar la búsqueda como un experimento con tres particiones estrictamente separadas. Se prueban muchas configuraciones, cada una se entrena con `train` y se puntúa con `validation`; el conjunto `test` permanece sellado hasta el final. Esto evita el **sesgo de selección optimista**: si eligiéramos la mejor configuración mirando `test`, esa métrica dejaría de ser una estimación honesta del desempeño futuro, porque habríamos ajustado nuestras decisiones al ruido específico de ese conjunto.

Sobre la estrategia de búsqueda, el laboratorio contrasta la intuición ingenua (probar en malla, grid search) con hallazgos empíricos más eficientes. La **búsqueda aleatoria** suele encontrar buenas configuraciones con menos evaluaciones porque, cuando pocos hiperparámetros dominan el desempeño, muestrear al azar explora más valores distintos de esos hiperparámetros importantes que una malla rígida. Frameworks modernos añaden búsqueda guiada (por ejemplo, muestreo bayesiano) y poda temprana de pruebas poco prometedoras.

## Fundamento matemático

Sea λ un vector de hiperparámetros en un espacio de búsqueda Λ (profundidad, ancho, dropout p, learning rate η, …). Para cada λ se entrena un modelo obteniendo parámetros óptimos sobre entrenamiento:

  θ*(λ) = argmin_θ ℒ_train(θ; λ)

y se evalúa su calidad en validación. La búsqueda de hiperparámetros es el problema anidado (bilevel):

  λ* = argmin_{λ ∈ Λ} ℒ_val( θ*(λ) )

El punto crítico es que **λ se elige mirando `validation`, nunca `test`**. El error de test solo se mide una vez, con λ* ya congelado, para estimar la generalización sin sesgo.

En la **búsqueda en malla** se discretiza cada dimensión y se prueban todas las combinaciones: el costo crece como el producto de los tamaños por dimensión (maldición de la dimensionalidad). En la **búsqueda aleatoria** se muestrean T configuraciones λ⁽¹⁾, …, λ⁽ᵀ⁾ de una distribución sobre Λ y se conserva el mejor. La intuición de por qué gana: si solo d_eff de las d dimensiones influyen de verdad, la malla desperdicia evaluaciones repitiendo los mismos valores de las dimensiones importantes, mientras que el muestreo aleatorio prueba T valores distintos de cada una.

Como cada θ*(λ) depende de la inicialización y del orden de los minibatches, la métrica de validación es una variable aleatoria. Por eso se reporta ℒ_val como media ± desviación sobre varias semillas: comparar dos configuraciones con un único número puede confundir una mejora real con ruido de entrenamiento. La formulación conecta cuatro elementos: representación de entrada, función del modelo, función de pérdida y regla de actualización. El notebook muestra las dimensiones de los tensores y conserva la misma implementación que el script de terminal.

## Protocolo científico

- Ajustar transformaciones, vocabulario, normalización y selección de variables solo con `train`.
- Usar `validation` para arquitectura, hiperparámetros, checkpoint y umbrales.
- Evaluar `test` una vez, después de congelar las decisiones.
- Comparar contra **Regresión logística**.
- Reportar variación entre semillas e intervalos de confianza; una métrica puntual no expresa toda la incertidumbre.

## Riesgos de interpretación

48.842 registros reales del censo de 1994.

El dataset refleja su proceso de recolección y no representa automáticamente otros períodos, países o poblaciones. Una asociación predictiva no demuestra causalidad.

## Pregunta crítica

> ¿El mejor trial generaliza a semillas nuevas?

## 🔗 Referencias

> Las referencias apuntan a las obras; no se reproduce su contenido, la redacción es original.

- Géron — *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* (3.ª ed., O'Reilly 2022), cap. 10 — introducción práctica a redes densas y al ajuste de sus hiperparámetros.
- Goodfellow, Bengio & Courville — *Deep Learning* (MIT Press, 2016), cap. 11 — metodología práctica para seleccionar hiperparámetros y depurar experimentos.
- Bergstra & Bengio (2012), *Random Search for Hyper-Parameter Optimization*, JMLR — evidencia de por qué la búsqueda aleatoria supera a la malla cuando pocos hiperparámetros dominan.
- Akiba et al. (2019), *Optuna: A Next-generation Hyperparameter Optimization Framework*, KDD — framework de búsqueda guiada con muestreo eficiente y poda temprana.
- Fuente del dataset: https://archive.ics.uci.edu/dataset/2/adult
- Consulte `docs/experiment-protocol.md`, `docs/reproducibility.md` y `docs/ethics-and-licenses.md`.

<!-- nav-bottom -->
## 🧭 Navegación del recorrido

| ⬅️ Laboratorio anterior | 🏠 Índice | Laboratorio siguiente ➡️ |
|---|:---:|---|
| [🔀 Fusión de sensores](../../labs/12_multimodal_fusion/README.md) | [Las 31 rutas](../../README.md#laboratorios) | [⚗️ Destilación de conocimiento](../../labs/14_knowledge_distillation/README.md) |

**En este laboratorio:** [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

[🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/13_hyperparameter_search/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
