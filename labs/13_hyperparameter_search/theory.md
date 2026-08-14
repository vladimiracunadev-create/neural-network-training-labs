# Teoría — Búsqueda de hiperparámetros

<!-- nav-top -->
> 🧭 **Ruta 14 / 31** · 🟠 [Parte 4 — Entrenar mejor, más barato y sin centralizar datos](../../parts/04-entrenamiento-eficiente.md)
>
> [⬅️ 🔀 Fusión de sensores](../../labs/12_multimodal_fusion/theory.md) · [🏠 Índice de rutas](../../parts/README.md) · [⚗️ Destilación de conocimiento ➡️](../../labs/14_knowledge_distillation/theory.md)
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

### Por qué la búsqueda aleatoria gana a la malla

El resultado de Bergstra y Bengio suele citarse como una preferencia práctica, y en realidad es un argumento geométrico que se puede seguir con lápiz.

Supóngase un presupuesto de 25 evaluaciones sobre dos hiperparámetros. La **malla** los reparte en 5×5, así que explora exactamente **5 valores distintos** de cada uno: los otros 20 puntos repiten valores ya probados en la otra dimensión. La búsqueda **aleatoria** con las mismas 25 evaluaciones prueba **25 valores distintos** de cada hiperparámetro, porque cada muestra aporta una coordenada nueva en todas las dimensiones a la vez.

Esa diferencia sería irrelevante si todos los hiperparámetros importaran por igual, pero no es el caso: en la práctica el desempeño depende fuertemente de unos pocos —la tasa de aprendizaje casi siempre— y es casi plano en los demás. Con una malla, la resolución sobre el hiperparámetro que sí importa queda limitada a 5 valores, y las otras 20 evaluaciones se gastan variando cosas que no cambian nada. Con muestreo aleatorio, las 25 evaluaciones aportan 25 puntos distintos sobre la dimensión relevante, sea cual sea.

El problema empeora con la dimensión. Una malla de k valores en d hiperparámetros exige k^d evaluaciones: pasar de 2 a 5 hiperparámetros con k = 5 lleva de 25 a 3 125 entrenamientos. Es la **maldición de la dimensionalidad** en su forma más concreta, y la razón de que la malla solo sea razonable con dos o tres parámetros y muy pocos valores.

Un corolario que también importa: la probabilidad de que **ninguna** de n muestras aleatorias caiga en el mejor 5 % del espacio es 0,95ⁿ. Con n = 60 eso es 0,046, es decir, hay un 95 % de probabilidad de encontrar una configuración dentro del mejor 5 % con solo **60 pruebas**, y ese número no depende de cuántas dimensiones haya. Es la garantía que hace defendible la búsqueda aleatoria como opción por defecto.

### El espacio de búsqueda importa tanto como el algoritmo

Un buscador solo puede encontrar lo que hay dentro del rango que se le da, así que definir el espacio es parte del experimento y debe declararse.

La regla más útil es que los hiperparámetros **multiplicativos se muestrean en escala logarítmica**. La tasa de aprendizaje es el caso claro: entre 10⁻⁵ y 10⁻¹ hay cuatro órdenes de magnitud, y muestrear uniformemente en ese intervalo pondría el 90 % de las muestras por encima de 10⁻², dejando prácticamente inexplorada la zona pequeña. Lo correcto es muestrear log₁₀(η) ~ U(−5, −1), que reparte por igual entre órdenes de magnitud. Lo mismo vale para el weight decay y para el tamaño de capa. Los parámetros **aditivos o acotados** —dropout entre 0 y 0,5, número de capas entre 1 y 4— sí se muestrean de forma uniforme.

Hay un fenómeno adicional que conviene anticipar: los hiperparámetros **interactúan**. Tasa de aprendizaje y tamaño de lote están acopladas —al aumentar el lote, el gradiente es menos ruidoso y admite pasos mayores—, y la tasa óptima suele escalar con el lote. Buscar cada uno por separado, fijando el otro, puede dejar fuera el óptimo conjunto. Es la razón de que la búsqueda se haga sobre el espacio completo y no parámetro a parámetro.

### El riesgo real de este laboratorio: sobreajustar la validación

Este es el punto que distingue una búsqueda rigurosa de una que se engaña a sí misma, y es la razón de que el protocolo del repositorio sea especialmente estricto aquí.

Cada configuración probada se evalúa en `validation`, y al final se elige la mejor. Pero elegir el máximo de n estimaciones ruidosas produce un valor **optimista**: es el mismo sesgo de selección de la ruta 10, 𝔼[max] ≥ max 𝔼. Cuanto mayor es n, mayor es el sesgo. Con cien configuraciones probadas, la métrica de validación de la ganadora incluye una porción apreciable de suerte, y **no es una estimación insesgada** de lo que rendirá con datos nuevos.

De ahí se sigue lo que hay que hacer y lo que no. La cifra que se reporta como resultado del laboratorio es la de `test`, medida **una sola vez** con la configuración ya elegida y sellada; el valor de validación de la ganadora se reporta como lo que es, un criterio de selección y no una estimación de desempeño. Y si se quisiera además estimar honestamente el error del procedimiento completo de búsqueda, haría falta una **validación cruzada anidada**: un bucle externo para estimar y uno interno para buscar, con un costo multiplicativo que este laboratorio no asume, pero que conviene saber que existe.

Un último detalle sobre el presupuesto: comparar dos estrategias de búsqueda solo tiene sentido **a igual número de evaluaciones**. Decir que la búsqueda bayesiana batió a la aleatoria es vacío si la primera probó 200 configuraciones y la segunda 20. Y el costo total —tiempo de cómputo acumulado, no solo el del modelo ganador— forma parte del resultado, porque una mejora de dos décimas que costó cien entrenamientos rara vez se justifica.

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
| [🔀 Fusión de sensores](../../labs/12_multimodal_fusion/README.md) | [Las 31 rutas](../../parts/README.md) | [⚗️ Destilación de conocimiento](../../labs/14_knowledge_distillation/README.md) |

**En este laboratorio:** [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🟠 [Parte 4 — Entrenar mejor, más barato y sin centralizar datos](../../parts/04-entrenamiento-eficiente.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/13_hyperparameter_search/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
