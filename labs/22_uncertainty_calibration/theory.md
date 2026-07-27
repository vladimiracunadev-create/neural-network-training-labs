# Teoría — Incertidumbre y calibración

## Propósito

Medir confianza, Brier score, ECE y temperature scaling.

## Idea central

Este laboratorio estudia **calibración probabilística** usando `breast_cancer_wisconsin`, un dataset público real procedente de UCI.

Un clasificador no solo decide una clase: también emite una **confianza**, la probabilidad que asigna a esa decisión. Un modelo está *calibrado* cuando esa confianza coincide con la frecuencia real de acierto: de todas las predicciones hechas con 80 % de confianza, aproximadamente el 80 % deberían ser correctas. La exactitud responde "¿acierta?"; la calibración responde "¿son creíbles sus probabilidades?", y son cosas distintas: una red puede acertar mucho y aun así ser sistemáticamente *sobreconfiada*, gritando 99 % cuando debería decir 70 %.

Esta distinción es crítica cuando la probabilidad alimenta una decisión posterior —fijar un umbral, priorizar un caso, cuantificar riesgo—. En un dataset de diagnóstico como `breast_cancer_wisconsin`, una confianza mal calibrada puede inducir una falsa sensación de certeza. El laboratorio mide la calidad probabilística con Brier score y ECE, y corrige la sobreconfianza con *temperature scaling*, ajustado en validación y evaluado una sola vez en test.

## Fundamento matemático

Calibrar logits z/T en validación y evaluar una vez en test.

Una red de clasificación produce **logits** z (puntuaciones sin normalizar) que se convierten en probabilidades con softmax: pₖ = e^{zₖ} / Σⱼ e^{zⱼ}. Las redes profundas modernas tienden a la **sobreconfianza**: el entrenamiento con entropía cruzada empuja los logits a valores extremos —porque acercarse a probabilidad 1 en la clase correcta reduce la pérdida indefinidamente— y el resultado son probabilidades más agudas de lo que la evidencia justifica. Calibrar no cambia qué clase se predice; cambia cuán *afiladas* son las probabilidades.

El **Brier score** mide el error cuadrático medio entre la probabilidad predicha y el resultado real: BS = (1/N)·Σₙ Σₖ (p_{n,k} − y_{n,k})², donde y es el vector one-hot de la etiqueta. Penaliza a la vez errores de clasificación y de confianza: predecir 0.9 en la clase correcta cuesta menos que predecir 0.6, pero predecir 0.9 en la clase *equivocada* cuesta mucho. Es una *proper scoring rule*: se minimiza reportando las probabilidades verdaderas.

El **Expected Calibration Error (ECE)** cuantifica directamente el desajuste entre confianza y acierto. Se agrupan las predicciones en B intervalos según su confianza; en cada intervalo b se comparan la exactitud observada acc(b) y la confianza media conf(b), y se promedia la brecha ponderando por el número de ejemplos: ECE = Σ_b (|Bᵦ|/N) · |acc(b) − conf(b)|. Un modelo perfectamente calibrado tiene ECE = 0; un valor alto revela sobreconfianza (conf > acc) o subconfianza (conf < acc), visible en el *reliability diagram*.

El **temperature scaling** es la corrección más simple y efectiva: divide todos los logits por un único escalar T > 0 antes del softmax, p̂ₖ = e^{zₖ/T} / Σⱼ e^{zⱼ/T}. Con T > 1 las probabilidades se suavizan (baja la confianza), con T < 1 se agudizan; T = 1 no cambia nada. Como el mismo T multiplica todos los logits, el orden relativo se conserva y por tanto **la exactitud y el ranking (AUC) no cambian**: solo se recalibra la confianza. El valor óptimo T\* se ajusta minimizando la entropía cruzada (o el NLL) sobre el conjunto de **validación**, nunca sobre test —ajustar sobre test contaminaría la evaluación—. Luego se evalúan Brier y ECE una sola vez en test con ese T\* congelado. Es importante entender que temperature scaling captura la incertidumbre *aleatórica* (ruido inherente); no distingue lo que el modelo *no sabe* (incertidumbre epistémica), para lo cual se recurre a MC Dropout o ensambles.

## Protocolo científico

- Ajustar transformaciones, vocabulario, normalización y selección de variables solo con `train`.
- Usar `validation` para arquitectura, hiperparámetros, checkpoint y umbrales.
- Evaluar `test` una vez, después de congelar las decisiones.
- Comparar contra **Regresión logística calibrada**.
- Reportar variación entre semillas e intervalos de confianza; una métrica puntual no expresa toda la incertidumbre.

## Riesgos de interpretación

No constituye una herramienta clínica ni consejo médico.

El dataset refleja su proceso de recolección y no representa automáticamente otros períodos, países o poblaciones. Una asociación predictiva no demuestra causalidad.

## Pregunta crítica

> ¿Una mayor accuracy implica probabilidades confiables?

## 🔗 Referencias

> Las referencias apuntan a las obras; no se reproduce su contenido, la redacción es original.

- Guo et al. (2017), *On Calibration of Modern Neural Networks*, ICML — evidencia de la sobreconfianza de las redes profundas y propuesta de temperature scaling; define ECE y reliability diagrams.
- Gal y Ghahramani (2016), *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning (MC Dropout)*, ICML — estimación de incertidumbre epistémica manteniendo dropout activo en inferencia.
- Lakshminarayanan, Pritzel y Blundell (2017), *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*, NeurIPS — ensambles como estimador robusto de incertidumbre predictiva.
- Fuente del dataset: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- Consulte `docs/experiment-protocol.md`, `docs/reproducibility.md` y `docs/ethics-and-licenses.md`.
