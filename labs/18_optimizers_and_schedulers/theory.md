# Teoría — Optimizadores y schedulers

## Propósito

Comparar SGD, Momentum, Adam y reducción de tasa de aprendizaje.

## Idea central

Este laboratorio estudia **comparación controlada de optimizadores y schedulers** usando `california_housing`, un dataset público real procedente de scikit-learn / StatLib.

## Fundamento matemático

Actualizaciones de parámetros y programación de learning rate.

La formulación debe conectarse con cuatro elementos: representación de entrada, función del modelo, función de pérdida y regla de actualización. El notebook muestra las dimensiones de los tensores y conserva la misma implementación que el script de terminal.

## Protocolo científico

- Ajustar transformaciones, vocabulario, normalización y selección de variables solo con `train`.
- Usar `validation` para arquitectura, hiperparámetros, checkpoint y umbrales.
- Evaluar `test` una vez, después de congelar las decisiones.
- Comparar contra **Media y Ridge**.
- Reportar variación entre semillas e intervalos de confianza; una métrica puntual no expresa toda la incertidumbre.

## Riesgos de interpretación

Datos reales del censo de California de 1990.

El dataset refleja su proceso de recolección y no representa automáticamente otros períodos, países o poblaciones. Una asociación predictiva no demuestra causalidad.

## Pregunta crítica

> ¿Cuál mejora más rápido y cuál generaliza mejor?

## Lecturas

- Fuente del dataset: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html
- Consulte `docs/experiment-protocol.md`, `docs/reproducibility.md` y `docs/ethics-and-licenses.md`.
