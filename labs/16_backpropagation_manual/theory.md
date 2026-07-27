# Teoría — Backpropagation manual

## Propósito

Derivar y programar backpropagation en una MLP pequeña.

## Idea central

Este laboratorio estudia **backpropagation manual** usando `iris`, un dataset público real procedente de UCI.

## Fundamento matemático

Regla de la cadena para W2, b2, W1 y b1.

La formulación debe conectarse con cuatro elementos: representación de entrada, función del modelo, función de pérdida y regla de actualización. El notebook muestra las dimensiones de los tensores y conserva la misma implementación que el script de terminal.

## Protocolo científico

- Ajustar transformaciones, vocabulario, normalización y selección de variables solo con `train`.
- Usar `validation` para arquitectura, hiperparámetros, checkpoint y umbrales.
- Evaluar `test` una vez, después de congelar las decisiones.
- Comparar contra **Regresión logística multinomial**.
- Reportar variación entre semillas e intervalos de confianza; una métrica puntual no expresa toda la incertidumbre.

## Riesgos de interpretación

150 mediciones botánicas reales de tres especies de Iris.

El dataset refleja su proceso de recolección y no representa automáticamente otros períodos, países o poblaciones. Una asociación predictiva no demuestra causalidad.

## Pregunta crítica

> ¿Dónde aparecen gradientes que explotan o desaparecen?

## Lecturas

- Fuente del dataset: https://archive.ics.uci.edu/dataset/53/iris
- Consulte `docs/experiment-protocol.md`, `docs/reproducibility.md` y `docs/ethics-and-licenses.md`.
