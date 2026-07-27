# Teoría — Búsqueda de hiperparámetros

## Propósito

Optimizar profundidad, ancho, dropout y learning rate sin tocar test.

## Idea central

Este laboratorio estudia **búsqueda de hiperparámetros sin tocar test** usando `adult_census`, un dataset público real procedente de UCI.

## Fundamento matemático

Selección por validación; test se usa solo tras elegir la mejor prueba.

La formulación debe conectarse con cuatro elementos: representación de entrada, función del modelo, función de pérdida y regla de actualización. El notebook muestra las dimensiones de los tensores y conserva la misma implementación que el script de terminal.

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

## Lecturas

- Fuente del dataset: https://archive.ics.uci.edu/dataset/2/adult
- Consulte `docs/experiment-protocol.md`, `docs/reproducibility.md` y `docs/ethics-and-licenses.md`.
