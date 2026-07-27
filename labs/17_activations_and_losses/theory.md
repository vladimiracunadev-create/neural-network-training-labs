# Teoría — Activaciones y funciones de pérdida

## Propósito

Comparar ReLU, GELU, Tanh y pérdidas apropiadas en clases desbalanceadas.

## Idea central

Este laboratorio estudia **comparación controlada de activaciones y pérdidas** usando `wine_quality`, un dataset público real procedente de UCI.

## Fundamento matemático

Derivadas, saturación y sensibilidad de CrossEntropy/Focal Loss.

La formulación debe conectarse con cuatro elementos: representación de entrada, función del modelo, función de pérdida y regla de actualización. El notebook muestra las dimensiones de los tensores y conserva la misma implementación que el script de terminal.

## Protocolo científico

- Ajustar transformaciones, vocabulario, normalización y selección de variables solo con `train`.
- Usar `validation` para arquitectura, hiperparámetros, checkpoint y umbrales.
- Evaluar `test` una vez, después de congelar las decisiones.
- Comparar contra **Regresión ordinal y Random Forest**.
- Reportar variación entre semillas e intervalos de confianza; una métrica puntual no expresa toda la incertidumbre.

## Riesgos de interpretación

Muestras reales de vinho verde con análisis fisicoquímico y evaluación sensorial.

El dataset refleja su proceso de recolección y no representa automáticamente otros períodos, países o poblaciones. Una asociación predictiva no demuestra causalidad.

## Pregunta crítica

> ¿La conclusión se mantiene en varias semillas?

## Lecturas

- Fuente del dataset: https://archive.ics.uci.edu/dataset/186/wine+quality
- Consulte `docs/experiment-protocol.md`, `docs/reproducibility.md` y `docs/ethics-and-licenses.md`.
