# Teoría — LSTM para series temporales

## Propósito

Pronosticar demanda horaria respetando el orden temporal.

## Idea central

Este laboratorio estudia **memoria recurrente para pronóstico temporal** usando `seoul_bike`, un dataset público real procedente de UCI.

## Fundamento matemático

Puertas input, forget y output de una LSTM.

La formulación debe conectarse con cuatro elementos: representación de entrada, función del modelo, función de pérdida y regla de actualización. El notebook muestra las dimensiones de los tensores y conserva la misma implementación que el script de terminal.

## Protocolo científico

- Ajustar transformaciones, vocabulario, normalización y selección de variables solo con `train`.
- Usar `validation` para arquitectura, hiperparámetros, checkpoint y umbrales.
- Evaluar `test` una vez, después de congelar las decisiones.
- Comparar contra **Persistencia, media móvil y Ridge**.
- Reportar variación entre semillas e intervalos de confianza; una métrica puntual no expresa toda la incertidumbre.

## Riesgos de interpretación

8.760 observaciones reales de arriendo de bicicletas y clima en Seúl.

El dataset refleja su proceso de recolección y no representa automáticamente otros períodos, países o poblaciones. Una asociación predictiva no demuestra causalidad.

## Pregunta crítica

> ¿El modelo supera persistencia en períodos de cambio?

## Lecturas

- Fuente del dataset: https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand
- Consulte `docs/experiment-protocol.md`, `docs/reproducibility.md` y `docs/ethics-and-licenses.md`.
