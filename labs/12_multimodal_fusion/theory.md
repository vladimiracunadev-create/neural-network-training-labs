# Teoría — Fusión de sensores

## Propósito

Fusionar acelerómetro y giroscopio de smartphones para reconocer actividades.

## Idea central

Este laboratorio estudia **fusión de ramas de sensores** usando `uci_har`, un dataset público real procedente de UCI.

## Fundamento matemático

f=[f_acc(x_acc); f_gyro(x_gyro)]; y=head(f).

La formulación debe conectarse con cuatro elementos: representación de entrada, función del modelo, función de pérdida y regla de actualización. El notebook muestra las dimensiones de los tensores y conserva la misma implementación que el script de terminal.

## Protocolo científico

- Ajustar transformaciones, vocabulario, normalización y selección de variables solo con `train`.
- Usar `validation` para arquitectura, hiperparámetros, checkpoint y umbrales.
- Evaluar `test` una vez, después de congelar las decisiones.
- Comparar contra **Acelerómetro solo, giroscopio solo y regresión logística**.
- Reportar variación entre semillas e intervalos de confianza; una métrica puntual no expresa toda la incertidumbre.

## Riesgos de interpretación

Señales inerciales reales de 30 participantes.

El dataset refleja su proceso de recolección y no representa automáticamente otros períodos, países o poblaciones. Una asociación predictiva no demuestra causalidad.

## Pregunta crítica

> ¿Qué modalidad explica cada actividad?

## Lecturas

- Fuente del dataset: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
- Consulte `docs/experiment-protocol.md`, `docs/reproducibility.md` y `docs/ethics-and-licenses.md`.
