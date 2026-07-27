# Teoría — Destilación de conocimiento

## Propósito

Transferir conocimiento de una CNN profesora a una estudiante compacta.

## Idea central

Este laboratorio estudia **transferencia de conocimiento profesor-estudiante** usando `cifar10`, un dataset público real procedente de Torchvision / University of Toronto.

## Fundamento matemático

L=α CE(y,s)+(1-α)T² KL(softmax(t/T)||softmax(s/T)).

La formulación debe conectarse con cuatro elementos: representación de entrada, función del modelo, función de pérdida y regla de actualización. El notebook muestra las dimensiones de los tensores y conserva la misma implementación que el script de terminal.

## Protocolo científico

- Ajustar transformaciones, vocabulario, normalización y selección de variables solo con `train`.
- Usar `validation` para arquitectura, hiperparámetros, checkpoint y umbrales.
- Evaluar `test` una vez, después de congelar las decisiones.
- Comparar contra **Estudiante entrenado solo con etiquetas**.
- Reportar variación entre semillas e intervalos de confianza; una métrica puntual no expresa toda la incertidumbre.

## Riesgos de interpretación

Mismo test real para profesor, estudiante base y estudiante destilada.

El dataset refleja su proceso de recolección y no representa automáticamente otros períodos, países o poblaciones. Una asociación predictiva no demuestra causalidad.

## Pregunta crítica

> ¿Qué temperatura equilibra mejor señales duras y blandas?

## Lecturas

- Fuente del dataset: https://www.cs.toronto.edu/~kriz/cifar.html
- Consulte `docs/experiment-protocol.md`, `docs/reproducibility.md` y `docs/ethics-and-licenses.md`.
