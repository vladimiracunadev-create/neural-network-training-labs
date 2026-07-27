# Teoría — Regularización

## Propósito

Medir dropout, weight decay y batch normalization.

## Idea central

Este laboratorio estudia **dropout, batch normalization y weight decay** usando `fashion_mnist`, un dataset público real procedente de Torchvision / Zalando Research.

## Fundamento matemático

Regularización explícita e implícita; brecha train-validation.

La formulación debe conectarse con cuatro elementos: representación de entrada, función del modelo, función de pérdida y regla de actualización. El notebook muestra las dimensiones de los tensores y conserva la misma implementación que el script de terminal.

## Protocolo científico

- Ajustar transformaciones, vocabulario, normalización y selección de variables solo con `train`.
- Usar `validation` para arquitectura, hiperparámetros, checkpoint y umbrales.
- Evaluar `test` una vez, después de congelar las decisiones.
- Comparar contra **MLP sin regularización**.
- Reportar variación entre semillas e intervalos de confianza; una métrica puntual no expresa toda la incertidumbre.

## Riesgos de interpretación

Prendas reales normalizadas en 28×28.

El dataset refleja su proceso de recolección y no representa automáticamente otros períodos, países o poblaciones. Una asociación predictiva no demuestra causalidad.

## Pregunta crítica

> ¿Qué técnica reduce sobreajuste sin subajustar?

## Lecturas

- Fuente del dataset: https://github.com/zalandoresearch/fashion-mnist
- Consulte `docs/experiment-protocol.md`, `docs/reproducibility.md` y `docs/ethics-and-licenses.md`.
