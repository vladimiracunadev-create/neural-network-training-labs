# Teoría — Incertidumbre y calibración

## Propósito

Medir confianza, Brier score, ECE y temperature scaling.

## Idea central

Este laboratorio estudia **calibración probabilística** usando `breast_cancer_wisconsin`, un dataset público real procedente de UCI.

## Fundamento matemático

Calibrar logits z/T en validación y evaluar una vez en test.

La formulación debe conectarse con cuatro elementos: representación de entrada, función del modelo, función de pérdida y regla de actualización. El notebook muestra las dimensiones de los tensores y conserva la misma implementación que el script de terminal.

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

## Lecturas

- Fuente del dataset: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- Consulte `docs/experiment-protocol.md`, `docs/reproducibility.md` y `docs/ethics-and-licenses.md`.
