# Teoría — Perceptrón con PyTorch

## Propósito

Aprender tensores, autograd, optimizadores y un clasificador lineal.

## Idea central

Este laboratorio estudia **clasificador lineal con autograd** usando `banknote_authentication`, un dataset público real procedente de UCI.

## Fundamento matemático

z=xW+b; BCEWithLogitsLoss.

La formulación debe conectarse con cuatro elementos: representación de entrada, función del modelo, función de pérdida y regla de actualización. El notebook muestra las dimensiones de los tensores y conserva la misma implementación que el script de terminal.

## Protocolo científico

- Ajustar transformaciones, vocabulario, normalización y selección de variables solo con `train`.
- Usar `validation` para arquitectura, hiperparámetros, checkpoint y umbrales.
- Evaluar `test` una vez, después de congelar las decisiones.
- Comparar contra **Regresión logística**.
- Reportar variación entre semillas e intervalos de confianza; una métrica puntual no expresa toda la incertidumbre.

## Riesgos de interpretación

Características extraídas de imágenes reales de billetes.

El dataset refleja su proceso de recolección y no representa automáticamente otros períodos, países o poblaciones. Una asociación predictiva no demuestra causalidad.

## Pregunta crítica

> ¿Qué ejemplos no puede separar un único hiperplano?

## Lecturas

- Fuente del dataset: https://archive.ics.uci.edu/dataset/267/banknote+authentication
- Consulte `docs/experiment-protocol.md`, `docs/reproducibility.md` y `docs/ethics-and-licenses.md`.
