# Teoría — RNN para texto

## Propósito

Clasificar sentimiento en reseñas reales usando embeddings y recurrencia.

## Idea central

Este laboratorio estudia **recurrencia sobre secuencias tokenizadas** usando `imdb`, un dataset público real procedente de Hugging Face / Stanford.

## Fundamento matemático

h_t=tanh(W_x x_t + W_h h_{t-1}+b).

La formulación debe conectarse con cuatro elementos: representación de entrada, función del modelo, función de pérdida y regla de actualización. El notebook muestra las dimensiones de los tensores y conserva la misma implementación que el script de terminal.

## Protocolo científico

- Ajustar transformaciones, vocabulario, normalización y selección de variables solo con `train`.
- Usar `validation` para arquitectura, hiperparámetros, checkpoint y umbrales.
- Evaluar `test` una vez, después de congelar las decisiones.
- Comparar contra **TF-IDF + regresión logística**.
- Reportar variación entre semillas e intervalos de confianza; una métrica puntual no expresa toda la incertidumbre.

## Riesgos de interpretación

Reseñas cinematográficas reales con partición oficial.

El dataset refleja su proceso de recolección y no representa automáticamente otros períodos, países o poblaciones. Una asociación predictiva no demuestra causalidad.

## Pregunta crítica

> ¿Qué información se pierde al truncar las reseñas?

## Lecturas

- Fuente del dataset: https://huggingface.co/datasets/stanfordnlp/imdb
- Consulte `docs/experiment-protocol.md`, `docs/reproducibility.md` y `docs/ethics-and-licenses.md`.
