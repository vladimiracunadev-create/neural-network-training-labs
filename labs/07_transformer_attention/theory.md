# Teoría — Transformer para noticias

## Propósito

Aplicar atención multi-cabeza a clasificación de noticias reales.

## Idea central

Este laboratorio estudia **autoatención para clasificación de texto** usando `ag_news`, un dataset público real procedente de Hugging Face.

## Fundamento matemático

Attention(Q,K,V)=softmax(QKᵀ/√d)V.

La formulación debe conectarse con cuatro elementos: representación de entrada, función del modelo, función de pérdida y regla de actualización. El notebook muestra las dimensiones de los tensores y conserva la misma implementación que el script de terminal.

## Protocolo científico

- Ajustar transformaciones, vocabulario, normalización y selección de variables solo con `train`.
- Usar `validation` para arquitectura, hiperparámetros, checkpoint y umbrales.
- Evaluar `test` una vez, después de congelar las decisiones.
- Comparar contra **TF-IDF + regresión logística**.
- Reportar variación entre semillas e intervalos de confianza; una métrica puntual no expresa toda la incertidumbre.

## Riesgos de interpretación

Noticias reales en cuatro categorías con particiones públicas.

El dataset refleja su proceso de recolección y no representa automáticamente otros períodos, países o poblaciones. Una asociación predictiva no demuestra causalidad.

## Pregunta crítica

> ¿La atención observada coincide con evidencia útil para la clase?

## Lecturas

- Fuente del dataset: https://huggingface.co/datasets/fancyzhx/ag_news
- Consulte `docs/experiment-protocol.md`, `docs/reproducibility.md` y `docs/ethics-and-licenses.md`.
