# Teoría — Autoencoder para fraude

## Propósito

Detectar transacciones fraudulentas mediante error de reconstrucción.

## Idea central

Este laboratorio estudia **reconstrucción para detección de anomalías** usando `credit_card_fraud`, un dataset público real procedente de Kaggle / ULB.

## Fundamento matemático

Minimizar ||x-decoder(encoder(x))||² sobre transacciones normales.

La formulación debe conectarse con cuatro elementos: representación de entrada, función del modelo, función de pérdida y regla de actualización. El notebook muestra las dimensiones de los tensores y conserva la misma implementación que el script de terminal.

## Protocolo científico

- Ajustar transformaciones, vocabulario, normalización y selección de variables solo con `train`.
- Usar `validation` para arquitectura, hiperparámetros, checkpoint y umbrales.
- Evaluar `test` una vez, después de congelar las decisiones.
- Comparar contra **Isolation Forest**.
- Reportar variación entre semillas e intervalos de confianza; una métrica puntual no expresa toda la incertidumbre.

## Riesgos de interpretación

284.807 transacciones reales; el laboratorio evita reequilibrar el conjunto de test.

El dataset refleja su proceso de recolección y no representa automáticamente otros períodos, países o poblaciones. Una asociación predictiva no demuestra causalidad.

## Pregunta crítica

> ¿Qué costo tiene priorizar recall frente a precision?

## Lecturas

- Fuente del dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Consulte `docs/experiment-protocol.md`, `docs/reproducibility.md` y `docs/ethics-and-licenses.md`.
