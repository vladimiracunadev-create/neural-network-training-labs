# Teoría — Aprendizaje federado por participante

## Propósito

Aplicar FedAvg usando participantes reales como clientes naturales.

## Idea central

Este laboratorio estudia **agregación federada de clientes reales** usando `uci_har_subjects`, un dataset público real procedente de UCI.

## Fundamento matemático

w_{t+1}=Σ_k(n_k/n)w_k.

La formulación debe conectarse con cuatro elementos: representación de entrada, función del modelo, función de pérdida y regla de actualización. El notebook muestra las dimensiones de los tensores y conserva la misma implementación que el script de terminal.

## Protocolo científico

- Ajustar transformaciones, vocabulario, normalización y selección de variables solo con `train`.
- Usar `validation` para arquitectura, hiperparámetros, checkpoint y umbrales.
- Evaluar `test` una vez, después de congelar las decisiones.
- Comparar contra **Entrenamiento centralizado**.
- Reportar variación entre semillas e intervalos de confianza; una métrica puntual no expresa toda la incertidumbre.

## Riesgos de interpretación

No crea clientes espaciales artificiales; conserva identificadores reales de sujetos.

El dataset refleja su proceso de recolección y no representa automáticamente otros períodos, países o poblaciones. Una asociación predictiva no demuestra causalidad.

## Pregunta crítica

> ¿Qué clientes quedan perjudicados por la agregación?

## Lecturas

- Fuente del dataset: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
- Consulte `docs/experiment-protocol.md`, `docs/reproducibility.md` y `docs/ethics-and-licenses.md`.
