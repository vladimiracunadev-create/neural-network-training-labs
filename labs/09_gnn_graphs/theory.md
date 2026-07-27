# Teoría — GNN sobre red de citas

## Propósito

Clasificar publicaciones científicas usando texto y enlaces de citas.

## Idea central

Este laboratorio estudia **propagación de mensajes sobre grafos** usando `cora`, un dataset público real procedente de PyTorch Geometric / Planetoid.

## Fundamento matemático

H^(l+1)=σ(D^-1/2 Â D^-1/2 H^l W^l).

La formulación debe conectarse con cuatro elementos: representación de entrada, función del modelo, función de pérdida y regla de actualización. El notebook muestra las dimensiones de los tensores y conserva la misma implementación que el script de terminal.

## Protocolo científico

- Ajustar transformaciones, vocabulario, normalización y selección de variables solo con `train`.
- Usar `validation` para arquitectura, hiperparámetros, checkpoint y umbrales.
- Evaluar `test` una vez, después de congelar las decisiones.
- Comparar contra **MLP sin aristas**.
- Reportar variación entre semillas e intervalos de confianza; una métrica puntual no expresa toda la incertidumbre.

## Riesgos de interpretación

Usa las máscaras públicas fijas de train, validación y test.

El dataset refleja su proceso de recolección y no representa automáticamente otros períodos, países o poblaciones. Una asociación predictiva no demuestra causalidad.

## Pregunta crítica

> ¿Cuánto aporta la estructura de citaciones?

## Lecturas

- Fuente del dataset: https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.datasets.Planetoid.html
- Consulte `docs/experiment-protocol.md`, `docs/reproducibility.md` y `docs/ethics-and-licenses.md`.
