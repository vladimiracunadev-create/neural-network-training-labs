# Teoría — Transfer learning con mascotas

## Propósito

Comparar extracción de características, fine-tuning y entrenamiento desde cero.

## Idea central

Este laboratorio estudia **reutilización de representaciones preentrenadas** usando `oxford_iiit_pet`, un dataset público real procedente de Torchvision / Oxford.

## Fundamento matemático

Inicializar con pesos ImageNet y ajustar capas seleccionadas.

La formulación debe conectarse con cuatro elementos: representación de entrada, función del modelo, función de pérdida y regla de actualización. El notebook muestra las dimensiones de los tensores y conserva la misma implementación que el script de terminal.

## Protocolo científico

- Ajustar transformaciones, vocabulario, normalización y selección de variables solo con `train`.
- Usar `validation` para arquitectura, hiperparámetros, checkpoint y umbrales.
- Evaluar `test` una vez, después de congelar las decisiones.
- Comparar contra **CNN pequeña entrenada desde cero**.
- Reportar variación entre semillas e intervalos de confianza; una métrica puntual no expresa toda la incertidumbre.

## Riesgos de interpretación

7.349 imágenes reales de 37 razas de perros y gatos.

El dataset refleja su proceso de recolección y no representa automáticamente otros períodos, países o poblaciones. Una asociación predictiva no demuestra causalidad.

## Pregunta crítica

> ¿Cuándo el preentrenamiento deja de aportar?

## Lecturas

- Fuente del dataset: https://www.robots.ox.ac.uk/~vgg/data/pets/
- Consulte `docs/experiment-protocol.md`, `docs/reproducibility.md` y `docs/ethics-and-licenses.md`.
