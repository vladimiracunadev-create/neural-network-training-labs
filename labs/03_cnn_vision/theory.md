# Teoría — CNN para visión

## Propósito

Entrenar una CNN y analizar errores sobre fotografías reales de diez clases.

## Idea central

Este laboratorio estudia **convoluciones para patrones espaciales** usando `cifar10`, un dataset público real procedente de Torchvision / University of Toronto.

## Fundamento matemático

Convolución, pooling, normalización y clasificación.

La formulación debe conectarse con cuatro elementos: representación de entrada, función del modelo, función de pérdida y regla de actualización. El notebook muestra las dimensiones de los tensores y conserva la misma implementación que el script de terminal.

## Protocolo científico

- Ajustar transformaciones, vocabulario, normalización y selección de variables solo con `train`.
- Usar `validation` para arquitectura, hiperparámetros, checkpoint y umbrales.
- Evaluar `test` una vez, después de congelar las decisiones.
- Comparar contra **Clasificador lineal sobre píxeles**.
- Reportar variación entre semillas e intervalos de confianza; una métrica puntual no expresa toda la incertidumbre.

## Riesgos de interpretación

CIFAR-10 contiene 60.000 imágenes reales de 32×32.

El dataset refleja su proceso de recolección y no representa automáticamente otros períodos, países o poblaciones. Una asociación predictiva no demuestra causalidad.

## Pregunta crítica

> ¿Qué clases concentran errores visualmente plausibles?

## Lecturas

- Fuente del dataset: https://www.cs.toronto.edu/~kriz/cifar.html
- Consulte `docs/experiment-protocol.md`, `docs/reproducibility.md` y `docs/ethics-and-licenses.md`.
