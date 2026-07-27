# Teoría — Exportación e inferencia

## Propósito

Exportar ONNX, validar paridad y medir latencia por lotes.

## Idea central

Este laboratorio estudia **exportación y perfil de inferencia** usando `cifar10`, un dataset público real procedente de Torchvision / University of Toronto.

## Fundamento matemático

Paridad numérica y costo de inferencia.

La formulación debe conectarse con cuatro elementos: representación de entrada, función del modelo, función de pérdida y regla de actualización. El notebook muestra las dimensiones de los tensores y conserva la misma implementación que el script de terminal.

## Protocolo científico

- Ajustar transformaciones, vocabulario, normalización y selección de variables solo con `train`.
- Usar `validation` para arquitectura, hiperparámetros, checkpoint y umbrales.
- Evaluar `test` una vez, después de congelar las decisiones.
- Comparar contra **PyTorch eager**.
- Reportar variación entre semillas e intervalos de confianza; una métrica puntual no expresa toda la incertidumbre.

## Riesgos de interpretación

Incluye predicción, exportación y benchmark reproducible.

El dataset refleja su proceso de recolección y no representa automáticamente otros períodos, países o poblaciones. Una asociación predictiva no demuestra causalidad.

## Pregunta crítica

> ¿Qué compromisos existen entre tamaño, latencia y precisión?

## Lecturas

- Fuente del dataset: https://www.cs.toronto.edu/~kriz/cifar.html
- Consulte `docs/experiment-protocol.md`, `docs/reproducibility.md` y `docs/ethics-and-licenses.md`.
