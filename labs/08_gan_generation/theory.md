# Teoría — GAN generativa

## Propósito

Generar prendas a partir de imágenes reales de Fashion-MNIST.

## Idea central

Este laboratorio estudia **aprendizaje adversarial generativo** usando `fashion_mnist`, un dataset público real procedente de Torchvision / Zalando Research.

## Fundamento matemático

min_G max_D E[log D(x)] + E[log(1-D(G(z)))].

La formulación debe conectarse con cuatro elementos: representación de entrada, función del modelo, función de pérdida y regla de actualización. El notebook muestra las dimensiones de los tensores y conserva la misma implementación que el script de terminal.

## Protocolo científico

- Ajustar transformaciones, vocabulario, normalización y selección de variables solo con `train`.
- Usar `validation` para arquitectura, hiperparámetros, checkpoint y umbrales.
- Evaluar `test` una vez, después de congelar las decisiones.
- Comparar contra **PCA generativa y distribución real de referencia**.
- Reportar variación entre semillas e intervalos de confianza; una métrica puntual no expresa toda la incertidumbre.

## Riesgos de interpretación

No usa anillos ni puntos inventados; entrena con prendas reales etiquetadas.

El dataset refleja su proceso de recolección y no representa automáticamente otros períodos, países o poblaciones. Una asociación predictiva no demuestra causalidad.

## Pregunta crítica

> ¿Cómo se distingue diversidad real de ruido visual?

## Lecturas

- Fuente del dataset: https://github.com/zalandoresearch/fashion-mnist
- Consulte `docs/experiment-protocol.md`, `docs/reproducibility.md` y `docs/ethics-and-licenses.md`.
