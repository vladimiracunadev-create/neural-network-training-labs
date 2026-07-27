# Teoría — MLP multiclase

## Propósito

Resolver clasificación no lineal con capas densas, activaciones y regularización.

## Idea central

Este laboratorio estudia **red multicapa para relaciones no lineales** usando `dry_bean`, un dataset público real procedente de UCI.

## Fundamento matemático

h=ReLU(xW1+b1); logits=hW2+b2.

La formulación debe conectarse con cuatro elementos: representación de entrada, función del modelo, función de pérdida y regla de actualización. El notebook muestra las dimensiones de los tensores y conserva la misma implementación que el script de terminal.

## Protocolo científico

- Ajustar transformaciones, vocabulario, normalización y selección de variables solo con `train`.
- Usar `validation` para arquitectura, hiperparámetros, checkpoint y umbrales.
- Evaluar `test` una vez, después de congelar las decisiones.
- Comparar contra **Regresión logística multinomial y Random Forest**.
- Reportar variación entre semillas e intervalos de confianza; una métrica puntual no expresa toda la incertidumbre.

## Riesgos de interpretación

13.611 granos de siete variedades reales y 16 atributos de forma.

El dataset refleja su proceso de recolección y no representa automáticamente otros períodos, países o poblaciones. Una asociación predictiva no demuestra causalidad.

## Pregunta crítica

> ¿La complejidad adicional supera de forma estable a la línea base?

## Lecturas

- Fuente del dataset: https://archive.ics.uci.edu/dataset/602/dry+bean+dataset
- Consulte `docs/experiment-protocol.md`, `docs/reproducibility.md` y `docs/ethics-and-licenses.md`.
