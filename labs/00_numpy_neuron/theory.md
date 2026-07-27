# Teoría — Neurona con NumPy

## Propósito

Implementar propagación, entropía cruzada y descenso de gradiente sin autograd.

## Idea central

Este laboratorio estudia **regresión logística implementada sin autograd** usando `breast_cancer_wisconsin`, un dataset público real procedente de UCI.

## Fundamento matemático

p(y=1|x)=σ(xw+b); gradiente de la entropía cruzada.

La formulación debe conectarse con cuatro elementos: representación de entrada, función del modelo, función de pérdida y regla de actualización. El notebook muestra las dimensiones de los tensores y conserva la misma implementación que el script de terminal.

## Protocolo científico

- Ajustar transformaciones, vocabulario, normalización y selección de variables solo con `train`.
- Usar `validation` para arquitectura, hiperparámetros, checkpoint y umbrales.
- Evaluar `test` una vez, después de congelar las decisiones.
- Comparar contra **DummyClassifier y regresión logística de scikit-learn**.
- Reportar variación entre semillas e intervalos de confianza; una métrica puntual no expresa toda la incertidumbre.

## Riesgos de interpretación

Datos clínicos reales derivados de imágenes digitalizadas de aspirados de masas mamarias.

El dataset refleja su proceso de recolección y no representa automáticamente otros períodos, países o poblaciones. Una asociación predictiva no demuestra causalidad.

## Pregunta crítica

> ¿Cómo cambia la convergencia al modificar la escala de las variables?

## Lecturas

- Fuente del dataset: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- Consulte `docs/experiment-protocol.md`, `docs/reproducibility.md` y `docs/ethics-and-licenses.md`.
