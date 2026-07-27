# Teoría — Proyecto final: churn de telecomunicaciones

## Propósito

Resolver de extremo a extremo un problema real de abandono de clientes con documentación, evaluación y despliegue.

## Idea central

Este laboratorio estudia **proyecto integral de churn** usando `iranian_churn`, un dataset público real procedente de UCI.

## Fundamento matemático

Clasificación, calibración, selección de umbral y costo de errores.

La formulación debe conectarse con cuatro elementos: representación de entrada, función del modelo, función de pérdida y regla de actualización. El notebook muestra las dimensiones de los tensores y conserva la misma implementación que el script de terminal.

## Protocolo científico

- Ajustar transformaciones, vocabulario, normalización y selección de variables solo con `train`.
- Usar `validation` para arquitectura, hiperparámetros, checkpoint y umbrales.
- Evaluar `test` una vez, después de congelar las decisiones.
- Comparar contra **Regresión logística y Gradient Boosting**.
- Reportar variación entre semillas e intervalos de confianza; una métrica puntual no expresa toda la incertidumbre.

## Riesgos de interpretación

3.150 clientes recolectados aleatoriamente de la base de una empresa iraní de telecomunicaciones durante 12 meses.

El dataset refleja su proceso de recolección y no representa automáticamente otros períodos, países o poblaciones. Una asociación predictiva no demuestra causalidad.

## Pregunta crítica

> ¿Cómo convertir resultados en una decisión responsable?

## Lecturas

- Fuente del dataset: https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset
- Consulte `docs/experiment-protocol.md`, `docs/reproducibility.md` y `docs/ethics-and-licenses.md`.
