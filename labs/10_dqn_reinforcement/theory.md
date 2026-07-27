# Teoría — DQN para inventario con demanda real

## Propósito

Aprender una política de reposición usando una secuencia de demanda observada en transacciones reales.

## Idea central

Este laboratorio estudia **valor de acciones con demanda histórica** usando `online_retail`, un dataset público real procedente de UCI.

## Fundamento matemático

y=r+γ max_a Q_target(s′,a); la demanda de cada paso proviene del historial real, no de un generador.

La formulación debe conectarse con cuatro elementos: representación de entrada, función del modelo, función de pérdida y regla de actualización. El notebook muestra las dimensiones de los tensores y conserva la misma implementación que el script de terminal.

## Protocolo científico

- Ajustar transformaciones, vocabulario, normalización y selección de variables solo con `train`.
- Usar `validation` para arquitectura, hiperparámetros, checkpoint y umbrales.
- Evaluar `test` una vez, después de congelar las decisiones.
- Comparar contra **Política de reposición periódica basada en demanda media histórica**.
- Reportar variación entre semillas e intervalos de confianza; una métrica puntual no expresa toda la incertidumbre.

## Riesgos de interpretación

La dinámica de inventario es un entorno educativo, pero la demanda diaria se construye exclusivamente desde transacciones reales de Online Retail.

El dataset refleja su proceso de recolección y no representa automáticamente otros períodos, países o poblaciones. Una asociación predictiva no demuestra causalidad.

## Pregunta crítica

> ¿La política es robusta a cambios en costo y demanda?

## Lecturas

- Fuente del dataset: https://archive.ics.uci.edu/dataset/352/online+retail
- Consulte `docs/experiment-protocol.md`, `docs/reproducibility.md` y `docs/ethics-and-licenses.md`.
