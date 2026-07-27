# DQN para inventario con demanda real

<!-- nav-top -->
> 🧭 [⬅️ Anterior](../../labs/09_gnn_graphs/README.md) · [🏠 Índice](../../README.md#laboratorios) · [Siguiente ➡️](../../labs/11_transfer_learning/README.md)
<!-- /nav-top -->

## Objetivo

Aprender una política de reposición usando una secuencia de demanda observada en transacciones reales.

## Dataset real

- **Dataset:** `online_retail`
- **Fuente:** UCI
- **Referencia:** https://archive.ics.uci.edu/dataset/352/online+retail
- **Licencia:** CC BY 4.0

La dinámica de inventario es un entorno educativo, pero la demanda diaria se construye exclusivamente desde transacciones reales de Online Retail.

## Diseño

La serie diaria se divide cronológicamente. La política aprende con `train`, se selecciona con `validation` y se evalúa una sola vez sobre `test`. El estado contiene inventario, demanda reciente y posición temporal; las acciones son cantidades discretas de reposición.

## Línea base

Política de reposición periódica basada en demanda media histórica.

## Ejecución

```bash
python labs/10_dqn_reinforcement/train.py --quick
python labs/10_dqn_reinforcement/train.py --config improved
```

## Métricas

`mean_return`, `stockout_rate`, `holding_cost` y `service_level`.

## Límites

El historial de demanda es real. Los costos y reglas de inventario son parámetros educativos y deben sustituirse por costos de negocio antes de cualquier uso operacional.


## Material formativo v3

- [`theory.md`](theory.md): fundamento, protocolo y riesgos de interpretación.
- [`experiments.md`](experiments.md): hipótesis, variables controladas y tabla multi-semilla.
- [`assessment.md`](assessment.md): preguntas y rúbrica de evaluación.
- [`lesson.yaml`](lesson.yaml): resultados de aprendizaje, prerrequisitos y entregables.

## Comandos profesionales

```bash
neural-labs quality --lab 10_dqn_reinforcement --quick
neural-labs benchmark --lab 10_dqn_reinforcement --quick --split-seed 42 --training-seeds 41 42 43
neural-labs leaderboard
```

## Sellado del experimento

La partición se controla con `split_seed`; la inicialización y el entrenamiento con `training_seed`. El conjunto `test` se abre solamente después de seleccionar el checkpoint mediante validación y escribir `experiment.lock.json`.

<!-- nav-bottom -->
## 🧭 Navegación del curso

| ⬅️ Anterior | Siguiente ➡️ |
|---|---|
| [🕸️ GNN sobre red de citas](../../labs/09_gnn_graphs/README.md) | [♻️ Transfer learning con mascotas](../../labs/11_transfer_learning/README.md) |

[🏠 Portada del repositorio](../../README.md) · [🌐 Ver en el sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/10_dqn_reinforcement/index.html)
<!-- /nav-bottom -->
