# DQN para inventario con demanda real

<!-- nav-top -->
> 🧭 **Ruta 11 / 31** · 🟣 [Parte 3 — Familias especializadas: generar, decidir, relacionar](../../parts/03-familias-especializadas.md)
>
> [⬅️ 🕸️ GNN sobre red de citas](../../labs/09_gnn_graphs/README.md) · [🏠 Índice de rutas](../../parts/README.md) · [♻️ Transfer learning con mascotas ➡️](../../labs/11_transfer_learning/README.md)
>
> **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

<!-- ficha -->
## 📋 Ficha del laboratorio

![ruta](https://img.shields.io/badge/ruta-11%20de%2031-7c5cff?style=flat-square) ![nivel](https://img.shields.io/badge/nivel-avanzado-8957e5?style=flat-square) ![categoría](https://img.shields.io/badge/categoría-Central-2e8b57?style=flat-square) ![horas](https://img.shields.io/badge/horas-~8%20h-f0b429?style=flat-square) ![dataset](https://img.shields.io/badge/dataset-online__retail-1f6feb?style=flat-square) ![selección](https://img.shields.io/badge/selección-mean__return-8957e5?style=flat-square)

| Campo | Valor |
|---|---|
| 🧭 Posición | Ruta **11 de 31** del recorrido · categoría central |
| 🎚️ Nivel | avanzado |
| ⏱️ Dedicación estimada | 8 horas |
| 🧩 Tarea | `reinforcement_learning` |
| 🏗️ Arquitectura | `dqn_inventory` |
| 🗄️ Dataset | [`online_retail`](https://archive.ics.uci.edu/dataset/352/online+retail) — UCI |
| ⚖️ Licencia del dataset | CC BY 4.0 |
| 🎯 Métrica de selección | `mean_return` sobre `validation` |
| 📏 Línea base a superar | Política de reposición periódica basada en demanda media histórica |
| 🔒 Política de `test` | se abre una sola vez, tras escribir `experiment.lock.json` |

### 🎯 Qué vas a poder hacer al terminar

- Aprender una política de reposición usando una secuencia de demanda observada en transacciones reales.
- Preparar y auditar el dataset real online_retail sin fuga de datos.
- Entrenar y evaluar valor de acciones con demanda histórica.
- Comparar contra la línea base: Política de reposición periódica basada en demanda media histórica.
- Interpretar intervalos de confianza, errores y limitaciones.

### 🧩 Prerrequisitos

- PyTorch intermedio
- optimización
- lectura de artículos técnicos

> Si alguno te falta, retrocede antes de continuar. Viniendo de [🕸️ GNN sobre red de citas](../../labs/09_gnn_graphs/README.md).

### ⚙️ `baseline` frente a `improved`

| Parámetro | [`baseline.yaml`](configs/baseline.yaml) | [`improved.yaml`](configs/improved.yaml) |
|---|---|---|
| Épocas | `20` | `50` |
| Tasa de aprendizaje | `0.001` | `0.0005` |
| Paciencia (early stopping) | `5` | `8` |
| Precisión mixta (AMP) | no | sí |
| Procesos de carga | `0` | `2` |

> Solo se muestran los parámetros en los que ambas configuraciones difieren. La elección entre una y otra se decide con `validation`, nunca con `test`.

### 📦 Entregables y criterios de aceptación

**Entregables**

- notebook ejecutado
- reporte experimental
- model card
- comparación con línea base
- respuesta a preguntas críticas

**Criterios de éxito**

- cero solapamiento entre train, validation y test
- selección basada únicamente en validation
- métricas finales acompañadas por incertidumbre
- conclusiones que distinguen evidencia de suposición

### 🗂️ Recursos del laboratorio

| Recurso | Archivo |
|---|---|
| 🧠 Teoría y referencias | [`theory.md`](theory.md) |
| 🔬 Plan de experimentos | [`experiments.md`](experiments.md) |
| 📝 Evaluación y rúbrica | [`assessment.md`](assessment.md) |
| 📓 Notebook de recorrido | [`notebook.ipynb`](notebook.ipynb) |
| ✏️ Notebook de estudiante | [`notebook_student.ipynb`](notebook_student.ipynb) |
| ✅ Notebook de solución | [`notebook_solution.ipynb`](notebook_solution.ipynb) |
| 🖥️ Script de terminal | [`train.py`](train.py) |
| 🎛️ Configuración base | [`configs/baseline.yaml`](configs/baseline.yaml) |
| 🎚️ Configuración ampliada | [`configs/improved.yaml`](configs/improved.yaml) |
| 🗄️ Ficha del dataset | [`data/dataset.yaml`](data/dataset.yaml) |
| 🧾 Metadatos de la lección | [`lesson.yaml`](lesson.yaml) |

<!-- /ficha -->

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
## 🧭 Navegación del recorrido

| ⬅️ Laboratorio anterior | 🏠 Índice | Laboratorio siguiente ➡️ |
|---|:---:|---|
| [🕸️ GNN sobre red de citas](../../labs/09_gnn_graphs/README.md) | [Las 31 rutas](../../parts/README.md) | [♻️ Transfer learning con mascotas](../../labs/11_transfer_learning/README.md) |

**En este laboratorio:** **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🟣 [Parte 3 — Familias especializadas: generar, decidir, relacionar](../../parts/03-familias-especializadas.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/10_dqn_reinforcement/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
