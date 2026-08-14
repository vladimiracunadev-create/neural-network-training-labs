# Plan de experimentos — GNN sobre red de citas

<!-- nav-top -->
> 🧭 **Ruta 10 / 31** · 🟣 [Parte 3 — Familias especializadas: generar, decidir, relacionar](../../parts/03-familias-especializadas.md)
>
> [⬅️ 🎨 GAN generativa](../../labs/08_gan_generation/experiments.md) · [🏠 Índice de rutas](../../parts/README.md) · [🕹️ DQN para inventario con demanda real ➡️](../../labs/10_dqn_reinforcement/experiments.md)
>
> [📄 Guía](README.md) · [🧠 Teoría](theory.md) · **🔬 Experimentos** · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

## Hipótesis principal

Clasificar publicaciones científicas usando texto y enlaces de citas. La hipótesis debe aceptarse o rechazarse comparando el modelo con **MLP sin aristas** y no solo observando que la pérdida disminuye.

## Experimento mínimo

1. Ejecutar `baseline.yaml` con tres semillas.
2. Ejecutar `improved.yaml` con las mismas semillas.
3. Mantener fija la partición de datos dentro de cada semilla.
4. Elegir la variante con `validation`.
5. Comparar la variante elegida contra la línea base en `test`.
6. Revisar intervalos de confianza, errores y costo computacional.

## Experimento específico

Comparar con un mlp que ignore aristas.

## Variables controladas

- Dataset y política de partición.
- Semillas declaradas.
- Presupuesto de épocas y criterio de parada.
- Métrica de selección: `accuracy` o la especificada en la configuración.
- Hardware y versiones registradas en `environment.json`.

## Tabla que debe completarse

| Variante | Semilla | Métrica validation | Métrica test | Tiempo | Parámetros | Observación |
|---|---:|---:|---:|---:|---:|---|
| baseline | 41 | | | | | |
| baseline | 42 | | | | | |
| baseline | 43 | | | | | |
| improved | 41 | | | | | |
| improved | 42 | | | | | |
| improved | 43 | | | | | |

## Criterio de conclusión

La conclusión debe declarar magnitud de la mejora, incertidumbre, costo adicional, errores relevantes y condiciones bajo las cuales el resultado podría no repetirse.

<!-- nav-bottom -->
## 🧭 Navegación del recorrido

| ⬅️ Laboratorio anterior | 🏠 Índice | Laboratorio siguiente ➡️ |
|---|:---:|---|
| [🎨 GAN generativa](../../labs/08_gan_generation/README.md) | [Las 31 rutas](../../parts/README.md) | [🕹️ DQN para inventario con demanda real](../../labs/10_dqn_reinforcement/README.md) |

**En este laboratorio:** [📄 Guía](README.md) · [🧠 Teoría](theory.md) · **🔬 Experimentos** · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🟣 [Parte 3 — Familias especializadas: generar, decidir, relacionar](../../parts/03-familias-especializadas.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/09_gnn_graphs/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
