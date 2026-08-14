# Plan de experimentos — Optimizadores y schedulers

<!-- nav-top -->
> 🧭 **Ruta 19 / 31** · 🔴 [Parte 5 — La mecánica fina, ahora en profundidad](../../parts/05-mecanica-fina.md)
>
> [⬅️ 📐 Activaciones y funciones de pérdida](../../labs/17_activations_and_losses/experiments.md) · [🏠 Índice de rutas](../../parts/README.md) · [🛡️ Regularización ➡️](../../labs/19_regularization_dropout_batchnorm/experiments.md)
>
> [📄 Guía](README.md) · [🧠 Teoría](theory.md) · **🔬 Experimentos** · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

## Hipótesis principal

Comparar SGD, Momentum, Adam y reducción de tasa de aprendizaje. La hipótesis debe aceptarse o rechazarse comparando el modelo con **Media y Ridge** y no solo observando que la pérdida disminuye.

## Experimento mínimo

1. Ejecutar `baseline.yaml` con tres semillas.
2. Ejecutar `improved.yaml` con las mismas semillas.
3. Mantener fija la partición de datos dentro de cada semilla.
4. Elegir la variante con `validation`.
5. Comparar la variante elegida contra la línea base en `test`.
6. Revisar intervalos de confianza, errores y costo computacional.

## Experimento específico

Comparar convergencia y costo por época.

## Variables controladas

- Dataset y política de partición.
- Semillas declaradas.
- Presupuesto de épocas y criterio de parada.
- Métrica de selección: `mae` o la especificada en la configuración.
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
| [📐 Activaciones y funciones de pérdida](../../labs/17_activations_and_losses/README.md) | [Las 31 rutas](../../parts/README.md) | [🛡️ Regularización](../../labs/19_regularization_dropout_batchnorm/README.md) |

**En este laboratorio:** [📄 Guía](README.md) · [🧠 Teoría](theory.md) · **🔬 Experimentos** · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🔴 [Parte 5 — La mecánica fina, ahora en profundidad](../../parts/05-mecanica-fina.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/18_optimizers_and_schedulers/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
