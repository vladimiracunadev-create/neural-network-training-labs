# Plan de experimentos — Incertidumbre y calibración

<!-- nav-top -->
> 🧭 **Ruta 23 / 31** · ⚫ [Parte 6 — Confiar en el modelo y sacarlo del cuaderno](../../parts/06-confianza-y-despliegue.md)
>
> [⬅️ 🔍 Explicabilidad](../../labs/21_explainability/experiments.md) · [🏠 Índice de rutas](../../parts/README.md) · [📦 Exportación e inferencia ➡️](../../labs/23_model_export_and_inference/experiments.md)
>
> [📄 Guía](README.md) · [🧠 Teoría](theory.md) · **🔬 Experimentos** · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

## Hipótesis principal

Medir confianza, Brier score, ECE y temperature scaling. La hipótesis debe aceptarse o rechazarse comparando el modelo con **Regresión logística calibrada** y no solo observando que la pérdida disminuye.

## Experimento mínimo

1. Ejecutar `baseline.yaml` con tres semillas.
2. Ejecutar `improved.yaml` con las mismas semillas.
3. Mantener fija la partición de datos dentro de cada semilla.
4. Elegir la variante con `validation`.
5. Comparar la variante elegida contra la línea base en `test`.
6. Revisar intervalos de confianza, errores y costo computacional.

## Experimento específico

Comparar ece y brier antes y después.

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
| [🔍 Explicabilidad](../../labs/21_explainability/README.md) | [Las 31 rutas](../../parts/README.md) | [📦 Exportación e inferencia](../../labs/23_model_export_and_inference/README.md) |

**En este laboratorio:** [📄 Guía](README.md) · [🧠 Teoría](theory.md) · **🔬 Experimentos** · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

⚫ [Parte 6 — Confiar en el modelo y sacarlo del cuaderno](../../parts/06-confianza-y-despliegue.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/22_uncertainty_calibration/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
