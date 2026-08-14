# Plan de experimentos — Proyecto final: churn de telecomunicaciones

<!-- nav-top -->
> 🧭 **Ruta 25 / 31** · [⬅️ 📦 Exportación e inferencia](../../labs/23_model_export_and_inference/experiments.md) · [🏠 Índice](../../README.md#laboratorios) · [🔧 Fine-tuning eficiente de transformer ➡️](../../advanced_labs/25_transformer_finetuning/experiments.md)
>
> [📄 Guía](README.md) · [🧠 Teoría](theory.md) · **🔬 Experimentos** · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

## Hipótesis principal

Resolver de extremo a extremo un problema real de abandono de clientes con documentación, evaluación y despliegue. La hipótesis debe aceptarse o rechazarse comparando el modelo con **Regresión logística y Gradient Boosting** y no solo observando que la pérdida disminuye.

## Experimento mínimo

1. Ejecutar `baseline.yaml` con tres semillas.
2. Ejecutar `improved.yaml` con las mismas semillas.
3. Mantener fija la partición de datos dentro de cada semilla.
4. Elegir la variante con `validation`.
5. Comparar la variante elegida contra la línea base en `test`.
6. Revisar intervalos de confianza, errores y costo computacional.

## Experimento específico

Formular hipótesis, segmentar errores y documentar riesgos.

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
| [📦 Exportación e inferencia](../../labs/23_model_export_and_inference/README.md) | [Las 31 rutas](../../README.md#laboratorios) | [🔧 Fine-tuning eficiente de transformer](../../advanced_labs/25_transformer_finetuning/README.md) |

**En este laboratorio:** [📄 Guía](README.md) · [🧠 Teoría](theory.md) · **🔬 Experimentos** · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

[🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/24_capstone_real_project/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
