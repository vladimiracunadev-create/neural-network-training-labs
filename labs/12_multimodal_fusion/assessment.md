# Evaluación — Fusión de sensores

<!-- nav-top -->
> 🧭 **Ruta 13 / 31** · 🟣 [Parte 3 — Familias especializadas: generar, decidir, relacionar](../../parts/03-familias-especializadas.md)
>
> [⬅️ ♻️ Transfer learning con mascotas](../../labs/11_transfer_learning/assessment.md) · [🏠 Índice de rutas](../../parts/README.md) · [🎛️ Búsqueda de hiperparámetros ➡️](../../labs/13_hyperparameter_search/assessment.md)
>
> [📄 Guía](README.md) · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · **📝 Evaluación**
<!-- /nav-top -->

## Evidencias obligatorias

- Dataset preparado y auditoría sin solapamientos.
- Notebook ejecutado sin celdas omitidas.
- Línea base y modelo neuronal comparados.
- Resultados de al menos tres semillas o justificación del costo.
- Análisis de errores y limitaciones.
- Model card actualizada.

## Preguntas

1. Explique con sus palabras: f=[f_acc(x_acc); f_gyro(x_gyro)]; y=head(f).
2. ¿Qué información del dataset solo puede utilizarse durante entrenamiento?
3. ¿Por qué la línea base **Acelerómetro solo, giroscopio solo y regresión logística** es una comparación razonable?
4. ¿Qué modalidad explica cada actividad?
5. ¿Qué cambiaría antes de usar este modelo fuera del laboratorio?

## Rúbrica

| Criterio | Insuficiente | Adecuado | Excelente | Peso |
|---|---|---|---|---:|
| Integridad de datos | mezcla particiones | separación correcta | auditoría, hashes y justificación | 20% |
| Implementación | no ejecuta | entrena y evalúa | código claro, reusable y probado | 20% |
| Diseño experimental | resultado aislado | comparación controlada | multi-semilla e incertidumbre | 20% |
| Análisis | repite métricas | interpreta errores | identifica sesgos, límites y costo | 25% |
| Comunicación | incompleta | reporte entendible | model card y conclusiones verificables | 15% |

La aprobación exige al menos 70% y cero errores críticos de fuga de datos.

<!-- nav-bottom -->
## 🧭 Navegación del recorrido

| ⬅️ Laboratorio anterior | 🏠 Índice | Laboratorio siguiente ➡️ |
|---|:---:|---|
| [♻️ Transfer learning con mascotas](../../labs/11_transfer_learning/README.md) | [Las 31 rutas](../../parts/README.md) | [🎛️ Búsqueda de hiperparámetros](../../labs/13_hyperparameter_search/README.md) |

**En este laboratorio:** [📄 Guía](README.md) · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · **📝 Evaluación** · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🟣 [Parte 3 — Familias especializadas: generar, decidir, relacionar](../../parts/03-familias-especializadas.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/12_multimodal_fusion/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
