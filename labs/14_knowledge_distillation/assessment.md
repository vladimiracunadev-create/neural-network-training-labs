# Evaluación — Destilación de conocimiento

<!-- nav-top -->
> 🧭 **Ruta 15 / 31** · [⬅️ 🎛️ Búsqueda de hiperparámetros](../../labs/13_hyperparameter_search/assessment.md) · [🏠 Índice](../../README.md#laboratorios) · [🌐 Aprendizaje federado por participante ➡️](../../labs/15_federated_learning/assessment.md)
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

1. Explique con sus palabras: L=α CE(y,s)+(1-α)T² KL(softmax(t/T)||softmax(s/T)).
2. ¿Qué información del dataset solo puede utilizarse durante entrenamiento?
3. ¿Por qué la línea base **Estudiante entrenado solo con etiquetas** es una comparación razonable?
4. ¿Qué temperatura equilibra mejor señales duras y blandas?
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
| [🎛️ Búsqueda de hiperparámetros](../../labs/13_hyperparameter_search/README.md) | [Las 31 rutas](../../README.md#laboratorios) | [🌐 Aprendizaje federado por participante](../../labs/15_federated_learning/README.md) |

**En este laboratorio:** [📄 Guía](README.md) · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · **📝 Evaluación** · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

[🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/14_knowledge_distillation/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
