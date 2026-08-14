# Evaluación — Perceptrón con PyTorch

<!-- nav-top -->
> 🧭 **Ruta 2 / 31** · 🟢 [Parte 1 — Fundamentos: de la derivada a la primera red](../../parts/01-fundamentos.md)
>
> [⬅️ 🔢 Neurona con NumPy](../../labs/00_numpy_neuron/assessment.md) · [🏠 Índice de rutas](../../parts/README.md) · [🌀 MLP multiclase ➡️](../../labs/02_mlp_nonlinear/assessment.md)
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

1. Explique con sus palabras: z=xW+b; BCEWithLogitsLoss.
2. ¿Qué información del dataset solo puede utilizarse durante entrenamiento?
3. ¿Por qué la línea base **Regresión logística** es una comparación razonable?
4. ¿Qué ejemplos no puede separar un único hiperplano?
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
| [🔢 Neurona con NumPy](../../labs/00_numpy_neuron/README.md) | [Las 31 rutas](../../parts/README.md) | [🌀 MLP multiclase](../../labs/02_mlp_nonlinear/README.md) |

**En este laboratorio:** [📄 Guía](README.md) · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · **📝 Evaluación** · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🟢 [Parte 1 — Fundamentos: de la derivada a la primera red](../../parts/01-fundamentos.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/01_pytorch_perceptron/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
