# Evaluación — Neurona con NumPy

<!-- nav-top -->
> 🧭 **Ruta 1 / 31** · 🟢 [Parte 1 — Fundamentos: de la derivada a la primera red](../../parts/01-fundamentos.md)
>
> ⬅️ *inicio del recorrido* · [🏠 Índice de rutas](../../parts/README.md) · [🧩 Perceptrón con PyTorch ➡️](../../labs/01_pytorch_perceptron/assessment.md)
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

1. Explique con sus palabras: p(y=1|x)=σ(xw+b); gradiente de la entropía cruzada.
2. ¿Qué información del dataset solo puede utilizarse durante entrenamiento?
3. ¿Por qué la línea base **DummyClassifier y regresión logística de scikit-learn** es una comparación razonable?
4. ¿Cómo cambia la convergencia al modificar la escala de las variables?
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
| *— inicio del recorrido* | [Las 31 rutas](../../parts/README.md) | [🧩 Perceptrón con PyTorch](../../labs/01_pytorch_perceptron/README.md) |

**En este laboratorio:** [📄 Guía](README.md) · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · **📝 Evaluación** · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🟢 [Parte 1 — Fundamentos: de la derivada a la primera red](../../parts/01-fundamentos.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/00_numpy_neuron/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
