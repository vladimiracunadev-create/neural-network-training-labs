# Evaluación — DQN para inventario con demanda real

<!-- nav-top -->
> 🧭 **Ruta 11 / 31** · 🟣 [Parte 3 — Familias especializadas: generar, decidir, relacionar](../../parts/03-familias-especializadas.md)
>
> [⬅️ 🕸️ GNN sobre red de citas](../../labs/09_gnn_graphs/assessment.md) · [🏠 Índice de rutas](../../parts/README.md) · [♻️ Transfer learning con mascotas ➡️](../../labs/11_transfer_learning/assessment.md)
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

1. Explique con sus palabras: y=r+γ max_a Q_target(s′,a); la demanda de cada paso proviene del historial real, no de un generador.
2. ¿Qué información del dataset solo puede utilizarse durante entrenamiento?
3. ¿Por qué la línea base **Política de reposición periódica basada en demanda media histórica** es una comparación razonable?
4. ¿La política es robusta a cambios en costo y demanda?
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
| [🕸️ GNN sobre red de citas](../../labs/09_gnn_graphs/README.md) | [Las 31 rutas](../../parts/README.md) | [♻️ Transfer learning con mascotas](../../labs/11_transfer_learning/README.md) |

**En este laboratorio:** [📄 Guía](README.md) · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · **📝 Evaluación** · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🟣 [Parte 3 — Familias especializadas: generar, decidir, relacionar](../../parts/03-familias-especializadas.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/10_dqn_reinforcement/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
