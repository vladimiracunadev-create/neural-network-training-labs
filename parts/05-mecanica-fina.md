# 🔴 Parte 5 — La mecánica fina, ahora en profundidad

> 🧭 [⬅️ Parte 4 — Entrenar mejor, más barato y sin centralizar datos](04-entrenamiento-eficiente.md) · [🏠 Índice de partes](README.md) · [📘 Portada](../README.md) · [Parte 6 — Confiar en el modelo y sacarlo del cuaderno ➡️](06-confianza-y-despliegue.md)

**Rutas:** 16–20 · **Clases:** 5 · **Nivel:** fundamentos · intermedio · **Dedicación estimada:** ~24 h

Segunda pasada por el motor, ya con la experiencia de haber entrenado modelos reales: lo que en la ruta 00 era una fórmula, aquí es una decisión de diseño que se mide, se compara entre semillas y se justifica.

## 🧭 Secuencia de la parte

```mermaid
flowchart LR
    L16["16<br/>Backpropagation manual"]
    L17["17<br/>Activaciones y funciones de pérdida"]
    L18["18<br/>Optimizadores y schedulers"]
    L19["19<br/>Regularización"]
    L20["20<br/>Aumento de datos"]
    L16 --> L17
    L17 --> L18
    L18 --> L19
    L19 --> L20
```

## 📚 Clases de esta parte

| # | Clase | Qué resuelve | Dataset | Horas |
|---:|---|---|---|---:|
| 16 | ∂ [Backpropagation manual](../labs/16_backpropagation_manual/README.md) | Derivar y programar backpropagation en una MLP pequeña | `iris` | 4 |
| 17 | 📐 [Activaciones y funciones de pérdida](../labs/17_activations_and_losses/README.md) | Comparar ReLU, GELU, Tanh y pérdidas apropiadas en clases desbalanceadas | `wine_quality` | 4 |
| 18 | ⚙️ [Optimizadores y schedulers](../labs/18_optimizers_and_schedulers/README.md) | Comparar SGD, Momentum, Adam y reducción de tasa de aprendizaje | `california_housing` | 4 |
| 19 | 🛡️ [Regularización](../labs/19_regularization_dropout_batchnorm/README.md) | Medir dropout, weight decay y batch normalization | `fashion_mnist` | 6 |
| 20 | 🔄 [Aumento de datos](../labs/20_data_augmentation/README.md) | Comparar recortes, volteos y perturbaciones sobre imágenes reales | `cifar10` | 6 |

> Empieza por ∂ **[Backpropagation manual](../labs/16_backpropagation_manual/README.md)** (ruta 17 de 31). Sus documentos: [📄 Guía](../labs/16_backpropagation_manual/README.md) · [🧠 Teoría](../labs/16_backpropagation_manual/theory.md) · [🔬 Experimentos](../labs/16_backpropagation_manual/experiments.md) · [📝 Evaluación](../labs/16_backpropagation_manual/assessment.md).

## 🎯 Qué llevas al terminar

Al completar esta parte, explicas por qué un entrenamiento converge, se estanca o sobreajusta.

Todas las clases comparten el mismo contrato: los transformadores se ajustan solo con
`train`, `validation` decide el modelo y `test` se abre una única vez tras escribir
`experiment.lock.json`.

---

[⬅️ Parte 4 — Entrenar mejor, más barato y sin centralizar datos](04-entrenamiento-eficiente.md) · [🏠 Índice de partes](README.md) · [📘 Portada del repositorio](../README.md) · [Parte 6 — Confiar en el modelo y sacarlo del cuaderno ➡️](06-confianza-y-despliegue.md)
