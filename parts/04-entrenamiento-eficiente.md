# 🟠 Parte 4 — Entrenar mejor, más barato y sin centralizar datos

> 🧭 [⬅️ Parte 3 — Familias especializadas: generar, decidir, relacionar](03-familias-especializadas.md) · [🏠 Índice de partes](README.md) · [📘 Portada](../README.md) · [Parte 5 — La mecánica fina, ahora en profundidad ➡️](05-mecanica-fina.md)

**Rutas:** 13–15 · **Clases:** 3 · **Nivel:** avanzado · **Dedicación estimada:** ~24 h

El modelo ya funciona: ahora hay que mejorarlo sin hacer trampas, encogerlo para que quepa donde debe correr, y entrenarlo cuando los datos no pueden salir de donde están.

## 🧭 Secuencia de la parte

```mermaid
flowchart LR
    L13["13<br/>Búsqueda de hiperparámetros"]
    L14["14<br/>Destilación de conocimiento"]
    L15["15<br/>Aprendizaje federado por participante"]
    L13 --> L14
    L14 --> L15
```

## 📚 Clases de esta parte

| # | Clase | Qué resuelve | Dataset | Horas |
|---:|---|---|---|---:|
| 13 | 🎛️ [Búsqueda de hiperparámetros](../labs/13_hyperparameter_search/README.md) | Optimizar profundidad, ancho, dropout y learning rate sin tocar test | `adult_census` | 8 |
| 14 | ⚗️ [Destilación de conocimiento](../labs/14_knowledge_distillation/README.md) | Transferir conocimiento de una CNN profesora a una estudiante compacta | `cifar10` | 8 |
| 15 | 🌐 [Aprendizaje federado por participante](../labs/15_federated_learning/README.md) | Aplicar FedAvg usando participantes reales como clientes naturales | `uci_har_subjects` | 8 |

> Empieza por 🎛️ **[Búsqueda de hiperparámetros](../labs/13_hyperparameter_search/README.md)** (ruta 14 de 31). Sus documentos: [📄 Guía](../labs/13_hyperparameter_search/README.md) · [🧠 Teoría](../labs/13_hyperparameter_search/theory.md) · [🔬 Experimentos](../labs/13_hyperparameter_search/experiments.md) · [📝 Evaluación](../labs/13_hyperparameter_search/assessment.md).

## 🎯 Qué llevas al terminar

Al completar esta parte, mejoras un modelo sin tocar `test` y sabes qué cuesta cada mejora.

Todas las clases comparten el mismo contrato: los transformadores se ajustan solo con
`train`, `validation` decide el modelo y `test` se abre una única vez tras escribir
`experiment.lock.json`.

---

[⬅️ Parte 3 — Familias especializadas: generar, decidir, relacionar](03-familias-especializadas.md) · [🏠 Índice de partes](README.md) · [📘 Portada del repositorio](../README.md) · [Parte 5 — La mecánica fina, ahora en profundidad ➡️](05-mecanica-fina.md)
