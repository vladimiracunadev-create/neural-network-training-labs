# Guía docente — Clase 26: Fine-tuning eficiente de transformer

> **Módulo 7: Especializaciones avanzadas** · Esta guía acompaña a quien facilita la clase; no es un guion que deba leerse literalmente.

## La conversación que abre la clase

LoRA modifica una fracción pequeña de los parámetros y plantea una pregunta medible: cuánto rendimiento se conserva por cada recurso ahorrado.

Plantee primero esta pregunta y permita que el grupo formule una predicción antes de mostrar código:

> **¿Cuánta adaptación necesita un transformer para una tarea nueva?**

## Qué debe quedar comprendido

- Comparar fine-tuning completo y LoRA sin tocar test durante selección.
- Interpretar accuracy, macro_f1
- Aplicar sellado de test y reproducibilidad

## Secuencia sugerida

| Momento | Duración | Acción docente | Evidencia del estudiante |
|---|---:|---|---|
| Activación | 10 min | Presentar el caso inicial y recoger predicciones. | Explica qué espera observar y por qué. |
| Construcción | 25 min | Conectar intuición, representación y matemática. | Dibuja o relata el mecanismo con sus propias palabras. |
| Demostración | 25 min | Recorrer el mapa visual y ejecutar el ejemplo mínimo. | Interpreta cada transición sin limitarse a describir la figura. |
| Investigación | 35 min | Guiar la práctica propia de esta clase. | Contrasta su predicción con una medida o gráfica. |
| Cierre | 15 min | Volver a la pregunta esencial y discutir límites. | Entrega una conclusión breve con evidencia y una duda abierta. |

## Práctica propia de esta materia

Comparar fine-tuning completo y LoRA con igual partición y reportar parámetros, memoria, latencia y macro-F1.

![Adaptar poco sin aprender poco](assets/class-map.svg)

La figura es un punto de entrada. Pida al estudiante que explique qué cambia entre los tres pasos y qué resultado observable invalidaría su explicación.

## Dificultad conceptual que conviene provocar

> **Idea engañosa:** Entrenar menos parámetros no garantiza menor latencia de inferencia; el ahorro principal puede estar solo en entrenamiento y almacenamiento.

No la corrija inmediatamente. Solicite un ejemplo o contraejemplo, ejecute una comparación y recién entonces reconstruya la explicación con el grupo.

## Preguntas para acompañar sin resolver

- ¿Qué observación concreta respalda esa afirmación?
- ¿Qué variable está cambiando y cuáles permanecen controladas?
- ¿La gráfica muestra el mecanismo o solamente una correlación?
- ¿En qué caso dejaría de ser válida esta conclusión?

## Cierre verificable

La clase termina cuando el estudiante puede responder la pregunta esencial, interpretar su visualización y señalar al menos una limitación. Una métrica aislada no reemplaza ninguna de esas tres evidencias.
