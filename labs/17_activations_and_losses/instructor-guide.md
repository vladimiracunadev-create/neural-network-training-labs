# Guía docente — Clase 18: Activaciones y funciones de pérdida

> **Módulo 5: Mecánica fina** · Esta guía acompaña a quien facilita la clase; no es un guion que deba leerse literalmente.

## La conversación que abre la clase

Dos modelos idénticos pueden reaccionar de manera opuesta al mismo ejemplo difícil si sus pérdidas asignan costos distintos.

Plantee primero esta pregunta y permita que el grupo formule una predicción antes de mostrar código:

> **¿Cómo cambia lo que aprende una red al cambiar activación o pérdida?**

## Qué debe quedar comprendido

- Comparar ReLU, GELU, Tanh y pérdidas apropiadas en clases desbalanceadas.
- Preparar y auditar el dataset real wine_quality sin fuga de datos.
- Entrenar y evaluar comparación controlada de activaciones y pérdidas.
- Comparar contra la línea base: Regresión ordinal y Random Forest.
- Interpretar intervalos de confianza, errores y limitaciones.

## Secuencia sugerida

| Momento | Duración | Acción docente | Evidencia del estudiante |
|---|---:|---|---|
| Activación | 10 min | Presentar el caso inicial y recoger predicciones. | Explica qué espera observar y por qué. |
| Construcción | 25 min | Conectar intuición, representación y matemática. | Dibuja o relata el mecanismo con sus propias palabras. |
| Demostración | 25 min | Recorrer el mapa visual y ejecutar el ejemplo mínimo. | Interpreta cada transición sin limitarse a describir la figura. |
| Investigación | 35 min | Guiar la práctica propia de esta clase. | Contrasta su predicción con una medida o gráfica. |
| Cierre | 15 min | Volver a la pregunta esencial y discutir límites. | Entrega una conclusión breve con evidencia y una duda abierta. |

## Práctica propia de esta materia

Comparar funciones y derivadas, provocar saturación y relacionar pérdidas con desbalance y costos de error.

![La forma de la función dirige el aprendizaje](assets/class-map.svg)

La figura es un punto de entrada. Pida al estudiante que explique qué cambia entre los tres pasos y qué resultado observable invalidaría su explicación.

## Dificultad conceptual que conviene provocar

> **Idea engañosa:** Elegir una pérdida es una decisión sobre qué errores importan, no un detalle intercambiable de implementación.

No la corrija inmediatamente. Solicite un ejemplo o contraejemplo, ejecute una comparación y recién entonces reconstruya la explicación con el grupo.

## Preguntas para acompañar sin resolver

- ¿Qué observación concreta respalda esa afirmación?
- ¿Qué variable está cambiando y cuáles permanecen controladas?
- ¿La gráfica muestra el mecanismo o solamente una correlación?
- ¿En qué caso dejaría de ser válida esta conclusión?

## Cierre verificable

La clase termina cuando el estudiante puede responder la pregunta esencial, interpretar su visualización y señalar al menos una limitación. Una métrica aislada no reemplaza ninguna de esas tres evidencias.
