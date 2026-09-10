# Guía docente — Clase 12: Transfer learning con mascotas

> **Módulo 3: Familias especializadas** · Esta guía acompaña a quien facilita la clase; no es un guion que deba leerse literalmente.

## La conversación que abre la clase

Un modelo que ya reconoce bordes, texturas y formas no parte de cero al ver mascotas, pero sus rasgos tampoco son perfectos para la nueva tarea.

Plantee primero esta pregunta y permita que el grupo formule una predicción antes de mostrar código:

> **¿Cuándo conviene reutilizar representaciones y cuándo hay que volver a aprenderlas?**

## Qué debe quedar comprendido

- Comparar extracción de características, fine-tuning y entrenamiento desde cero.
- Preparar y auditar el dataset real oxford_iiit_pet sin fuga de datos.
- Entrenar y evaluar reutilización de representaciones preentrenadas.
- Comparar contra la línea base: CNN pequeña entrenada desde cero.
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

Comparar extracción de características, fine-tuning y entrenamiento desde cero con igual presupuesto.

![Qué se conserva y qué se adapta](assets/class-map.svg)

La figura es un punto de entrada. Pida al estudiante que explique qué cambia entre los tres pasos y qué resultado observable invalidaría su explicación.

## Dificultad conceptual que conviene provocar

> **Idea engañosa:** Fine-tuning no siempre gana; con pocos datos o una tasa alta puede destruir representaciones útiles.

No la corrija inmediatamente. Solicite un ejemplo o contraejemplo, ejecute una comparación y recién entonces reconstruya la explicación con el grupo.

## Preguntas para acompañar sin resolver

- ¿Qué observación concreta respalda esa afirmación?
- ¿Qué variable está cambiando y cuáles permanecen controladas?
- ¿La gráfica muestra el mecanismo o solamente una correlación?
- ¿En qué caso dejaría de ser válida esta conclusión?

## Cierre verificable

La clase termina cuando el estudiante puede responder la pregunta esencial, interpretar su visualización y señalar al menos una limitación. Una métrica aislada no reemplaza ninguna de esas tres evidencias.
