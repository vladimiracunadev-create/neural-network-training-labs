# Guía docente — Clase 17: Backpropagation manual

> **Módulo 5: Mecánica fina** · Esta guía acompaña a quien facilita la clase; no es un guion que deba leerse literalmente.

## La conversación que abre la clase

La salida conoce el error, pero las primeras capas están lejos de él. La regla de la cadena transporta esa señal hacia atrás.

Plantee primero esta pregunta y permita que el grupo formule una predicción antes de mostrar código:

> **¿Cómo se reparte la responsabilidad del error entre todas las capas?**

## Qué debe quedar comprendido

- Derivar y programar backpropagation en una MLP pequeña.
- Preparar y auditar el dataset real iris sin fuga de datos.
- Entrenar y evaluar backpropagation manual.
- Comparar contra la línea base: Regresión logística multinomial.
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

Derivar una MLP capa por capa, comprobar el gradiente y graficar su norma a distintas profundidades.

![La responsabilidad viaja hacia atrás](assets/class-map.svg)

La figura es un punto de entrada. Pida al estudiante que explique qué cambia entre los tres pasos y qué resultado observable invalidaría su explicación.

## Dificultad conceptual que conviene provocar

> **Idea engañosa:** Un programa que ejecuta backward sin error puede seguir teniendo gradientes matemáticamente incorrectos.

No la corrija inmediatamente. Solicite un ejemplo o contraejemplo, ejecute una comparación y recién entonces reconstruya la explicación con el grupo.

## Preguntas para acompañar sin resolver

- ¿Qué observación concreta respalda esa afirmación?
- ¿Qué variable está cambiando y cuáles permanecen controladas?
- ¿La gráfica muestra el mecanismo o solamente una correlación?
- ¿En qué caso dejaría de ser válida esta conclusión?

## Cierre verificable

La clase termina cuando el estudiante puede responder la pregunta esencial, interpretar su visualización y señalar al menos una limitación. Una métrica aislada no reemplaza ninguna de esas tres evidencias.
