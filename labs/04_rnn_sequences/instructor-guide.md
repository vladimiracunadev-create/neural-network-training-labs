# Guía docente — Clase 05: RNN para texto

> **Módulo 2: Arquitecturas** · Esta guía acompaña a quien facilita la clase; no es un guion que deba leerse literalmente.

## La conversación que abre la clase

Al leer una reseña, una negación temprana puede cambiar el sentido de una frase mucho después. La RNN intenta conservar ese contexto paso a paso.

Plantee primero esta pregunta y permita que el grupo formule una predicción antes de mostrar código:

> **¿Cuánta información del comienzo de una secuencia llega realmente al final?**

## Qué debe quedar comprendido

- Clasificar sentimiento en reseñas reales usando embeddings y recurrencia.
- Preparar y auditar el dataset real imdb sin fuga de datos.
- Entrenar y evaluar recurrencia sobre secuencias tokenizadas.
- Comparar contra la línea base: TF-IDF + regresión logística.
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

Seguir el estado oculto, comparar longitudes y observar cómo el gradiente se debilita o explota con la distancia.

![Memoria y pérdida de señal en una RNN](assets/class-map.svg)

La figura es un punto de entrada. Pida al estudiante que explique qué cambia entre los tres pasos y qué resultado observable invalidaría su explicación.

## Dificultad conceptual que conviene provocar

> **Idea engañosa:** El último estado no contiene necesariamente toda la secuencia; la información puede degradarse mucho antes de llegar a él.

No la corrija inmediatamente. Solicite un ejemplo o contraejemplo, ejecute una comparación y recién entonces reconstruya la explicación con el grupo.

## Preguntas para acompañar sin resolver

- ¿Qué observación concreta respalda esa afirmación?
- ¿Qué variable está cambiando y cuáles permanecen controladas?
- ¿La gráfica muestra el mecanismo o solamente una correlación?
- ¿En qué caso dejaría de ser válida esta conclusión?

## Cierre verificable

La clase termina cuando el estudiante puede responder la pregunta esencial, interpretar su visualización y señalar al menos una limitación. Una métrica aislada no reemplaza ninguna de esas tres evidencias.
