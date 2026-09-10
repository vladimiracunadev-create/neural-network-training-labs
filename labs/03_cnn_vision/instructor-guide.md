# Guía docente — Clase 04: CNN para visión

> **Módulo 2: Arquitecturas** · Esta guía acompaña a quien facilita la clase; no es un guion que deba leerse literalmente.

## La conversación que abre la clase

Un borde sigue siendo un borde en cualquier zona de una fotografía. Una CNN reutiliza el mismo detector en toda la imagen.

Plantee primero esta pregunta y permita que el grupo formule una predicción antes de mostrar código:

> **¿Por qué una convolución reconoce un patrón aunque cambie de posición?**

## Qué debe quedar comprendido

- Entrenar una CNN y analizar errores sobre fotografías reales de diez clases.
- Preparar y auditar el dataset real cifar10 sin fuga de datos.
- Entrenar y evaluar convoluciones para patrones espaciales.
- Comparar contra la línea base: Clasificador lineal sobre píxeles.
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

Mirar filtros y mapas de activación, calcular el campo receptivo y estudiar errores por clase en CIFAR-10.

![De píxeles a rasgos visuales](assets/class-map.svg)

La figura es un punto de entrada. Pida al estudiante que explique qué cambia entre los tres pasos y qué resultado observable invalidaría su explicación.

## Dificultad conceptual que conviene provocar

> **Idea engañosa:** Una activación intensa no demuestra que la red haya comprendido un objeto; solo muestra respuesta a un patrón aprendido.

No la corrija inmediatamente. Solicite un ejemplo o contraejemplo, ejecute una comparación y recién entonces reconstruya la explicación con el grupo.

## Preguntas para acompañar sin resolver

- ¿Qué observación concreta respalda esa afirmación?
- ¿Qué variable está cambiando y cuáles permanecen controladas?
- ¿La gráfica muestra el mecanismo o solamente una correlación?
- ¿En qué caso dejaría de ser válida esta conclusión?

## Cierre verificable

La clase termina cuando el estudiante puede responder la pregunta esencial, interpretar su visualización y señalar al menos una limitación. Una métrica aislada no reemplaza ninguna de esas tres evidencias.
