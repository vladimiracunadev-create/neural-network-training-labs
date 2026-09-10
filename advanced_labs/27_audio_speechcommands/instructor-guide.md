# Guía docente — Clase 28: Clasificación de audio con SpeechCommands

> **Módulo 7: Especializaciones avanzadas** · Esta guía acompaña a quien facilita la clase; no es un guion que deba leerse literalmente.

## La conversación que abre la clase

Una palabra dura un segundo, pero sus frecuencias cambian rápidamente. El log-mel convierte ese cambio en una imagen tiempo–frecuencia.

Plantee primero esta pregunta y permita que el grupo formule una predicción antes de mostrar código:

> **¿Qué hace visible un espectrograma que la onda temporal oculta?**

## Qué debe quedar comprendido

- Clasificar comandos hablados desde waveform y log-mel spectrograms.
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

Escuchar y visualizar señales, construir espectrogramas y medir degradación con ruido y SpecAugment.

![De vibración a patrón tiempo–frecuencia](assets/class-map.svg)

La figura es un punto de entrada. Pida al estudiante que explique qué cambia entre los tres pasos y qué resultado observable invalidaría su explicación.

## Dificultad conceptual que conviene provocar

> **Idea engañosa:** Un modelo preciso en audio limpio puede depender del micrófono o del fondo y fallar ante voces reales distintas.

No la corrija inmediatamente. Solicite un ejemplo o contraejemplo, ejecute una comparación y recién entonces reconstruya la explicación con el grupo.

## Preguntas para acompañar sin resolver

- ¿Qué observación concreta respalda esa afirmación?
- ¿Qué variable está cambiando y cuáles permanecen controladas?
- ¿La gráfica muestra el mecanismo o solamente una correlación?
- ¿En qué caso dejaría de ser válida esta conclusión?

## Cierre verificable

La clase termina cuando el estudiante puede responder la pregunta esencial, interpretar su visualización y señalar al menos una limitación. Una métrica aislada no reemplaza ninguna de esas tres evidencias.
