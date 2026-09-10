# Guía docente — Clase 11: DQN para inventario con demanda real

> **Módulo 3: Familias especializadas** · Esta guía acompaña a quien facilita la clase; no es un guion que deba leerse literalmente.

## La conversación que abre la clase

Pedir poco hoy puede provocar una venta perdida mañana; pedir demasiado inmoviliza inventario. La recompensa conecta ambas consecuencias.

Plantee primero esta pregunta y permita que el grupo formule una predicción antes de mostrar código:

> **¿Cómo aprende una política cuando cada decisión cambia los datos que verá después?**

## Qué debe quedar comprendido

- Aprender una política de reposición usando una secuencia de demanda observada en transacciones reales.
- Preparar y auditar el dataset real online_retail sin fuga de datos.
- Entrenar y evaluar valor de acciones con demanda histórica.
- Comparar contra la línea base: Política de reposición periódica basada en demanda media histórica.
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

Leer trayectorias, comparar políticas y estudiar retorno, quiebres de stock y nivel de servicio por separado.

![De una decisión presente a una recompensa futura](assets/class-map.svg)

La figura es un punto de entrada. Pida al estudiante que explique qué cambia entre los tres pasos y qué resultado observable invalidaría su explicación.

## Dificultad conceptual que conviene provocar

> **Idea engañosa:** Maximizar retorno promedio no asegura una política estable ni un nivel de servicio aceptable en los peores episodios.

No la corrija inmediatamente. Solicite un ejemplo o contraejemplo, ejecute una comparación y recién entonces reconstruya la explicación con el grupo.

## Preguntas para acompañar sin resolver

- ¿Qué observación concreta respalda esa afirmación?
- ¿Qué variable está cambiando y cuáles permanecen controladas?
- ¿La gráfica muestra el mecanismo o solamente una correlación?
- ¿En qué caso dejaría de ser válida esta conclusión?

## Cierre verificable

La clase termina cuando el estudiante puede responder la pregunta esencial, interpretar su visualización y señalar al menos una limitación. Una métrica aislada no reemplaza ninguna de esas tres evidencias.
