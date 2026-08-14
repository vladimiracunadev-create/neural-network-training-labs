# Plan de experimentos — Segmentación semántica con U-Net

<!-- nav-top -->
> 🧭 **Ruta 27 / 31** · 🔬 [Parte 7 — Especializaciones avanzadas](../../parts/07-especializaciones-avanzadas.md)
>
> [⬅️ 🔧 Fine-tuning eficiente de transformer](../../advanced_labs/25_transformer_finetuning/experiments.md) · [🏠 Índice de rutas](../../parts/README.md) · [🎙️ Clasificación de audio con SpeechCommands ➡️](../../advanced_labs/27_audio_speechcommands/experiments.md)
>
> [📄 Guía](README.md) · [🧠 Teoría](theory.md) · **🔬 Experimentos** · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

## Qué significa «experimentar» aquí

Entrenar un modelo y mirar el número que sale **no** es un experimento: es una observación. Un experimento empieza por una afirmación que podría resultar falsa, sigue fijando todo lo que no se está poniendo a prueba, y termina midiendo si la afirmación se sostiene cuando se repite.

La diferencia importa porque el entrenamiento de una red tiene azar por dentro: la inicialización de los pesos y el orden en que se barajan los lotes cambian el resultado aunque no cambies nada más. Si comparas una sola corrida contra otra sola corrida, no sabes si la diferencia viene de tu idea o de la semilla. Por eso todo lo que sigue está organizado para separar **la señal** (el efecto de lo que cambiaste) del **ruido** (la variabilidad propia del entrenamiento).

## La hipótesis de este laboratorio

> Segmentar mascota, fondo y contorno con IoU por clase.

Formulada como algo que se puede refutar: **el modelo de este laboratorio supera a Máscara de clase mayoritaria en `mean_iou`, y la diferencia es mayor que la dispersión entre semillas.**

Qué la haría falsa —y esto también es un resultado que hay que reportar—:

- El modelo no supera a Máscara de clase mayoritaria.
- Lo supera, pero por menos de lo que varía el propio modelo al cambiar la semilla de entrenamiento.
- Lo supera solo con una semilla concreta y no con las demás.
- Lo supera a costa de un tiempo o un tamaño que el problema no justifica.

### El experimento propio de esta ruta

Además de la comparación con la línea base, aquí interesa una pregunta específica: **Comparar U-Net pequeña con y sin aumento geométrico y ponderación de clases**. Es la comparación que da sentido al tema de este laboratorio; la de la línea base solo dice si el modelo sirve, mientras que esta dice *qué parte* del diseño es la que aporta.

## Qué se varía y qué se mantiene fijo

Un experimento es interpretable cuando cambia **una** cosa a la vez. Esta tabla dice qué papel juega cada elemento y, sobre todo, por qué:

| Elemento | En este experimento | Por qué |
|---|---|---|
| Partición de los datos (`--split-seed`) | **Fija** en 42 | Si cambiara entre corridas, no podrías saber si la diferencia viene de tu modelo o de que le tocaron datos distintos. |
| Semilla de entrenamiento (`--training-seed`) | **Varía**: 41, 42, 43 | Es exactamente el ruido que quieres medir. Sin varias semillas no tienes con qué comparar la mejora. |
| Configuración (`--config`) | **Varía**: `baseline` y `improved` | Es la intervención bajo estudio: lo único que estás poniendo a prueba. |
| Dataset y preprocesamiento | **Fijos** | Ajustar la normalización o el vocabulario con otros datos cambiaría el problema, no el modelo. |
| Presupuesto de épocas y criterio de parada | **Fijos dentro de cada configuración** | Dar más épocas a una variante que a otra es comparar dos cosas distintas. |
| Hardware y versiones | **Registrados** en `environment.json` | No siempre se pueden fijar, pero sí dejar anotados: explican diferencias de tiempo y, a veces, de resultado. |
| El conjunto `test` | **Intacto** hasta el final | Se abre una sola vez, después de que `mean_iou` sobre `validation` haya elegido al ganador. |

## Cómo ejecutar la serie

En las especializaciones no hay comando de repetición automática: se lanza el entrenamiento una vez por semilla, manteniendo fija la partición.

```bash
neural-labs train-advanced --track 26_segmentation_unet --split-seed 42 --training-seed 41
neural-labs train-advanced --track 26_segmentation_unet --split-seed 42 --training-seed 42
neural-labs train-advanced --track 26_segmentation_unet --split-seed 42 --training-seed 43
```

Cada corrida escribe su propio directorio en `runs-advanced/`, con `metrics.json` e `history.json`.

## La tabla que debes completar

| Variante | Semilla | Métrica en `validation` | Métrica en `test` | Tiempo | Parámetros | Observación |
|---|---:|---:|---:|---:|---:|---|
| baseline | 41 | | | | | |
| baseline | 42 | | | | | |
| baseline | 43 | | | | | |
| improved | 41 | | | | | |
| improved | 42 | | | | | |
| improved | 43 | | | | | |

De dónde sale cada columna, para que no haya que adivinarlo:

- **Métrica en `validation`**: el mejor valor de `mean_iou` durante el entrenamiento; está en `history.csv` y resumido en `metrics.json`.
- **Métrica en `test`**: el valor final, en `metrics.json`, ya con el modelo congelado.
- **Tiempo**: `wall_time_seconds` de `metrics.json`.
- **Parámetros**: `parameters` de `metrics.json`; es la medida honesta del tamaño del modelo.
- **Observación**: lo que viste y los números no dicen —una curva que se dispara, una clase que concentra los errores, una corrida que no convergió—.

## Cómo decidir con esos números

1. **Compara medias, pero decide con la dispersión.** Calcula media y desviación entre las tres semillas de cada variante. Una mejora de 0,3 puntos entre variantes cuya dispersión interna es de 1,2 puntos no es una mejora: es ruido con buena suerte.
2. **Si los rangos se solapan, dilo.** «No se observó una diferencia distinguible del ruido con tres semillas» es una conclusión legítima y mucho más útil que un número inventado de confianza.
3. **Decide con `validation`.** La columna de `test` sirve para reportar el resultado final una vez, no para elegir la variante ganadora.
4. **El costo forma parte del resultado.** Si `improved` gana por poco y cuesta el triple de tiempo, la conclusión honesta menciona ambas cosas.
5. **Vuelve a la línea base.** La comparación contra Máscara de clase mayoritaria es la que dice si todo el aparato de la red neuronal estaba justificado para este problema.

## Qué debe decir tu conclusión

Una conclusión completa contiene cinco cosas, y se puede escribir en un párrafo:

- **Magnitud**: cuánto mejoró, en qué métrica y sobre qué conjunto.
- **Incertidumbre**: cuánto varió entre semillas, y si la mejora sobrevive a esa variación.
- **Costo**: tiempo, memoria o tamaño adicional que hubo que pagar.
- **Errores**: dónde falla el modelo, no solo cuánto acierta.
- **Condiciones**: en qué circunstancias no esperarías que este resultado se repitiera.

## Errores de diseño que invalidan el experimento

- **Cambiar dos cosas a la vez.** Si tocas la arquitectura y la tasa de aprendizaje en la misma corrida, el resultado no atribuye el efecto a ninguna de las dos.
- **Comparar con particiones distintas.** Es el error más frecuente y el más difícil de detectar después: parece una mejora del modelo y es un reparto distinto de los datos.
- **Quedarse con la mejor semilla.** Elegir la corrida más favorable y reportar solo esa es seleccionar el ruido. Se reportan todas.
- **Ajustar mirando `test`.** En cuanto una decisión se toma con el resultado de `test`, ese conjunto deja de estimar el rendimiento con datos nuevos.
- **No registrar el entorno.** Sin versiones ni hardware, una diferencia de tiempo o de resultado se vuelve inexplicable meses después.

<!-- nav-bottom -->
## 🧭 Navegación del recorrido

| ⬅️ Laboratorio anterior | 🏠 Índice | Laboratorio siguiente ➡️ |
|---|:---:|---|
| [🔧 Fine-tuning eficiente de transformer](../../advanced_labs/25_transformer_finetuning/README.md) | [Las 31 rutas](../../parts/README.md) | [🎙️ Clasificación de audio con SpeechCommands](../../advanced_labs/27_audio_speechcommands/README.md) |

**En este laboratorio:** [📄 Guía](README.md) · [🧠 Teoría](theory.md) · **🔬 Experimentos** · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🔬 [Parte 7 — Especializaciones avanzadas](../../parts/07-especializaciones-avanzadas.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/26_segmentation_unet/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
