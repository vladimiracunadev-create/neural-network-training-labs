# Plan de experimentos — Transfer learning con mascotas

<!-- nav-top -->
> 🧭 **Ruta 12 / 31** · 🟣 [Parte 3 — Familias especializadas: generar, decidir, relacionar](../../parts/03-familias-especializadas.md)
>
> [⬅️ 🕹️ DQN para inventario con demanda real](../../labs/10_dqn_reinforcement/experiments.md) · [🏠 Índice de rutas](../../parts/README.md) · [🔀 Fusión de sensores ➡️](../../labs/12_multimodal_fusion/experiments.md)
>
> [📄 Guía](README.md) · [🧠 Teoría](theory.md) · **🔬 Experimentos** · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

## Qué significa «experimentar» aquí

Entrenar un modelo y mirar el número que sale **no** es un experimento: es una observación. Un experimento empieza por una afirmación que podría resultar falsa, sigue fijando todo lo que no se está poniendo a prueba, y termina midiendo si la afirmación se sostiene cuando se repite.

La diferencia importa porque el entrenamiento de una red tiene azar por dentro: la inicialización de los pesos y el orden en que se barajan los lotes cambian el resultado aunque no cambies nada más. Si comparas una sola corrida contra otra sola corrida, no sabes si la diferencia viene de tu idea o de la semilla. Por eso todo lo que sigue está organizado para separar **la señal** (el efecto de lo que cambiaste) del **ruido** (la variabilidad propia del entrenamiento).

## La hipótesis de este laboratorio

> Comparar extracción de características, fine-tuning y entrenamiento desde cero.

Formulada como algo que se puede refutar: **el modelo de este laboratorio supera a CNN pequeña entrenada desde cero en `macro_f1`, y la diferencia es mayor que la dispersión entre semillas.**

Qué la haría falsa —y esto también es un resultado que hay que reportar—:

- El modelo no supera a CNN pequeña entrenada desde cero.
- Lo supera, pero por menos de lo que varía el propio modelo al cambiar la semilla de entrenamiento.
- Lo supera solo con una semilla concreta y no con las demás.
- Lo supera a costa de un tiempo o un tamaño que el problema no justifica.

### El experimento propio de esta ruta

Además de la comparación con la línea base, aquí interesa una pregunta específica: **Comparar congelamiento parcial y fine-tuning**. Es la comparación que da sentido al tema de este laboratorio; la de la línea base solo dice si el modelo sirve, mientras que esta dice *qué parte* del diseño es la que aporta.

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
| El conjunto `test` | **Intacto** hasta el final | Se abre una sola vez, después de que `macro_f1` sobre `validation` haya elegido al ganador. |

## Las dos configuraciones que vas a comparar

El laboratorio trae dos configuraciones ya escritas. Estos son los parámetros en los que **difieren** —el resto es idéntico, que es justamente lo que hace legible la comparación—:

| Parámetro | `baseline.yaml` | `improved.yaml` | Qué controla |
|---|---|---|---|
| Épocas | `20` | `50` | Cuántas pasadas completas sobre `train`. |
| Tasa de aprendizaje | `0.001` | `0.0005` | Cuánto se mueve cada peso en la dirección del gradiente. |
| Paciencia | `5` | `8` | Épocas sin mejorar en `validation` antes de detener el entrenamiento. |
| Precisión mixta | no | sí | Usar float16 donde se puede: acelera en GPU, no cambia el protocolo. |
| Procesos de carga | `0` | `2` | Procesos que preparan los lotes en paralelo. |

Elegir entre ellas es una decisión de `validation`. Si `improved` gana en validación, se queda; si no, la configuración base es la respuesta y hay que decirlo así.

## Cómo ejecutar la serie

El comando `benchmark` hace exactamente esto: repite el entrenamiento manteniendo la partición fija y cambiando solo la semilla de entrenamiento.

```bash
neural-labs benchmark --lab 11_transfer_learning \
  --config baseline --split-seed 42 --training-seeds 41 42 43

neural-labs benchmark --lab 11_transfer_learning \
  --config improved --split-seed 42 --training-seeds 41 42 43
```

Por dentro, `benchmark` no hace magia: llama a `run_lab` una vez por semilla con la misma partición y resume los resultados. Desde Python es literalmente eso:

```python
from neural_labs.experiments import run_lab
from neural_labs.benchmarking import summarize_benchmark

registros = []
for semilla in (41, 42, 43):
    resultado = run_lab(
        "11_transfer_learning",
        config_name="baseline",
        split_seed=42,          # la partición NO cambia
        training_seed=semilla,  # solo cambia la inicialización
    )
    registros.append({"training_seed": semilla, "metrics": resultado.metrics})

print(summarize_benchmark(registros))   # media y dispersión entre semillas
```

Empieza con `--quick` para comprobar que la serie corre entera; los números que se reportan salen de la ejecución completa.

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

- **Métrica en `validation`**: el mejor valor de `macro_f1` durante el entrenamiento; está en `history.csv` y resumido en `metrics.json`.
- **Métrica en `test`**: el valor final, en `metrics.json`, ya con el modelo congelado.
- **Tiempo**: `wall_time_seconds` de `metrics.json`.
- **Parámetros**: `parameters` de `metrics.json`; es la medida honesta del tamaño del modelo.
- **Observación**: lo que viste y los números no dicen —una curva que se dispara, una clase que concentra los errores, una corrida que no convergió—.

## Cómo decidir con esos números

1. **Compara medias, pero decide con la dispersión.** Calcula media y desviación entre las tres semillas de cada variante. Una mejora de 0,3 puntos entre variantes cuya dispersión interna es de 1,2 puntos no es una mejora: es ruido con buena suerte.
2. **Si los rangos se solapan, dilo.** «No se observó una diferencia distinguible del ruido con tres semillas» es una conclusión legítima y mucho más útil que un número inventado de confianza.
3. **Decide con `validation`.** La columna de `test` sirve para reportar el resultado final una vez, no para elegir la variante ganadora.
4. **El costo forma parte del resultado.** Si `improved` gana por poco y cuesta el triple de tiempo, la conclusión honesta menciona ambas cosas.
5. **Vuelve a la línea base.** La comparación contra CNN pequeña entrenada desde cero es la que dice si todo el aparato de la red neuronal estaba justificado para este problema.

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
| [🕹️ DQN para inventario con demanda real](../../labs/10_dqn_reinforcement/README.md) | [Las 31 rutas](../../parts/README.md) | [🔀 Fusión de sensores](../../labs/12_multimodal_fusion/README.md) |

**En este laboratorio:** [📄 Guía](README.md) · [🧠 Teoría](theory.md) · **🔬 Experimentos** · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🟣 [Parte 3 — Familias especializadas: generar, decidir, relacionar](../../parts/03-familias-especializadas.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/11_transfer_learning/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
