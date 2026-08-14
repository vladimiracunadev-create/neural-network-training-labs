# Exportación e inferencia

<!-- nav-top -->
> 🧭 **Ruta 24 / 31** · ⚫ [Parte 6 — Confiar en el modelo y sacarlo del cuaderno](../../parts/06-confianza-y-despliegue.md)
>
> [⬅️ 🎯 Incertidumbre y calibración](../../labs/22_uncertainty_calibration/README.md) · [🏠 Índice de rutas](../../parts/README.md) · [🏁 Proyecto final: churn de telecomunicaciones ➡️](../../labs/24_capstone_real_project/README.md)
>
> **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

## 🎯 Qué vas a hacer aquí

Exportar ONNX, validar paridad y medir latencia por lotes.

Es la **ruta 24 de 31** del recorrido y pertenece a ⚫ la parte 6, *Confiar en el modelo y sacarlo del cuaderno*. Llegas desde **Incertidumbre y calibración** y lo que hagas aquí lo da por supuesto **Proyecto final: churn de telecomunicaciones**.

Trabajarás con el dataset **`cifar10`** (Torchvision / University of Toronto, licencia: Consultar términos CIFAR-10), y tendrás que superar la línea base **PyTorch eager**, decidiendo con la métrica `macro_f1` medida sobre `validation`. Nivel avanzado, unas **8 horas** de dedicación.

**Lo que conviene traer resuelto de las rutas anteriores:** PyTorch intermedio, optimización, lectura de artículos técnicos.

**Al terminar deberías ser capaz de:**

- Exportar ONNX, validar paridad y medir latencia por lotes.
- Preparar y auditar el dataset real cifar10 sin fuga de datos.
- Entrenar y evaluar exportación y perfil de inferencia.
- Comparar contra la línea base: PyTorch eager.
- Interpretar intervalos de confianza, errores y limitaciones.

## 🧠 La teoría de este laboratorio

Esta sección es la explicación completa del tema. No hace falta abrir otro archivo para entender lo que viene después: aquí está la idea, la matemática que la sostiene y sus límites. (El mismo texto vive en `theory.md`, que es la fuente desde la que se genera esta guía, junto con la bibliografía del final.)

### De qué trata

Este laboratorio estudia **exportación y perfil de inferencia** usando `cifar10`, un dataset público real procedente de Torchvision / University of Toronto.

Un modelo entrenado en PyTorch vive dentro de un intérprete de Python y un grafo dinámico; llevarlo a producción exige *desacoplarlo* de ese entorno y empaquetarlo en un formato portátil que un motor de inferencia optimizado pueda ejecutar en servidores, móviles o dispositivos edge. **ONNX** (Open Neural Network Exchange) es ese formato intermedio: representa la red como un grafo estático de operadores estándar, independiente del framework de origen. Exportar consiste en trazar el modelo entrenado y volcarlo a ONNX; a partir de ahí un runtime como ONNX Runtime lo carga y lo ejecuta con optimizaciones de bajo nivel.

La pregunta central del laboratorio es de **compromisos**: al pasar de PyTorch eager al modelo exportado —y opcionalmente cuantizado— ganamos velocidad y reducimos tamaño, pero debemos verificar que no rompemos la corrección. Por eso el flujo es exportar, validar *paridad numérica* (que ONNX y PyTorch producen la misma salida) y luego perfilar latencia, throughput y tamaño sobre `cifar10`, comparando siempre contra la línea base de PyTorch eager.

### La matemática, paso a paso

Paridad numérica y costo de inferencia.

La **paridad numérica** es la condición de que el modelo exportado calcule esencialmente la misma función que el original. No se exige igualdad bit a bit —el reordenamiento de operaciones y las diferencias de kernels producen redondeos distintos en aritmética de punto flotante— sino que la diferencia esté acotada por una tolerancia. Se comprueba pasando las mismas entradas x por ambos grafos y midiendo, por ejemplo, la desviación máxima absoluta: máxₓ ‖f_PyTorch(x) − f_ONNX(x)‖_∞ < ε, con ε del orden de 10⁻⁵ para float32. Esta prueba es imprescindible porque un export puede "compilar" correctamente y aun así alterar la semántica (por dimensiones dinámicas mal trazadas, operadores no soportados o un modo train/eval equivocado): sin verificar paridad, un modelo desplegado podría dar predicciones sutilmente distintas a las validadas.

El **costo de inferencia** se descompone en tres magnitudes que suelen estar en tensión. La **latencia** es el tiempo de una sola pasada hacia adelante (relevante cuando importa responder rápido a cada petición); se reporta con estadísticos robustos como la mediana y percentiles (p50, p95) porque su distribución tiene colas. El **throughput** es el número de muestras procesadas por segundo, que crece al agrupar entradas en lotes: un lote de tamaño B amortiza los costos fijos por invocación y aprovecha el paralelismo del hardware, de modo que throughput ≈ B / latencia_lote, aunque a costa de mayor latencia por muestra individual. El **tamaño del modelo** (en MB) condiciona la memoria y el ancho de banda, decisivos en edge.

La **cuantización** es la palanca principal para reducir ambos, tamaño y latencia. Consiste en representar pesos y activaciones con enteros de baja precisión (típicamente INT8) en lugar de float32, mediante una función afín de escala s y punto cero z: r ≈ s·(q − z), donde r es el valor real y q su representación entera. Al usar 8 bits en vez de 32, la memoria se reduce hasta ≈4× y las operaciones enteras son más rápidas y eficientes energéticamente en el hardware adecuado. El precio es una pérdida de precisión numérica que puede degradar la exactitud; por eso s y z se calibran cuidadosamente y la cuantización se trata como otro compromiso a *medir*, no a asumir. La lectura global: exportación e inferencia optimizada solo son válidas si la paridad se verifica primero y el impacto en exactitud, latencia y tamaño se cuantifica de forma reproducible.

### Qué es un grafo exportado y por qué puede diferir

Exportar no es guardar los pesos: es capturar **la función** que el modelo calcula, en un formato que otro motor pueda ejecutar sin Python. El exportador recorre el modelo con una entrada de ejemplo, registra las operaciones que se ejecutan y las escribe como un grafo de operadores estandarizados junto con sus pesos.

De ahí se sigue la limitación fundamental del método: se captura **el camino que esa entrada recorrió**. Si el modelo tiene control de flujo dependiente de los datos —un `if` sobre un valor del tensor, un bucle cuya longitud depende de la entrada—, la rama no recorrida no queda en el grafo, y el modelo exportado calculará algo distinto para entradas que la habrían tomado. El exportador basado en `torch.export` detecta buena parte de estos casos, pero la regla práctica sigue vigente: un modelo pensado para exportarse se escribe sin lógica dependiente de los valores.

Hay dos diferencias más que explican por qué la paridad numérica nunca es exacta. La primera es que el motor de destino **fusiona operaciones** —convolución + normalización + ReLU en un solo núcleo— y reordena cálculos para ir más rápido; como la suma en punto flotante no es asociativa, (a + b) + c y a + (b + c) difieren en los últimos bits. La segunda es que las **dimensiones dinámicas** deben declararse al exportar: si no se marca el eje del lote como dinámico, el grafo queda fijado al tamaño de la entrada de ejemplo y fallará con cualquier otro.

Por eso la verificación no se hace con igualdad exacta sino con tolerancias, comparando la salida del modelo original y la del exportado sobre un conjunto de entradas:

máx |y_torch − y_onnx| ≤ atol + rtol · |y_torch|,

con valores típicos atol = 10⁻⁵ y rtol = 10⁻³ en float32. Y la comprobación debe hacerse sobre **varias** entradas, incluidos casos extremos y distintos tamaños de lote: verificar con una sola entrada no dice nada sobre las ramas que esa entrada no recorrió.

### Qué hace la cuantización y qué cuesta

La cuantización representa pesos y activaciones con enteros de 8 bits en lugar de flotantes de 32. El mapeo es afín y se define con dos números por tensor:

r ≈ S · (q − Z),   con   S = (r_max − r_min) / (q_max − q_min),

donde r es el valor real, q el entero, S la **escala** y Z el **punto cero**. El beneficio inmediato es un factor 4 de reducción de tamaño; el beneficio mayor, que la aritmética entera es más rápida y consume mucha menos energía en el hardware que la soporta —y que el movimiento de datos, que suele dominar la latencia, se reduce en la misma proporción—.

Las dos variantes se distinguen por cuándo se calculan esas constantes. En la **cuantización dinámica** —la que usa este laboratorio— los pesos se cuantizan una vez al exportar y las activaciones se cuantizan al vuelo en cada inferencia, midiendo su rango en el momento. No requiere datos ni reentrenamiento, y por eso es la opción por defecto. En la **cuantización estática** el rango de las activaciones se estima previamente pasando un conjunto de calibración representativo, lo que elimina el costo de medir en tiempo de ejecución y suele dar más velocidad, a cambio de necesitar datos.

La granularidad importa: una escala por tensor es simple pero pierde precisión si los canales tienen rangos muy distintos; una escala **por canal** en las capas convolucionales conserva bastante más exactitud por un costo mínimo. Y cuando la degradación es inaceptable, queda el **entrenamiento consciente de cuantización**, que simula el redondeo durante el entrenamiento para que la red aprenda a tolerarlo.

La pérdida de exactitud debe medirse, no suponerse, y en el mismo conjunto de evaluación que el modelo original. Reportar «se redujo el tamaño 4×» sin la métrica al lado es reportar la mitad del resultado.

### Cómo se mide la latencia sin engañarse

La medición de tiempos es donde se cometen los errores más fáciles de detectar y más frecuentes.

**Calentamiento.** Las primeras inferencias incluyen la inicialización de núcleos, la reserva de memoria y, en GPU, la compilación de kernels: pueden ser un orden de magnitud más lentas. Se descartan las primeras repeticiones antes de medir.

**Repetición y estadística.** Una sola medición captura ruido del sistema operativo. Se repite decenas de veces y se reporta la **mediana** y un percentil alto —p95 o p99— además de la media, porque en un servicio real lo que determina la experiencia es la cola de la distribución, no el promedio.

**Sincronización.** En GPU las operaciones son asíncronas: medir el tiempo sin sincronizar mide el encolado, no la ejecución, y produce cifras absurdamente buenas.

**Latencia y throughput no son lo mismo.** La latencia es el tiempo de una petición; el throughput, las peticiones por segundo. Aumentar el tamaño de lote mejora el segundo y **empeora** el primero, porque hay que esperar a llenar el lote. Cuál optimizar depende del caso de uso —interactivo o por lotes— y la respuesta no es la misma.

**Condiciones declaradas.** Hardware, número de hilos, versión del motor y tamaño de lote forman parte del resultado: una latencia sin esos datos no es comparable con ninguna otra.

Por último, el **contrato de inferencia** que el laboratorio genera es lo que hace utilizable el artefacto. Un modelo exportado sin su preprocesamiento es una función a la que nadie sabe qué darle: las mismas medias y desviaciones de normalización, el mismo orden y nombre de las variables, el mismo tamaño de imagen, el mismo mapeo de índices a clases. La causa más común de que un modelo funcione en el cuaderno y falle en producción no es el modelo: es un preprocesamiento reimplementado de forma ligeramente distinta al otro lado.

> **La pregunta que deberías poder responder al terminar:** ¿Qué compromisos existen entre tamaño, latencia y precisión?

### Qué se mide y con qué se decide

El laboratorio reporta `accuracy`, `latency_ms`, `throughput`, `model_size_mb`. De todas ellas, la que **decide** qué modelo se conserva es `macro_f1`, y se mide siempre sobre `validation`: es la única forma de que `test` siga siendo una estimación honesta de lo que pasará con datos nuevos.

## 📓 Los tres cuadernos

El laboratorio incluye tres cuadernos Jupyter de **26 celdas** cada uno, de las cuales **13 son de código ejecutable**. Los tres recorren el mismo camino —descargar el dataset real, auditar la partición, entrenar, sellar el experimento y evaluar `test` una vez— y se diferencian en cuánto viene resuelto:

| Cuaderno | Qué trae | Cuándo usarlo |
|---|---|---|
| [📓 `notebook.ipynb`](notebook.ipynb) | El recorrido completo con **todo el código escrito y ejecutable**, celda a celda, intercalado con las explicaciones. | Para leer y ejecutar de principio a fin, entendiendo qué hace cada paso. |
| [✏️ `notebook_student.ipynb`](notebook_student.ipynb) | El mismo recorrido con **2 celdas vaciadas**, marcadas con `# YOUR CODE HERE`, que hay que completar. | Para practicar: se ejecuta igual, pero falla hasta que completas los huecos. |
| [✅ `notebook_solution.ipynb`](notebook_solution.ipynb) | Las celdas anteriores ya resueltas, marcadas con `# SOLUCIÓN DE REFERENCIA`. | Para contrastar tu respuesta después de intentarlo. |

> **Aviso honesto sobre el estado actual.** Hoy `notebook.ipynb` y `notebook_solution.ipynb` tienen **el mismo contenido**, y los ejercicios que los separan del cuaderno de estudiante son **2**. Es decir: el código del laboratorio está completo y es ejecutable en los tres, pero la versión de estudiante todavía no propone una práctica extensa. Está anotado en el [roadmap](../../ROADMAP.md) y se dice aquí para que nadie descubra el límite después de abrir el archivo.

### Cómo abrirlos

Los cuadernos necesitan el extra `notebooks`, que instala Jupyter junto con el paquete:

```bash
pip install -e ".[dev,notebooks]"
jupyter lab labs/23_model_export_and_inference/notebook.ipynb
```

También se abren desde VS Code —con la extensión de Jupyter— haciendo doble clic en el archivo, o desde la interfaz clásica con `jupyter notebook`. El primer arranque descarga el dataset real desde su proveedor, así que la primera ejecución tarda más y **requiere conexión**.

Si prefieres ejecutar sin abrir un cuaderno, `train.py` hace exactamente lo mismo desde la terminal, y la sección de comandos de arriba explica cada opción.

## 🖥️ Los comandos, explicados

Todo el laboratorio se maneja con una sola herramienta de terminal, `neural-labs`, que se instala junto con el paquete (`pip install -e ".[dev,notebooks]"`). Cada subcomando hace **una** cosa del protocolo, y por eso se pueden ejecutar por separado: preparar datos, auditar la partición, entrenar, repetir con varias semillas.

La forma general es siempre la misma:

```bash
neural-labs <subcomando> --lab <identificador> [opciones]
```

| Opción | Valor por defecto | Valores | Qué hace y cuándo cambiarla |
|---|---|---|---|
| `--lab` | `23_model_export_and_inference` | obligatorio | Qué laboratorio se ejecuta. Solo acepta los identificadores del catálogo. |
| `--quick` | desactivado | — | Usa una fracción real del dataset y pocas épocas. Sirve para comprobar la instalación, no para concluir nada sobre el modelo. |
| `--split-seed N` | `42` | entero | Semilla que decide **qué ejemplo cae en qué partición**. Se mantiene fija al comparar modelos. |
| `--training-seed N` | `42` | entero | Semilla de la inicialización de pesos y del barajado de lotes. Es la que se varía para medir cuánta diferencia es simple azar. |
| `--config` | `baseline` | `baseline` · `improved` | Cuál de las dos configuraciones del laboratorio se usa. |
| `--device` | `auto` | `auto` · `cpu` · `cuda` · `mps` | Dónde entrenar. `auto` elige GPU si está disponible y cae a CPU si no. |
| `--training-seeds A B C` | `41 42 43` | enteros | Solo en `benchmark`: la lista de semillas de entrenamiento que se repiten. |
| `--output-dir` | `runs` | ruta | Dónde se escribe el directorio de la ejecución. |

### El script del laboratorio

`labs/23_model_export_and_inference/train.py` no es un programa distinto: fija el `--lab` y delega en la misma herramienta, de modo que estas dos líneas hacen exactamente lo mismo.

```bash
python labs/23_model_export_and_inference/train.py --quick
neural-labs train --lab 23_model_export_and_inference --quick
```

### Lo mismo desde Python

Si prefieres trabajar en un cuaderno o llamar al laboratorio desde tu propio código, la misma ejecución se lanza así. La función devuelve un objeto con el directorio de la ejecución, las métricas y el historial ya cargados:

```python
from neural_labs.experiments import run_lab

resultado = run_lab(
    "23_model_export_and_inference",
    quick=True,          # False para la ejecución completa
    config_name="baseline",
    split_seed=42,       # fija la partición
    training_seed=43,    # varía la inicialización
)

print(resultado.run_dir)   # dónde quedaron los archivos
print(resultado.metrics)   # el diccionario de métricas finales
```

Y para preparar el dataset sin entrenar —útil para inspeccionarlo antes—:

```python
from neural_labs.datasets import prepare_dataset

datos = prepare_dataset("23_model_export_and_inference", quick=True, seed=42)
print(datos.summary)       # tamaño de cada partición y metadatos de la fuente
```

## 🪜 Paso a paso

Cada paso dice qué ocurre por dentro, por qué se hace en ese orden y cómo comprobar que salió bien. El orden no es una convención de estilo: es el que ejecuta el código, y alterarlo invalida el resultado.

### Paso 1 — Traer el dataset real y partirlo

**Qué ocurre.** Descarga `cifar10` desde su proveedor y construye las tres particiones —`train`, `validation` y `test`— con la semilla de partición que le pases.

**Por qué.** La partición se fija **antes** de entrenar y su semilla (`--split-seed`) es independiente de la del entrenamiento (`--training-seed`). Si ambas fueran la misma, no podrías distinguir si un cambio en el resultado viene de haber repartido los datos de otro modo o de haber inicializado los pesos de otro modo.

```bash
neural-labs dataset --lab 23_model_export_and_inference --quick --split-seed 42
```

**Cómo sabes que salió bien.** El comando termina sin error y deja el dataset en caché. Si no hay red, **falla**: no se rellena con datos inventados (`data/dataset.yaml` declara `fallback_to_generated_data: false`).

### Paso 2 — Comprobar que las particiones no se tocan

**Qué ocurre.** Recorre los identificadores de las tres particiones y verifica que ninguno aparece en dos de ellas.

**Por qué.** Una sola fila compartida entre `train` y `test` basta para que la métrica final quede inflada. Es el error más caro del oficio porque no produce ningún síntoma visible: el modelo simplemente «parece» mejor de lo que es.

```bash
neural-labs audit --lab 23_model_export_and_inference --quick --split-seed 42
```

**Cómo sabes que salió bien.** La auditoría reporta cero solapamientos. Si reporta alguno, no sigas: el resto del laboratorio no significaría nada.

### Paso 3 — Mirar los datos antes de modelarlos

**Qué ocurre.** Genera el informe de calidad (valores faltantes, balance de clases, rangos) y el de deriva entre particiones.

**Por qué.** Un desbalance fuerte o una diferencia de distribución entre `train` y `test` cambia qué métrica tiene sentido leer. Descubrirlo después de entrenar obliga a repetirlo todo.

```bash
neural-labs quality --lab 23_model_export_and_inference --quick --split-seed 42
```

**Cómo sabes que salió bien.** Obtienes `data_quality.json` y `drift_report.json`; ábrelos antes de decidir la configuración.

### Paso 4 — Estudiar la teoría del laboratorio

**Qué ocurre.** Leer [`theory.md`](theory.md): la idea central, el desarrollo matemático, los riesgos de interpretación y la bibliografía de la que sale todo eso.

**Por qué.** Sin esto, el entrenamiento es una caja que devuelve números. La teoría es lo que te permite decidir qué mirar y reconocer cuándo un resultado es sospechoso.

**Cómo sabes que salió bien.** Puedes responder, con tus palabras, qué calcula el modelo y por qué esa arquitectura encaja con la tarea `multiclass_classification`.

### Paso 5 — Entrenar y seleccionar con `validation`

**Qué ocurre.** El entrenamiento recorre las épocas midiendo en `validation` después de cada una, y conserva el checkpoint con el mejor valor de `macro_f1`.

**Por qué.** El conjunto de validación existe para tomar decisiones —arquitectura, hiperparámetros, cuándo parar—. Si esas decisiones se tomaran mirando `test`, `test` dejaría de ser una estimación de lo que pasará con datos nuevos y pasaría a ser parte del entrenamiento.

```bash
python labs/23_model_export_and_inference/train.py --quick
# o, con control explícito de las dos semillas:
neural-labs train --lab 23_model_export_and_inference --config baseline --split-seed 42 --training-seed 43
```

**Cómo sabes que salió bien.** En `runs/23_model_export_and_inference/<ejecución>/` aparecen `history.csv` y `best_model.pt`; la métrica de validación mejora respecto de la primera época.

### Paso 6 — Compararte con la línea base

**Qué ocurre.** El repositorio entrena por su cuenta **PyTorch eager** y guarda su resultado, primero sobre `validation` y —solo al final— sobre `test`.

**Por qué.** Una métrica sola no dice si el modelo aporta algo. Puede que un método mucho más simple llegue igual de lejos, y entonces la complejidad añadida no está justificada. Esta comparación es la que convierte un número en un argumento.

**Cómo sabes que salió bien.** Comparas `metrics.json` con `baseline_metrics.json`. Si tu modelo no supera la línea base, el resultado del laboratorio es exactamente ese, y hay que reportarlo.

### Paso 7 — El sellado: `experiment.lock.json`

**Qué ocurre.** Antes de tocar `test`, el código escribe un archivo que fija el laboratorio, las dos semillas, la configuración, la métrica de selección, el checkpoint elegido y el hash del dataset.

**Por qué.** Es la frontera del experimento. A partir de ahí, cualquier ajuste que hagas mirando `test` queda a la vista: el sello dice qué habías decidido *antes* de ver el resultado final. Sin ese archivo, nadie —incluido tú dentro de un mes— puede distinguir una predicción de una racionalización.

**Cómo sabes que salió bien.** El archivo existe y su contenido coincide con lo que creías haber ejecutado.

### Paso 8 — Evaluar `test` una sola vez y medir la incertidumbre

**Qué ocurre.** Con el checkpoint congelado se evalúa `test`, se calculan intervalos de confianza por bootstrap y se desglosan las métricas por subgrupo.

**Por qué.** Un número puntual esconde cuánto podría moverse. Los intervalos dicen si la diferencia con la línea base es real o cabe dentro del ruido; el desglose por subgrupo revela si el promedio está tapando un grupo donde el modelo funciona mucho peor.

**Cómo sabes que salió bien.** Tienes `metrics.json`, `confidence_intervals.json` y `subgroup_metrics.json`, y puedes decir la magnitud de la mejora **y** su incertidumbre.

### Paso 9 — Repetir con varias semillas de entrenamiento

**Qué ocurre.** Se repite el entrenamiento manteniendo **fija** la partición y cambiando solo la semilla de entrenamiento.

**Por qué.** Dos ejecuciones idénticas salvo por la inicialización pueden diferir bastante. Si no mides esa dispersión, corres el riesgo de celebrar una mejora que era una semilla afortunada.

```bash
neural-labs benchmark --lab 23_model_export_and_inference --quick --split-seed 42 --training-seeds 41 42 43
```

**Cómo sabes que salió bien.** Obtienes media y dispersión entre semillas, no un único número.

### Paso 10 — Documentar y cerrar

**Qué ocurre.** Cada ejecución deja `model_card.md` y `report.md`; el plan de experimentos vive en [`experiments.md`](experiments.md) y la rúbrica en [`assessment.md`](assessment.md).

**Por qué.** Un resultado sin su contexto —qué datos, qué decisiones, qué límites— no es reutilizable. La model card es lo que permite que otra persona sepa cuándo *no* debería usar tu modelo.

**Cómo sabes que salió bien.** Completaste la tabla multi-semilla de `experiments.md` y respondiste las preguntas de `assessment.md`.

## 🔍 Cómo leer lo que produce la ejecución

Cada ejecución escribe su propio directorio con nombre único, de modo que dos corridas nunca se pisan. Esto es lo que encontrarás dentro:

| Archivo | Qué contiene y qué mirar |
|---|---|
| `config.yaml` · `environment.json` | La configuración exacta y el entorno (versiones, dispositivo) de la ejecución. |
| `dataset_manifest.json` · `dataset_card.md` | Procedencia, licencia, hash y tamaño de cada partición. |
| `data_quality.json` · `drift_report.json` | Calidad de los datos y diferencias de distribución entre particiones. |
| `baseline_validation_metrics.json` | La línea base medida en `validation`, **antes** de entrenar. |
| `history.csv` · `history.png` | Pérdida y métricas época a época: aquí se ve si el entrenamiento converge o sobreajusta. |
| `experiment.lock.json` | El sello del experimento, escrito antes de tocar `test`. |
| `metrics.json` | El resultado final en `test`, más tiempo, dispositivo y número de parámetros. |
| `baseline_metrics.json` | La línea base en `test`, calculada después de tu evaluación final. |
| `best_model.pt` · `last_model.pt` | El checkpoint elegido por validación y el último, para poder compararlos. |
| `confidence_intervals.json` | Intervalos por bootstrap: cuánto podría moverse cada métrica. |
| `subgroup_metrics.json` | El mismo resultado desglosado por subgrupo, para ver qué esconde el promedio. |
| `predictions.csv` | Predicción por ejemplo, para analizar los errores uno a uno. |
| `confusion_matrix.png` | Qué clases se confunden entre sí. |
| `model_spec.json` · `inference_contract.json` | Qué entrada espera el modelo y qué devuelve: lo que necesita quien lo despliegue. |
| `model_card.md` · `report.md` | La ficha del modelo y el informe legible de la ejecución. |
| `model.onnx` | **Propio de esta ruta.** El modelo exportado; `metrics.json` añade latencia, throughput y tamaño. |

## ⚠️ Dónde suele perderse la gente

- **`--quick` no es una versión pequeña del resultado, es una prueba de que todo corre.** En esta ruta recorta a 1024 ejemplos de entrenamiento · 256 de validación · 256 de test · 2 épocas. Sirve para comprobar la instalación y la descarga; cualquier conclusión sobre el modelo exige la ejecución completa.
- **Cambiar algo después de ver `test` invalida la comparación.** Si al mirar el resultado final se te ocurre una mejora, la ruta correcta es volver a `validation`, decidir allí, y sellar de nuevo.
- **Las dos semillas no son intercambiables.** `--split-seed` cambia *qué datos* caen en cada partición; `--training-seed` cambia *cómo se inicializa y baraja* el entrenamiento. Para comparar modelos se fija la primera y se varía la segunda.
- **Límite declarado de este dataset.** Incluye predicción, exportación y benchmark reproducible.

### Riesgos al interpretar los resultados

Incluye predicción, exportación y benchmark reproducible.

El dataset refleja su proceso de recolección y no representa automáticamente otros períodos, países o poblaciones. Una asociación predictiva no demuestra causalidad.

## ✅ Antes de darlo por terminado

El laboratorio está aprobado cuando se cumplen estos criterios:

- [ ] cero solapamiento entre train, validation y test
- [ ] selección basada únicamente en validation
- [ ] métricas finales acompañadas por incertidumbre
- [ ] conclusiones que distinguen evidencia de suposición

Y cuando tienes estos entregables:

- [ ] notebook ejecutado
- [ ] reporte experimental
- [ ] model card
- [ ] comparación con línea base
- [ ] respuesta a preguntas críticas

El plan experimental con la tabla que hay que completar está en `experiments.md`, y las preguntas con su rúbrica, en `assessment.md`. Ambos documentos se abren desde la barra de navegación de arriba.

### Para ir más lejos

- Cambia una decisión experimental y justifícala con el resultado en `validation`, no con la intuición.
- Analiza los errores por clase o por segmento: casi siempre se concentran en un subconjunto reconocible.
- Compara costo, precisión y latencia; el mejor modelo no siempre es el que gana por décimas.
- Documenta sesgos, limitaciones y usos para los que **no** recomendarías este modelo.

## 📚 Fuentes

La teoría de arriba no es original de este repositorio: se apoya en la literatura de referencia del área y en los papers originales de cada arquitectura. Estas son las obras concretas, y lo que aporta cada una:

> Las referencias apuntan a las obras; no se reproduce su contenido, la redacción es original.

- Huyen — *Designing Machine Learning Systems* (O'Reilly, 2022), capítulos de despliegue y optimización de modelos — compromisos de latencia, throughput y compresión en producción.
- Jacob et al. (2018), *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference*, CVPR — esquema de cuantización INT8 con aritmética entera y su calibración.
- Documentación oficial de ONNX — especificación del formato de intercambio y conjunto de operadores: https://onnx.ai/
- Documentación oficial de PyTorch (`torch.onnx` / `torch.export`) — exportación de modelos y validación de paridad: https://pytorch.org/docs/stable/onnx.html
- Fuente del dataset: https://www.cs.toronto.edu/~kriz/cifar.html
- Consulte `docs/experiment-protocol.md`, `docs/reproducibility.md` y `docs/ethics-and-licenses.md`.

### Los archivos de este laboratorio

Todo lo que necesitas está en esta carpeta. Cada enlace abre el archivo directamente:

| Archivo | Qué es |
|---|---|
| [📄 `README.md`](README.md) | Esta guía. |
| [🧠 `theory.md`](theory.md) | La teoría completa con su bibliografía; es la fuente del apartado teórico de arriba. |
| [🔬 `experiments.md`](experiments.md) | El plan experimental y la tabla multi-semilla que hay que completar. |
| [📝 `assessment.md`](assessment.md) | Las preguntas de evaluación y la rúbrica con la que se corrigen. |
| [📓 `notebook.ipynb`](notebook.ipynb) | El recorrido completo con todo el código escrito y ejecutable. |
| [✏️ `notebook_student.ipynb`](notebook_student.ipynb) | El mismo recorrido con las celdas de ejercicio vacías. |
| [✅ `notebook_solution.ipynb`](notebook_solution.ipynb) | Los ejercicios resueltos, para contrastar. |
| [🖥️ `train.py`](train.py) | El mismo entrenamiento desde la terminal, sin abrir un cuaderno. |
| [🎛️ `configs/baseline.yaml`](configs/baseline.yaml) | Épocas, lote, tasa de aprendizaje y qué recorta `--quick`. |
| [🎚️ `configs/improved.yaml`](configs/improved.yaml) | La configuración ampliada que se compara contra la base. |
| [🗄️ `data/dataset.yaml`](data/dataset.yaml) | Fuente, licencia, política de partición y límites del dataset. |
| [🧾 `lesson.yaml`](lesson.yaml) | Nivel, prerrequisitos, resultados de aprendizaje y criterios. |
| [🖥️ `index.html`](index.html) | Esta misma clase como página autocontenida, para leerla sin conexión. |

Y fuera de la carpeta, tres referencias que esta guía usa: el catálogo `configs/labs.yaml` —de donde salen el objetivo, la línea base y las métricas—, el código `src/neural_labs/experiments.py` —que define el orden de los pasos y los archivos que escribe cada ejecución— y `docs/experiment-protocol.md`, con la regla general del protocolo.

Los datasets se descargan de su proveedor original y conservan su licencia; este repositorio no los redistribuye ni sustituye una descarga fallida por datos generados.

<!-- nav-bottom -->
## 🧭 Navegación del recorrido

| ⬅️ Laboratorio anterior | 🏠 Índice | Laboratorio siguiente ➡️ |
|---|:---:|---|
| [🎯 Incertidumbre y calibración](../../labs/22_uncertainty_calibration/README.md) | [Las 31 rutas](../../parts/README.md) | [🏁 Proyecto final: churn de telecomunicaciones](../../labs/24_capstone_real_project/README.md) |

**En este laboratorio:** **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

⚫ [Parte 6 — Confiar en el modelo y sacarlo del cuaderno](../../parts/06-confianza-y-despliegue.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/23_model_export_and_inference/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
