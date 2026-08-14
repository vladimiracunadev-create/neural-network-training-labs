# Fusión de sensores

<!-- nav-top -->
> 🧭 **Ruta 13 / 31** · 🟣 [Parte 3 — Familias especializadas: generar, decidir, relacionar](../../parts/03-familias-especializadas.md)
>
> [⬅️ ♻️ Transfer learning con mascotas](../../labs/11_transfer_learning/README.md) · [🏠 Índice de rutas](../../parts/README.md) · [🎛️ Búsqueda de hiperparámetros ➡️](../../labs/13_hyperparameter_search/README.md)
>
> **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

## 🎯 Qué vas a hacer aquí

Fusionar acelerómetro y giroscopio de smartphones para reconocer actividades.

Es la **ruta 13 de 31** del recorrido y pertenece a 🟣 la parte 3, *Familias especializadas: generar, decidir, relacionar*. Llegas desde **Transfer learning con mascotas** y lo que hagas aquí lo da por supuesto **Búsqueda de hiperparámetros**.

Trabajarás con el dataset **`uci_har`** (UCI, licencia: CC BY 4.0), y tendrás que superar la línea base **Acelerómetro solo, giroscopio solo y regresión logística**, decidiendo con la métrica `macro_f1` medida sobre `validation`. Nivel avanzado, unas **8 horas** de dedicación.

**Lo que conviene traer resuelto de las rutas anteriores:** PyTorch intermedio, optimización, lectura de artículos técnicos.

**Al terminar deberías ser capaz de:**

- Fusionar acelerómetro y giroscopio de smartphones para reconocer actividades.
- Preparar y auditar el dataset real uci_har sin fuga de datos.
- Entrenar y evaluar fusión de ramas de sensores.
- Comparar contra la línea base: Acelerómetro solo, giroscopio solo y regresión logística.
- Interpretar intervalos de confianza, errores y limitaciones.

## 🧠 La teoría de este laboratorio

Esta sección es la explicación completa del tema. No hace falta abrir otro archivo para entender lo que viene después: aquí está la idea, la matemática que la sostiene y sus límites. (El mismo texto vive en `theory.md`, que es la fuente desde la que se genera esta guía, junto con la bibliografía del final.)

### De qué trata

Este laboratorio estudia **fusión de ramas de sensores** usando `uci_har`, un dataset público real procedente de UCI. La intuición es que cada sensor aporta una vista parcial y complementaria del mismo fenómeno físico: el acelerómetro captura la aceleración lineal (útil para distinguir estar de pie de caminar), mientras que el giroscopio mide la velocidad angular (útil para detectar giros y cambios de orientación). Ninguna modalidad basta por sí sola para separar todas las actividades, pero combinadas reducen la ambigüedad.

La **fusión tardía** (late fusion) que se practica aquí procesa cada modalidad con su propia rama (una red que aprende una representación específica del sensor) y luego concatena esas representaciones antes de la cabeza de clasificación. La alternativa de **fusión temprana** (early fusion) concatenaría las señales crudas desde el inicio. La fusión tardía suele ser más robusta cuando las modalidades tienen escalas, ruidos y estructuras temporales distintas, porque permite que cada rama normalice y abstraiga su señal antes de mezclarlas. El aprendizaje de estas representaciones intermedias es el mecanismo central del deep learning aplicado a datos heterogéneos.

El laboratorio también invita a comparar frente a líneas base de una sola modalidad y frente a un modelo lineal. Esto responde la pregunta de si la fusión realmente agrega valor o si una sola rama ya resuelve la tarea. Medir la ganancia marginal de cada sensor es tan importante como alcanzar buena exactitud global.

### La matemática, paso a paso

La entrada se divide en dos vistas de la misma ventana temporal: x_acc (canales del acelerómetro) y x_gyro (canales del giroscopio). Cada rama aplica una función parametrizada que produce un vector de características (embedding):

  h_acc = f_acc(x_acc; θ_acc),  h_gyro = f_gyro(x_gyro; θ_gyro)

La fusión concatena ambas representaciones y la cabeza produce los logits de las K actividades:

  f = [h_acc ; h_gyro],  z = head(f; θ_head),  ŷ = softmax(z)

donde la probabilidad de la clase k es ŷ_k = e^{z_k} / Σⱼ e^{z_j}. El entrenamiento minimiza la entropía cruzada sobre N ejemplos:

  ℒ(θ) = −(1/N) Σᵢ Σₖ y_{i,k} · log ŷ_{i,k}

con y_{i,k} la codificación one-hot de la etiqueta verdadera. La actualización de todos los parámetros θ = {θ_acc, θ_gyro, θ_head} sigue el descenso de gradiente estocástico:

  θ ← θ − η · ∇_θ ℒ

El gradiente ∇_θ ℒ se propaga por retropropagación a través de la cabeza y luego se **reparte** por las dos ramas. Aquí aparece la clave de la fusión: el error retrocede por ambos caminos simultáneamente, de modo que cada rama recibe una señal de aprendizaje condicionada por lo que la otra ya aporta. Por eso el modelo puede aprender a que el giroscopio se especialice en los patrones que el acelerómetro no discrimina bien. La formulación conecta cuatro elementos: representación de entrada (las dos vistas), función del modelo (ramas + cabeza), función de pérdida (entropía cruzada) y regla de actualización (SGD con ∇). El notebook muestra las dimensiones de los tensores y conserva la misma implementación que el script de terminal.

### Dónde se juntan las señales: tres arquitecturas, tres supuestos

«Fusionar» no es una operación única. Hay tres puntos del recorrido donde se pueden unir dos sensores, y cada uno codifica un supuesto distinto sobre el problema.

La **fusión temprana** concatena las señales crudas antes de cualquier procesamiento: x = [x_acc, x_gyro], y a partir de ahí hay un solo modelo. Es la más simple y la que más libertad da al modelo para descubrir relaciones cruzadas desde el primer instante, pero supone que ambas señales están **alineadas y en escalas comparables**, y hace imposible saber qué aportó cada una.

La **fusión tardía** entrena un modelo por sensor y combina sus predicciones —promedio, voto o una capa que aprende los pesos—: ŷ = w_acc·ŷ_acc + w_gyro·ŷ_gyro. Es robusta —si un sensor falla, el otro sigue prediciendo—, permite entrenar cada rama por separado y es trivialmente interpretable. Su límite es serio: al combinar solo al final, **no puede aprender interacciones** entre modalidades. Si distinguir «subir escaleras» de «caminar» requiere cruzar una firma del acelerómetro con una del giroscopio, la fusión tardía nunca verá esa combinación.

La **fusión intermedia** —la de este laboratorio— es el punto medio: cada sensor tiene su propia rama que aprende una representación adaptada a su naturaleza, y esas representaciones se concatenan antes de la cabeza común, h = [h_acc, h_gyro]. Conserva la especialización por sensor y sí permite que la cabeza aprenda interacciones. Es la razón de que sea la opción por defecto en reconocimiento de actividad.

Un detalle del gradiente que conviene tener presente: al concatenar y pasar por una capa densa, cada rama recibe gradiente a través de **su bloque de columnas** de la matriz de la cabeza. Si una modalidad es mucho más informativa, la cabeza tiende a apoyarse en ella, sus gradientes dominan y la otra rama aprende poco —fenómeno conocido como **modalidad perezosa**—. Se detecta comparando el desempeño de cada rama por separado con el del modelo fusionado: si la fusión no supera a la mejor rama sola, no hubo fusión real, hubo una rama trabajando y otra decorando.

### La ablación es el experimento, no un extra

La pregunta que da sentido al laboratorio no es «¿qué exactitud alcanza el modelo fusionado?» sino «¿cuánto aporta cada sensor?», y solo la **ablación** la responde. Se entrenan tres modelos con el mismo protocolo —solo acelerómetro, solo giroscopio, y ambos— y se comparan.

La lectura de los resultados es directa y admite tres desenlaces, todos informativos. Si la fusión supera claramente a las dos ramas individuales, las señales son **complementarias**: cada una aporta información que la otra no tiene. Si la fusión iguala a la mejor rama, son **redundantes** para esta tarea, y el segundo sensor es costo sin beneficio —una conclusión valiosa, porque cada sensor consume batería y ancho de banda—. Y si la fusión es **peor** que la mejor rama sola, el modelo está sobreajustando la dimensión añadida o una de las ramas está introduciendo ruido, lo que apunta a un problema de capacidad o de normalización.

Para que la comparación sea válida, las tres variantes deben compartir partición, semillas, presupuesto de épocas y criterio de parada. Cambiar el número de parámetros al quitar una rama es inevitable; lo que no puede cambiar es nada más.

### El detalle que hace que este dataset se filtre

Aquí hay una trampa específica de las señales de sensores, y es la razón por la que este laboratorio insiste en la auditoría de particiones.

Las ventanas de UCI HAR se construyen con **solapamiento** —cada ventana comparte la mitad de sus muestras con la siguiente—. Si esas ventanas se reparten al azar entre `train` y `test`, dos ventanas casi idénticas acaban una en cada lado, y el modelo evalúa sobre datos que prácticamente ha visto. La exactitud sube varios puntos sin que nada falle a la vista, y el resultado no se sostiene con datos nuevos.

Peor aún: aunque las ventanas no se solaparan, repartir al azar mezcla al **mismo sujeto** entre entrenamiento y evaluación. Cada persona camina, se sienta y sube escaleras con una firma característica, así que el modelo puede reconocer al sujeto y usar eso para predecir su actividad. Lo que se mide entonces no es «reconocer actividades» sino «reconocer a estas doce personas», y el modelo se derrumba con un usuario nuevo. La partición correcta es **por sujeto**: unos sujetos completos para entrenar, otros distintos para evaluar. Es exactamente el escenario que explora la ruta 15, y la razón de que ambos laboratorios usen el mismo dataset con particiones distintas.

Sobre el preprocesamiento, dos reglas que se derivan del mismo principio. La normalización se ajusta **solo con `train`**, y con las estadísticas de cada canal por separado: acelerómetro y giroscopio miden magnitudes físicas distintas —aceleración y velocidad angular— y en unidades distintas, así que estandarizarlos juntos deja a uno dominando la escala del otro. Y si los sensores tuvieran frecuencias de muestreo distintas, habría que remuestrearlos a una rejilla común antes de concatenar; asumir alineación sin comprobarla es una fuente silenciosa de degradación.

> **La pregunta que deberías poder responder al terminar:** ¿Qué modalidad explica cada actividad?

### Qué se mide y con qué se decide

El laboratorio reporta `accuracy`, `balanced_accuracy`, `macro_f1`. De todas ellas, la que **decide** qué modelo se conserva es `macro_f1`, y se mide siempre sobre `validation`: es la única forma de que `test` siga siendo una estimación honesta de lo que pasará con datos nuevos.

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
jupyter lab labs/12_multimodal_fusion/notebook.ipynb
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
| `--lab` | `12_multimodal_fusion` | obligatorio | Qué laboratorio se ejecuta. Solo acepta los identificadores del catálogo. |
| `--quick` | desactivado | — | Usa una fracción real del dataset y pocas épocas. Sirve para comprobar la instalación, no para concluir nada sobre el modelo. |
| `--split-seed N` | `42` | entero | Semilla que decide **qué ejemplo cae en qué partición**. Se mantiene fija al comparar modelos. |
| `--training-seed N` | `42` | entero | Semilla de la inicialización de pesos y del barajado de lotes. Es la que se varía para medir cuánta diferencia es simple azar. |
| `--config` | `baseline` | `baseline` · `improved` | Cuál de las dos configuraciones del laboratorio se usa. |
| `--device` | `auto` | `auto` · `cpu` · `cuda` · `mps` | Dónde entrenar. `auto` elige GPU si está disponible y cae a CPU si no. |
| `--training-seeds A B C` | `41 42 43` | enteros | Solo en `benchmark`: la lista de semillas de entrenamiento que se repiten. |
| `--output-dir` | `runs` | ruta | Dónde se escribe el directorio de la ejecución. |

### El script del laboratorio

`labs/12_multimodal_fusion/train.py` no es un programa distinto: fija el `--lab` y delega en la misma herramienta, de modo que estas dos líneas hacen exactamente lo mismo.

```bash
python labs/12_multimodal_fusion/train.py --quick
neural-labs train --lab 12_multimodal_fusion --quick
```

### Lo mismo desde Python

Si prefieres trabajar en un cuaderno o llamar al laboratorio desde tu propio código, la misma ejecución se lanza así. La función devuelve un objeto con el directorio de la ejecución, las métricas y el historial ya cargados:

```python
from neural_labs.experiments import run_lab

resultado = run_lab(
    "12_multimodal_fusion",
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

datos = prepare_dataset("12_multimodal_fusion", quick=True, seed=42)
print(datos.summary)       # tamaño de cada partición y metadatos de la fuente
```

## 🪜 Paso a paso

Cada paso dice qué ocurre por dentro, por qué se hace en ese orden y cómo comprobar que salió bien. El orden no es una convención de estilo: es el que ejecuta el código, y alterarlo invalida el resultado.

### Paso 1 — Traer el dataset real y partirlo

**Qué ocurre.** Descarga `uci_har` desde su proveedor y construye las tres particiones —`train`, `validation` y `test`— con la semilla de partición que le pases.

**Por qué.** La partición se fija **antes** de entrenar y su semilla (`--split-seed`) es independiente de la del entrenamiento (`--training-seed`). Si ambas fueran la misma, no podrías distinguir si un cambio en el resultado viene de haber repartido los datos de otro modo o de haber inicializado los pesos de otro modo.

```bash
neural-labs dataset --lab 12_multimodal_fusion --quick --split-seed 42
```

**Cómo sabes que salió bien.** El comando termina sin error y deja el dataset en caché. Si no hay red, **falla**: no se rellena con datos inventados (`data/dataset.yaml` declara `fallback_to_generated_data: false`).

### Paso 2 — Comprobar que las particiones no se tocan

**Qué ocurre.** Recorre los identificadores de las tres particiones y verifica que ninguno aparece en dos de ellas.

**Por qué.** Una sola fila compartida entre `train` y `test` basta para que la métrica final quede inflada. Es el error más caro del oficio porque no produce ningún síntoma visible: el modelo simplemente «parece» mejor de lo que es.

```bash
neural-labs audit --lab 12_multimodal_fusion --quick --split-seed 42
```

**Cómo sabes que salió bien.** La auditoría reporta cero solapamientos. Si reporta alguno, no sigas: el resto del laboratorio no significaría nada.

### Paso 3 — Mirar los datos antes de modelarlos

**Qué ocurre.** Genera el informe de calidad (valores faltantes, balance de clases, rangos) y el de deriva entre particiones.

**Por qué.** Un desbalance fuerte o una diferencia de distribución entre `train` y `test` cambia qué métrica tiene sentido leer. Descubrirlo después de entrenar obliga a repetirlo todo.

```bash
neural-labs quality --lab 12_multimodal_fusion --quick --split-seed 42
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
python labs/12_multimodal_fusion/train.py --quick
# o, con control explícito de las dos semillas:
neural-labs train --lab 12_multimodal_fusion --config baseline --split-seed 42 --training-seed 43
```

**Cómo sabes que salió bien.** En `runs/12_multimodal_fusion/<ejecución>/` aparecen `history.csv` y `best_model.pt`; la métrica de validación mejora respecto de la primera época.

### Paso 6 — Compararte con la línea base

**Qué ocurre.** El repositorio entrena por su cuenta **Acelerómetro solo, giroscopio solo y regresión logística** y guarda su resultado, primero sobre `validation` y —solo al final— sobre `test`.

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
neural-labs benchmark --lab 12_multimodal_fusion --quick --split-seed 42 --training-seeds 41 42 43
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

## ⚠️ Dónde suele perderse la gente

- **`--quick` no es una versión pequeña del resultado, es una prueba de que todo corre.** En esta ruta recorta a 1024 ejemplos de entrenamiento · 256 de validación · 256 de test · 2 épocas. Sirve para comprobar la instalación y la descarga; cualquier conclusión sobre el modelo exige la ejecución completa.
- **Cambiar algo después de ver `test` invalida la comparación.** Si al mirar el resultado final se te ocurre una mejora, la ruta correcta es volver a `validation`, decidir allí, y sellar de nuevo.
- **Las dos semillas no son intercambiables.** `--split-seed` cambia *qué datos* caen en cada partición; `--training-seed` cambia *cómo se inicializa y baraja* el entrenamiento. Para comparar modelos se fija la primera y se varía la segunda.
- **Límite declarado de este dataset.** Señales inerciales reales de 30 participantes.

### Riesgos al interpretar los resultados

Señales inerciales reales de 30 participantes.

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

- Goodfellow, Bengio & Courville — *Deep Learning* (MIT Press, 2016) — marco general sobre aprendizaje de representaciones y por qué las capas intermedias abstraen mejor los datos heterogéneos.
- Baltrušaitis, Ahuja & Morency (2019), *Multimodal Machine Learning: A Survey and Taxonomy*, IEEE TPAMI — taxonomía de estrategias de fusión (temprana, tardía, híbrida) y de los desafíos de alinear modalidades.
- Radford et al. (2021), *Learning Transferable Visual Models from Natural Language Supervision (CLIP)*, ICML — ejemplo influyente de aprender un espacio compartido entre modalidades distintas.
- Fuente del dataset: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
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
| [♻️ Transfer learning con mascotas](../../labs/11_transfer_learning/README.md) | [Las 31 rutas](../../parts/README.md) | [🎛️ Búsqueda de hiperparámetros](../../labs/13_hyperparameter_search/README.md) |

**En este laboratorio:** **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🟣 [Parte 3 — Familias especializadas: generar, decidir, relacionar](../../parts/03-familias-especializadas.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/12_multimodal_fusion/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
