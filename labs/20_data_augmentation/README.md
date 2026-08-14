# Aumento de datos

<!-- nav-top -->
> 🧭 **Ruta 21 / 31** · 🔴 [Parte 5 — La mecánica fina, ahora en profundidad](../../parts/05-mecanica-fina.md)
>
> [⬅️ 🛡️ Regularización](../../labs/19_regularization_dropout_batchnorm/README.md) · [🏠 Índice de rutas](../../parts/README.md) · [🔍 Explicabilidad ➡️](../../labs/21_explainability/README.md)
>
> **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

## 🎯 Qué vas a hacer aquí

Comparar recortes, volteos y perturbaciones sobre imágenes reales.

Es la **ruta 21 de 31** del recorrido y pertenece a 🔴 la parte 5, *La mecánica fina, ahora en profundidad*. Llegas desde **Regularización** y lo que hagas aquí lo da por supuesto **Explicabilidad**.

Trabajarás con el dataset **`cifar10`** (Torchvision / University of Toronto, licencia: Consultar términos CIFAR-10), y tendrás que superar la línea base **CNN sin aumento**, decidiendo con la métrica `macro_f1` medida sobre `validation`. Nivel intermedio, unas **6 horas** de dedicación.

**Lo que conviene traer resuelto de las rutas anteriores:** PyTorch básico, particiones train/validation/test, métricas de evaluación.

**Al terminar deberías ser capaz de:**

- Comparar recortes, volteos y perturbaciones sobre imágenes reales.
- Preparar y auditar el dataset real cifar10 sin fuga de datos.
- Entrenar y evaluar aumento de datos seleccionado por validation.
- Comparar contra la línea base: CNN sin aumento.
- Interpretar intervalos de confianza, errores y limitaciones.

## 🧠 La teoría de este laboratorio

Esta sección es la explicación completa del tema. No hace falta abrir otro archivo para entender lo que viene después: aquí está la idea, la matemática que la sostiene y sus límites. (El mismo texto vive en `theory.md`, que es la fuente desde la que se genera esta guía, junto con la bibliografía del final.)

### De qué trata

Este laboratorio estudia **aumento de datos seleccionado por validation** usando `cifar10`, un dataset público real procedente de Torchvision / University of Toronto.

El aumento de datos (*data augmentation*) genera, sobre la marcha, variantes transformadas de cada imagen de entrenamiento —recortes, volteos horizontales, cambios de brillo o color— manteniendo su etiqueta. La motivación es sencilla y profunda: si sabemos que la clase "gato" no cambia porque la imagen se desplace unos píxeles o se refleje en espejo, entonces exponer a la red a esas versiones le enseña una **invariancia** que de otro modo tendría que descubrir por sí sola (o no aprendería nunca). Efectivamente, ampliamos el conjunto de entrenamiento con ejemplos plausibles y así reducimos el sobreajuste sin recolectar más datos.

La clave metodológica es que las transformaciones codifican *conocimiento previo* sobre qué variaciones son irrelevantes para la tarea, y ese conocimiento debe ser correcto: un volteo horizontal es inocuo para reconocer animales, pero destruiría la etiqueta de un dígito o de un texto. Por eso el catálogo y la intensidad del aumento se eligen con `validation`, no con `test`, y la evaluación final se hace siempre sobre imágenes de test *sin* aumentar. Sobre `cifar10` (60.000 imágenes a color de 32×32 en 10 clases) comparamos una CNN con y sin aumento para aislar su contribución.

### La matemática, paso a paso

Invariancias y regularización por transformaciones.

Sea T una transformación (recorte, volteo, jitter de color) muestreada de una distribución p(T) que preserva la etiqueta: si (x, y) es un par imagen–clase, queremos que el modelo cumpla f(T(x)) ≈ f(x) para toda T. El aumento de datos convierte el objetivo de entrenamiento en una **esperanza sobre transformaciones**: en lugar de minimizar ℒ(f(x), y) minimizamos 𝔼_{T∼p(T)}[ ℒ(f(T(x)), y) ]. En la práctica esa esperanza se aproxima con Monte Carlo: cada época, cada imagen se ve bajo una T distinta muestreada al azar, de modo que el modelo nunca recibe exactamente el mismo ejemplo dos veces. El efecto es que la red aprende a asignar la misma etiqueta a toda una *órbita* de versiones de x, es decir, aprende invariancia (o al menos robustez) frente a esa familia de transformaciones.

Visto como regularización, el aumento suaviza la función aprendida: promediar la pérdida sobre pequeñas perturbaciones de la entrada penaliza que f cambie bruscamente ante variaciones que la etiqueta considera irrelevantes, lo que empuja hacia fronteras de decisión más estables. Frente a la regularización explícita (weight decay, que actúa sobre los pesos) o al dropout (que actúa sobre las activaciones), el aumento actúa sobre el **espacio de entrada** e inyecta el sesgo inductivo de forma directa e interpretable. Técnicas como Cutout borran una región rectangular de la imagen para forzar el uso de múltiples pistas, mientras que estrategias aprendidas como AutoAugment *buscan* la política de transformaciones p(T) que maximiza la exactitud de validación, en lugar de fijarla a mano.

El riesgo es que una transformación demasiado agresiva rompa la premisa de invariancia y cambie de hecho la etiqueta (un recorte que elimina el objeto, un giro que convierte un 6 en un 9): entonces se inyecta ruido de etiqueta y el rendimiento cae. La condición de validez es siempre la misma: T debe preservar la semántica de la clase. La medición sobre imágenes de test sin aumento garantiza que la mejora reportada refleje generalización real y no un artefacto del procedimiento de evaluación.

### Aumentar datos es declarar una invariancia

Aplicar un recorte aleatorio o un volteo horizontal parece un truco para «tener más datos». Es algo más preciso: es **afirmar** que la etiqueta no cambia bajo esa transformación. Cuando se entrena con pares (T(x), y) para T en un conjunto 𝒯, se le está diciendo al modelo que f(T(x)) debe valer lo mismo que f(x) para toda T de ese conjunto.

De ahí se sigue el criterio para elegir transformaciones, y también el error más caro: una transformación que **sí** cambia la etiqueta enseña algo falso. El volteo horizontal es seguro en CIFAR-10 —un avión volteado sigue siendo un avión— y sería destructivo en reconocimiento de dígitos o de texto, donde distinguir una `b` de una `d` depende justamente de la orientación. Rotar 180° un `6` produce un `9` con etiqueta equivocada. La lista de aumentaciones no es genérica: **depende del dominio**, y justificarla forma parte del diseño experimental.

El efecto sobre la función objetivo es explícito. En vez de minimizar la pérdida sobre los datos observados, se minimiza su esperanza sobre la distribución aumentada,

ℒ_aug(θ) = 𝔼_(x,y) 𝔼_(T∼𝒯) [ ℓ( f_θ(T(x)), y ) ],

y esa esperanza extra actúa como un regularizador: penaliza que la salida varíe cuando la entrada se mueve dentro de las transformaciones declaradas, es decir, **suaviza la función aprendida** en las direcciones que 𝒯 recorre. Por eso el aumento y el weight decay de la ruta 19 no son intercambiables: uno restringe la magnitud de los pesos, el otro restringe la forma de la función en direcciones concretas y elegidas.

Con transformaciones estocásticas aplicadas en cada época, el modelo prácticamente **nunca ve dos veces el mismo ejemplo**, lo que dificulta la memorización. Ese es el mecanismo por el que el aumento reduce la brecha entre entrenamiento y validación, y la razón de que su efecto sea mayor cuanto más pequeño es el conjunto de datos.

### Solo en `train`, y por qué es tan fácil equivocarse

Las transformaciones aleatorias se aplican **únicamente al conjunto de entrenamiento**. Aplicarlas a `validation` o a `test` introduce ruido aleatorio en la evaluación: dos ejecuciones sobre el mismo modelo darían métricas distintas, y la comparación entre configuraciones dejaría de ser válida. La evaluación debe ser determinista.

Esto obliga a separar dos cosas que suelen ir juntas en el mismo bloque de código: el **preprocesamiento** —redimensionar, convertir a tensor, normalizar con las estadísticas de `train`— se aplica a las tres particiones, y el **aumento** —recortes, volteos, perturbaciones de color— solo a una. Mezclarlos en una sola cadena aplicada a todo es el error de implementación más común de esta ruta, y su síntoma es un `validation` inexplicablemente ruidoso.

Existe una excepción deliberada y bien definida: el **aumento en inferencia** (TTA), que consiste en promediar las predicciones sobre varias versiones transformadas de la misma entrada. Suele mejorar algo la métrica a costa de multiplicar el tiempo de inferencia, pero es una **decisión de despliegue** que debe declararse, no un aumento accidental de la evaluación. Si se usa, se usa igual en todas las variantes comparadas.

### Las aumentaciones que mezclan ejemplos

Más allá de las transformaciones geométricas y de color, hay una familia que opera sobre pares de ejemplos y merece conocerse porque cambia también la etiqueta.

**Mixup** interpola linealmente dos ejemplos y sus etiquetas:

x̃ = λ·xᵢ + (1 − λ)·xⱼ,   ỹ = λ·yᵢ + (1 − λ)·yⱼ,   con λ ∼ Beta(α, α).

El modelo aprende así que entre dos clases la transición debe ser gradual, lo que suaviza la frontera de decisión y —efecto documentado— mejora la **calibración** de las probabilidades, que es justo lo que mide la ruta 22. **CutMix** hace lo mismo con parches: recorta una región de una imagen y la pega en otra, ponderando las etiquetas por el área ocupada, lo que preserva la estructura local que el mixup difumina.

Ambas rompen el supuesto de que la etiqueta es una clase única y exigen una pérdida que acepte objetivos blandos; la entropía cruzada la admite sin cambios, sumando los dos términos ponderados.

### Cómo se mide si el aumento aportó

La línea base del laboratorio —**la misma CNN sin aumento**— es la comparación que da sentido a todo, y debe correr con partición, semillas, arquitectura y presupuesto de épocas idénticos. Solo cambia el conjunto 𝒯.

Hay un detalle temporal que conviene anticipar al leer las curvas: el aumento **ralentiza el ajuste al entrenamiento**. Con el mismo número de épocas, la variante aumentada suele mostrar peor métrica de entrenamiento y tardar más en converger, porque cada época presenta un problema ligeramente distinto. Comparar a igual número de épocas puede por tanto **subestimar** su beneficio, y por eso conviene mirar también la curva completa y no solo el punto final.

Y hay dos resultados que reportar por separado. El primero es la métrica **limpia**, sobre datos sin transformar, que dice si el aumento mejoró la generalización ordinaria. El segundo es la **robustez**: evaluar el modelo sobre entradas perturbadas —ruido, desenfoque, cambios de brillo— y medir cuánto cae. Un modelo puede ganar poco en limpio y mucho en robustez, y ese es exactamente el caso en que el aumento vale la pena aunque la tabla principal apenas se mueva.

> **La pregunta que deberías poder responder al terminar:** ¿La mejora proviene de invariancias coherentes?

### Qué se mide y con qué se decide

El laboratorio reporta `accuracy`, `macro_f1`, `robust_accuracy`. De todas ellas, la que **decide** qué modelo se conserva es `macro_f1`, y se mide siempre sobre `validation`: es la única forma de que `test` siga siendo una estimación honesta de lo que pasará con datos nuevos.

## 📓 Los tres cuadernos

El laboratorio se puede recorrer en Jupyter, y trae tres cuadernos con papeles distintos. Los tres siguen el mismo camino —descargar el dataset real, auditar la partición, entrenar, sellar el experimento y evaluar `test` una vez—; lo que cambia es qué te toca escribir a ti:

| Cuaderno | Qué trae | Cuándo usarlo |
|---|---|---|
| [📓 `notebook.ipynb`](notebook.ipynb) | El **recorrido de referencia**: 22 celdas (9 de código) con **todo el código escrito y ejecutable**, intercalado con las explicaciones. No trae ejercicios. | Para leer y ejecutar de principio a fin. |
| [✏️ `notebook_student.ipynb`](notebook_student.ipynb) | El mismo recorrido más **5 ejercicios evaluables** (37 celdas en total). Las celdas de ejercicio están marcadas con `# YOUR CODE HERE` y debajo de cada una hay una comprobación. | Para practicar. |
| [✅ `notebook_solution.ipynb`](notebook_solution.ipynb) | Los mismos ejercicios **resueltos**, marcados con `# SOLUCIÓN DE REFERENCIA`. Cada solución se ejecuta en la integración continua, así que se sabe que pasa. | Para contrastar después de intentarlo. |

### Qué se practica en los ejercicios

Cinco de ellos no son de arquitectura sino del **contrato experimental**, que es lo que distingue a estos laboratorios de un tutorial: auditar la partición, decidir con `validation`, compararse con la línea base, sellar antes de abrir `test` y dejar el plan por escrito. Se resuelven con Python estándar —**sin descargar el dataset ni entrenar**—, así que se corrigen en segundos y sin GPU, y cada uno está parametrizado con los valores de este laboratorio: su métrica de selección, su línea base y su experimento propio.

### Cómo abrirlos

Los cuadernos necesitan el extra `notebooks`, que instala Jupyter junto con el paquete:

```bash
pip install -e ".[dev,notebooks]"
jupyter lab labs/20_data_augmentation/notebook.ipynb
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
| `--lab` | `20_data_augmentation` | obligatorio | Qué laboratorio se ejecuta. Solo acepta los identificadores del catálogo. |
| `--quick` | desactivado | — | Usa una fracción real del dataset y pocas épocas. Sirve para comprobar la instalación, no para concluir nada sobre el modelo. |
| `--split-seed N` | `42` | entero | Semilla que decide **qué ejemplo cae en qué partición**. Se mantiene fija al comparar modelos. |
| `--training-seed N` | `42` | entero | Semilla de la inicialización de pesos y del barajado de lotes. Es la que se varía para medir cuánta diferencia es simple azar. |
| `--config` | `baseline` | `baseline` · `improved` | Cuál de las dos configuraciones del laboratorio se usa. |
| `--device` | `auto` | `auto` · `cpu` · `cuda` · `mps` | Dónde entrenar. `auto` elige GPU si está disponible y cae a CPU si no. |
| `--training-seeds A B C` | `41 42 43` | enteros | Solo en `benchmark`: la lista de semillas de entrenamiento que se repiten. |
| `--output-dir` | `runs` | ruta | Dónde se escribe el directorio de la ejecución. |

### El script del laboratorio

`labs/20_data_augmentation/train.py` no es un programa distinto: fija el `--lab` y delega en la misma herramienta, de modo que estas dos líneas hacen exactamente lo mismo.

```bash
python labs/20_data_augmentation/train.py --quick
neural-labs train --lab 20_data_augmentation --quick
```

### Lo mismo desde Python

Si prefieres trabajar en un cuaderno o llamar al laboratorio desde tu propio código, la misma ejecución se lanza así. La función devuelve un objeto con el directorio de la ejecución, las métricas y el historial ya cargados:

```python
from neural_labs.experiments import run_lab

resultado = run_lab(
    "20_data_augmentation",
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

datos = prepare_dataset("20_data_augmentation", quick=True, seed=42)
print(datos.summary)       # tamaño de cada partición y metadatos de la fuente
```

## 🪜 Paso a paso

Cada paso dice qué ocurre por dentro, por qué se hace en ese orden y cómo comprobar que salió bien. El orden no es una convención de estilo: es el que ejecuta el código, y alterarlo invalida el resultado.

### Paso 1 — Traer el dataset real y partirlo

**Qué ocurre.** Descarga `cifar10` desde su proveedor y construye las tres particiones —`train`, `validation` y `test`— con la semilla de partición que le pases.

**Por qué.** La partición se fija **antes** de entrenar y su semilla (`--split-seed`) es independiente de la del entrenamiento (`--training-seed`). Si ambas fueran la misma, no podrías distinguir si un cambio en el resultado viene de haber repartido los datos de otro modo o de haber inicializado los pesos de otro modo.

```bash
neural-labs dataset --lab 20_data_augmentation --quick --split-seed 42
```

**Cómo sabes que salió bien.** El comando termina sin error y deja el dataset en caché. Si no hay red, **falla**: no se rellena con datos inventados (`data/dataset.yaml` declara `fallback_to_generated_data: false`).

### Paso 2 — Comprobar que las particiones no se tocan

**Qué ocurre.** Recorre los identificadores de las tres particiones y verifica que ninguno aparece en dos de ellas.

**Por qué.** Una sola fila compartida entre `train` y `test` basta para que la métrica final quede inflada. Es el error más caro del oficio porque no produce ningún síntoma visible: el modelo simplemente «parece» mejor de lo que es.

```bash
neural-labs audit --lab 20_data_augmentation --quick --split-seed 42
```

**Cómo sabes que salió bien.** La auditoría reporta cero solapamientos. Si reporta alguno, no sigas: el resto del laboratorio no significaría nada.

### Paso 3 — Mirar los datos antes de modelarlos

**Qué ocurre.** Genera el informe de calidad (valores faltantes, balance de clases, rangos) y el de deriva entre particiones.

**Por qué.** Un desbalance fuerte o una diferencia de distribución entre `train` y `test` cambia qué métrica tiene sentido leer. Descubrirlo después de entrenar obliga a repetirlo todo.

```bash
neural-labs quality --lab 20_data_augmentation --quick --split-seed 42
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
python labs/20_data_augmentation/train.py --quick
# o, con control explícito de las dos semillas:
neural-labs train --lab 20_data_augmentation --config baseline --split-seed 42 --training-seed 43
```

**Cómo sabes que salió bien.** En `runs/20_data_augmentation/<ejecución>/` aparecen `history.csv` y `best_model.pt`; la métrica de validación mejora respecto de la primera época.

### Paso 6 — Compararte con la línea base

**Qué ocurre.** El repositorio entrena por su cuenta **CNN sin aumento** y guarda su resultado, primero sobre `validation` y —solo al final— sobre `test`.

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
neural-labs benchmark --lab 20_data_augmentation --quick --split-seed 42 --training-seeds 41 42 43
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
| `augmentation_comparison.json` | **Propio de esta ruta.** El mismo modelo con y sin aumento de datos. |

## ⚠️ Dónde suele perderse la gente

- **`--quick` no es una versión pequeña del resultado, es una prueba de que todo corre.** En esta ruta recorta a 1024 ejemplos de entrenamiento · 256 de validación · 256 de test · 2 épocas. Sirve para comprobar la instalación y la descarga; cualquier conclusión sobre el modelo exige la ejecución completa.
- **Cambiar algo después de ver `test` invalida la comparación.** Si al mirar el resultado final se te ocurre una mejora, la ruta correcta es volver a `validation`, decidir allí, y sellar de nuevo.
- **Las dos semillas no son intercambiables.** `--split-seed` cambia *qué datos* caen en cada partición; `--training-seed` cambia *cómo se inicializa y baraja* el entrenamiento. Para comparar modelos se fija la primera y se varía la segunda.
- **Límite declarado de este dataset.** La evaluación usa imágenes de test sin aumento.

### Riesgos al interpretar los resultados

La evaluación usa imágenes de test sin aumento.

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

- Géron — *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* (3.ª ed., O'Reilly, 2022), cap. 14 — visión por computador con CNN y uso del aumento de datos para mejorar la generalización.
- Shorten y Khoshgoftaar (2019), *A survey on Image Data Augmentation for Deep Learning*, Journal of Big Data — panorámica sistemática de técnicas de aumento de imágenes.
- DeVries y Taylor (2017), *Improved Regularization of Convolutional Neural Networks with Cutout*, arXiv — borrado aleatorio de regiones como regularizador.
- Cubuk et al. (2019), *AutoAugment: Learning Augmentation Strategies from Data*, CVPR — búsqueda automática de políticas de aumento optimizadas por validación.
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
| [🛡️ Regularización](../../labs/19_regularization_dropout_batchnorm/README.md) | [Las 31 rutas](../../parts/README.md) | [🔍 Explicabilidad](../../labs/21_explainability/README.md) |

**En este laboratorio:** **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🔴 [Parte 5 — La mecánica fina, ahora en profundidad](../../parts/05-mecanica-fina.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/20_data_augmentation/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
