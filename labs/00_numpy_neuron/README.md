# Neurona con NumPy

<!-- nav-top -->
> 🧭 **Ruta 1 / 31** · 🟢 [Parte 1 — Fundamentos: de la derivada a la primera red](../../parts/01-fundamentos.md)
>
> ⬅️ *inicio del recorrido* · [🏠 Índice de rutas](../../parts/README.md) · [🧩 Perceptrón con PyTorch ➡️](../../labs/01_pytorch_perceptron/README.md)
>
> **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

## 🎯 Qué vas a hacer aquí

Implementar propagación, entropía cruzada y descenso de gradiente sin autograd.

Es la **ruta 1 de 31** del recorrido y pertenece a 🟢 la parte 1, *Fundamentos: de la derivada a la primera red*. Es el punto de partida; después viene **Perceptrón con PyTorch**.

Trabajarás con el dataset **`breast_cancer_wisconsin`** (UCI, licencia: CC BY 4.0), y tendrás que superar la línea base **DummyClassifier y regresión logística de scikit-learn**, decidiendo con la métrica `f1` medida sobre `validation`. Nivel fundamentos, unas **4 horas** de dedicación.

**Lo que conviene traer resuelto de las rutas anteriores:** Python básico, NumPy, álgebra lineal elemental.

**Al terminar deberías ser capaz de:**

- Implementar propagación, entropía cruzada y descenso de gradiente sin autograd.
- Preparar y auditar el dataset real breast_cancer_wisconsin sin fuga de datos.
- Entrenar y evaluar regresión logística implementada sin autograd.
- Comparar contra la línea base: DummyClassifier y regresión logística de scikit-learn.
- Interpretar intervalos de confianza, errores y limitaciones.

## 🧠 La teoría de este laboratorio

Esta sección es la explicación completa del tema. No hace falta abrir otro archivo para entender lo que viene después: aquí está la idea, la matemática que la sostiene y sus límites. (El mismo texto vive en `theory.md`, que es la fuente desde la que se genera esta guía, junto con la bibliografía del final.)

### De qué trata

Este laboratorio estudia **regresión logística implementada sin autograd** usando `breast_cancer_wisconsin`, un dataset público real procedente de UCI.

La regresión logística es la unidad de construcción más simple del aprendizaje profundo: una sola neurona que combina linealmente sus entradas y las pasa por una no linealidad suave. Aquí no delegamos nada en un motor de diferenciación automática; escribimos a mano la propagación hacia adelante, la pérdida y las derivadas. El objetivo pedagógico es doble: entender de dónde salen los gradientes (no aparecen por magia) y comprobar que una neurona bien planteada resuelve un problema clínico real de diagnóstico binario (tumor benigno frente a maligno) a partir de 30 medidas morfológicas del núcleo celular.

Al forzar la derivación explícita, el laboratorio hace visible la cadena completa: cada peso wⱼ tiene una responsabilidad concreta sobre el error, y esa responsabilidad es exactamente lo que el gradiente cuantifica. Cuando en los laboratorios siguientes deleguemos esto en `autograd`, sabremos qué está calculando la máquina por debajo.

### La matemática, paso a paso

El modelo predice la probabilidad de que la clase sea positiva combinando las entradas de forma lineal y aplastando el resultado al intervalo (0, 1) con la función logística (sigmoide):

p(y=1 | x) = σ(z),  con  z = x·w + b = Σⱼ xⱼwⱼ + b,  y  σ(z) = 1 / (1 + e⁻ᶻ)

La sigmoide convierte una puntuación real ilimitada z en una probabilidad. Su forma en "S" comprime valores muy negativos hacia 0 y muy positivos hacia 1, dejando la mayor sensibilidad alrededor de z = 0, donde σ(0) = 0.5 marca la frontera de decisión. El sesgo b desplaza esa frontera y los pesos w orientan el hiperplano separador en el espacio de las 30 características.

Para ajustar los parámetros medimos el desacuerdo con la **entropía cruzada binaria** (equivalente a la log-verosimilitud negativa de un modelo Bernoulli). Para un conjunto de N ejemplos:

L = −(1/N) Σᵢ [ yᵢ·ln(pᵢ) + (1 − yᵢ)·ln(1 − pᵢ) ]

Esta pérdida penaliza con fuerza creciente la confianza equivocada: si el modelo asigna pᵢ ≈ 0 a un caso realmente positivo, ln(pᵢ) → −∞. Elegir entropía cruzada en lugar del error cuadrático no es arbitrario: al combinarla con la sigmoide, el gradiente se simplifica de forma notable y evita las mesetas de aprendizaje que produciría σ′(z) elevada al cuadrado.

El resultado clave, que este laboratorio deriva a mano, es que el gradiente de la pérdida respecto a los parámetros depende solo del **error de predicción** (pᵢ − yᵢ):

∂L/∂wⱼ = (1/N) Σᵢ (pᵢ − yᵢ)·xᵢⱼ    y    ∂L/∂b = (1/N) Σᵢ (pᵢ − yᵢ)

La intuición es transparente: si el modelo predice de más (pᵢ > yᵢ), el gradiente empuja los pesos en dirección contraria a las entradas activas; si predice de menos, los empuja a favor. La magnitud del ajuste es proporcional tanto al error como al valor de la característica, por eso la **escala de las variables importa**: una variable con valores muy grandes domina el gradiente y desestabiliza la convergencia si no se normaliza.

Finalmente, el descenso de gradiente actualiza los parámetros iterativamente con una tasa de aprendizaje η:

w ← w − η·∇_w L    ;    b ← b − η·∂L/∂b

La formulación debe conectarse con cuatro elementos: representación de entrada, función del modelo, función de pérdida y regla de actualización. El notebook muestra las dimensiones de los tensores y conserva la misma implementación que el script de terminal.

> **La pregunta que deberías poder responder al terminar:** ¿Cómo cambia la convergencia al modificar la escala de las variables?

### Qué se mide y con qué se decide

El laboratorio reporta `accuracy`, `balanced_accuracy`, `precision`, `recall`, `f1`, `roc_auc`, `pr_auc`. De todas ellas, la que **decide** qué modelo se conserva es `f1`, y se mide siempre sobre `validation`: es la única forma de que `test` siga siendo una estimación honesta de lo que pasará con datos nuevos.

## 🖥️ Los comandos, explicados

Todo el laboratorio se maneja con una sola herramienta de terminal, `neural-labs`, que se instala junto con el paquete (`pip install -e ".[dev,notebooks]"`). Cada subcomando hace **una** cosa del protocolo, y por eso se pueden ejecutar por separado: preparar datos, auditar la partición, entrenar, repetir con varias semillas.

La forma general es siempre la misma:

```bash
neural-labs <subcomando> --lab <identificador> [opciones]
```

| Opción | Valor por defecto | Valores | Qué hace y cuándo cambiarla |
|---|---|---|---|
| `--lab` | `00_numpy_neuron` | obligatorio | Qué laboratorio se ejecuta. Solo acepta los identificadores del catálogo. |
| `--quick` | desactivado | — | Usa una fracción real del dataset y pocas épocas. Sirve para comprobar la instalación, no para concluir nada sobre el modelo. |
| `--split-seed N` | `42` | entero | Semilla que decide **qué ejemplo cae en qué partición**. Se mantiene fija al comparar modelos. |
| `--training-seed N` | `42` | entero | Semilla de la inicialización de pesos y del barajado de lotes. Es la que se varía para medir cuánta diferencia es simple azar. |
| `--config` | `baseline` | `baseline` · `improved` | Cuál de las dos configuraciones del laboratorio se usa. |
| `--device` | `auto` | `auto` · `cpu` · `cuda` · `mps` | Dónde entrenar. `auto` elige GPU si está disponible y cae a CPU si no. |
| `--training-seeds A B C` | `41 42 43` | enteros | Solo en `benchmark`: la lista de semillas de entrenamiento que se repiten. |
| `--output-dir` | `runs` | ruta | Dónde se escribe el directorio de la ejecución. |

### El script del laboratorio

`labs/00_numpy_neuron/train.py` no es un programa distinto: fija el `--lab` y delega en la misma herramienta, de modo que estas dos líneas hacen exactamente lo mismo.

```bash
python labs/00_numpy_neuron/train.py --quick
neural-labs train --lab 00_numpy_neuron --quick
```

### Lo mismo desde Python

Si prefieres trabajar en un cuaderno o llamar al laboratorio desde tu propio código, la misma ejecución se lanza así. La función devuelve un objeto con el directorio de la ejecución, las métricas y el historial ya cargados:

```python
from neural_labs.experiments import run_lab

resultado = run_lab(
    "00_numpy_neuron",
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

datos = prepare_dataset("00_numpy_neuron", quick=True, seed=42)
print(datos.summary)       # tamaño de cada partición y metadatos de la fuente
```

## 🪜 Paso a paso

Cada paso dice qué ocurre por dentro, por qué se hace en ese orden y cómo comprobar que salió bien. El orden no es una convención de estilo: es el que ejecuta el código, y alterarlo invalida el resultado.

### Paso 1 — Traer el dataset real y partirlo

**Qué ocurre.** Descarga `breast_cancer_wisconsin` desde su proveedor y construye las tres particiones —`train`, `validation` y `test`— con la semilla de partición que le pases.

**Por qué.** La partición se fija **antes** de entrenar y su semilla (`--split-seed`) es independiente de la del entrenamiento (`--training-seed`). Si ambas fueran la misma, no podrías distinguir si un cambio en el resultado viene de haber repartido los datos de otro modo o de haber inicializado los pesos de otro modo.

```bash
neural-labs dataset --lab 00_numpy_neuron --quick --split-seed 42
```

**Cómo sabes que salió bien.** El comando termina sin error y deja el dataset en caché. Si no hay red, **falla**: no se rellena con datos inventados (`data/dataset.yaml` declara `fallback_to_generated_data: false`).

### Paso 2 — Comprobar que las particiones no se tocan

**Qué ocurre.** Recorre los identificadores de las tres particiones y verifica que ninguno aparece en dos de ellas.

**Por qué.** Una sola fila compartida entre `train` y `test` basta para que la métrica final quede inflada. Es el error más caro del oficio porque no produce ningún síntoma visible: el modelo simplemente «parece» mejor de lo que es.

```bash
neural-labs audit --lab 00_numpy_neuron --quick --split-seed 42
```

**Cómo sabes que salió bien.** La auditoría reporta cero solapamientos. Si reporta alguno, no sigas: el resto del laboratorio no significaría nada.

### Paso 3 — Mirar los datos antes de modelarlos

**Qué ocurre.** Genera el informe de calidad (valores faltantes, balance de clases, rangos) y el de deriva entre particiones.

**Por qué.** Un desbalance fuerte o una diferencia de distribución entre `train` y `test` cambia qué métrica tiene sentido leer. Descubrirlo después de entrenar obliga a repetirlo todo.

```bash
neural-labs quality --lab 00_numpy_neuron --quick --split-seed 42
```

**Cómo sabes que salió bien.** Obtienes `data_quality.json` y `drift_report.json`; ábrelos antes de decidir la configuración.

### Paso 4 — Estudiar la teoría del laboratorio

**Qué ocurre.** Leer [`theory.md`](theory.md): la idea central, el desarrollo matemático, los riesgos de interpretación y la bibliografía de la que sale todo eso.

**Por qué.** Sin esto, el entrenamiento es una caja que devuelve números. La teoría es lo que te permite decidir qué mirar y reconocer cuándo un resultado es sospechoso.

**Cómo sabes que salió bien.** Puedes responder, con tus palabras, qué calcula el modelo y por qué esa arquitectura encaja con la tarea `binary_classification`.

### Paso 5 — Entrenar y seleccionar con `validation`

**Qué ocurre.** El entrenamiento recorre las épocas midiendo en `validation` después de cada una, y conserva el checkpoint con el mejor valor de `f1`.

**Por qué.** El conjunto de validación existe para tomar decisiones —arquitectura, hiperparámetros, cuándo parar—. Si esas decisiones se tomaran mirando `test`, `test` dejaría de ser una estimación de lo que pasará con datos nuevos y pasaría a ser parte del entrenamiento.

```bash
python labs/00_numpy_neuron/train.py --quick
# o, con control explícito de las dos semillas:
neural-labs train --lab 00_numpy_neuron --config baseline --split-seed 42 --training-seed 43
```

**Cómo sabes que salió bien.** En `runs/00_numpy_neuron/<ejecución>/` aparecen `history.csv` y `best_model.npz`; la métrica de validación mejora respecto de la primera época.

### Paso 6 — Compararte con la línea base

**Qué ocurre.** El repositorio entrena por su cuenta **DummyClassifier y regresión logística de scikit-learn** y guarda su resultado, primero sobre `validation` y —solo al final— sobre `test`.

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
neural-labs benchmark --lab 00_numpy_neuron --quick --split-seed 42 --training-seeds 41 42 43
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
| `best_model.npz` | El checkpoint elegido por validación. Esta ruta se implementa en NumPy, así que no hay un `.pt` de PyTorch. |
| `confidence_intervals.json` | Intervalos por bootstrap: cuánto podría moverse cada métrica. |
| `subgroup_metrics.json` | El mismo resultado desglosado por subgrupo, para ver qué esconde el promedio. |
| `predictions.csv` | Predicción por ejemplo, para analizar los errores uno a uno. |
| `confusion_matrix.png` | Qué clases se confunden entre sí. |
| `model_spec.json` · `inference_contract.json` | Qué entrada espera el modelo y qué devuelve: lo que necesita quien lo despliegue. |
| `model_card.md` · `report.md` | La ficha del modelo y el informe legible de la ejecución. |
| `last_model.npz` | **Propio de esta ruta.** El estado de la última época, para contrastarlo con el mejor. |

## ⚠️ Dónde suele perderse la gente

- **`--quick` no es una versión pequeña del resultado, es una prueba de que todo corre.** En esta ruta recorta a 1024 ejemplos de entrenamiento · 256 de validación · 256 de test · 2 épocas. Sirve para comprobar la instalación y la descarga; cualquier conclusión sobre el modelo exige la ejecución completa.
- **Cambiar algo después de ver `test` invalida la comparación.** Si al mirar el resultado final se te ocurre una mejora, la ruta correcta es volver a `validation`, decidir allí, y sellar de nuevo.
- **Las dos semillas no son intercambiables.** `--split-seed` cambia *qué datos* caen en cada partición; `--training-seed` cambia *cómo se inicializa y baraja* el entrenamiento. Para comparar modelos se fija la primera y se varía la segunda.
- **Límite declarado de este dataset.** Datos clínicos reales derivados de imágenes digitalizadas de aspirados de masas mamarias.

### Riesgos al interpretar los resultados

Datos clínicos reales derivados de imágenes digitalizadas de aspirados de masas mamarias.

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

- Bishop — *Pattern Recognition and Machine Learning* (1.ª ed., Springer 2006), cap. 4 (modelos lineales para clasificación) — deriva la regresión logística y su verosimilitud.
- Goodfellow, Bengio & Courville — *Deep Learning* (MIT Press 2016), cap. 5–6 — fundamentos de aprendizaje y redes hacia adelante, entropía cruzada y gradientes.
- Géron — *Hands-On Machine Learning* (3.ª ed., O'Reilly 2022), cap. 4 y 10 — regresión logística práctica y la neurona como base de las redes.
- Nielsen — *Neural Networks and Deep Learning* (online, 2015), cap. 1–2 — intuición de la neurona sigmoide y la retropropagación derivada a mano.
- Rosenblatt (1958), *The perceptron: a probabilistic model for information storage and organization in the brain*, Psychological Review — origen histórico de la neurona artificial entrenable.
- Fuente del dataset: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- Consulte `docs/experiment-protocol.md`, `docs/reproducibility.md` y `docs/ethics-and-licenses.md`.

### Cómo comprobar lo que dice esta guía

Ninguna cifra ni afirmación de esta página está escrita de memoria. Cada una se puede verificar en un archivo del repositorio:

| Lo que dice la guía | Dónde comprobarlo |
|---|---|
| Objetivo, línea base, métricas y arquitectura | `configs/labs.yaml` |
| Fuente, licencia, procedencia y límites del dataset | `data/dataset.yaml` |
| Épocas, tamaño de lote, tasa de aprendizaje y recorte de `--quick` | `configs/baseline.yaml` y `configs/improved.yaml` |
| Nivel, prerrequisitos, resultados de aprendizaje y criterios | `lesson.yaml` |
| Opciones de los comandos y sus valores por defecto | `src/neural_labs/cli.py` |
| El orden de los pasos y los archivos que escribe cada ejecución | `src/neural_labs/experiments.py` |
| La teoría y su bibliografía | `theory.md` |
| La regla general del protocolo | `docs/experiment-protocol.md` |

Los datasets se descargan de su proveedor original y conservan su licencia; este repositorio no los redistribuye ni sustituye una descarga fallida por datos generados.

<!-- nav-bottom -->
## 🧭 Navegación del recorrido

| ⬅️ Laboratorio anterior | 🏠 Índice | Laboratorio siguiente ➡️ |
|---|:---:|---|
| *— inicio del recorrido* | [Las 31 rutas](../../parts/README.md) | [🧩 Perceptrón con PyTorch](../../labs/01_pytorch_perceptron/README.md) |

**En este laboratorio:** **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🟢 [Parte 1 — Fundamentos: de la derivada a la primera red](../../parts/01-fundamentos.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/00_numpy_neuron/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
