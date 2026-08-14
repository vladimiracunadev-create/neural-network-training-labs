# CNN para visión

<!-- nav-top -->
> 🧭 **Ruta 4 / 31** · 🔵 [Parte 2 — Arquitecturas según la forma del dato](../../parts/02-arquitecturas.md)
>
> [⬅️ 🌀 MLP multiclase](../../labs/02_mlp_nonlinear/README.md) · [🏠 Índice de rutas](../../parts/README.md) · [🔁 RNN para texto ➡️](../../labs/04_rnn_sequences/README.md)
>
> **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

## 🎯 Qué vas a hacer aquí

Entrenar una CNN y analizar errores sobre fotografías reales de diez clases.

Es la **ruta 4 de 31** del recorrido y pertenece a 🔵 la parte 2, *Arquitecturas según la forma del dato*. Llegas desde **MLP multiclase** y lo que hagas aquí lo da por supuesto **RNN para texto**.

Trabajarás con el dataset **`cifar10`** (Torchvision / University of Toronto, licencia: Consultar términos CIFAR-10), y tendrás que superar la línea base **Clasificador lineal sobre píxeles**, decidiendo con la métrica `macro_f1` medida sobre `validation`. Nivel intermedio, unas **6 horas** de dedicación.

**Lo que conviene traer resuelto de las rutas anteriores:** PyTorch básico, particiones train/validation/test, métricas de evaluación.

**Al terminar deberías ser capaz de:**

- Entrenar una CNN y analizar errores sobre fotografías reales de diez clases.
- Preparar y auditar el dataset real cifar10 sin fuga de datos.
- Entrenar y evaluar convoluciones para patrones espaciales.
- Comparar contra la línea base: Clasificador lineal sobre píxeles.
- Interpretar intervalos de confianza, errores y limitaciones.

## 🧠 La teoría de este laboratorio

Esta sección es la explicación completa del tema. No hace falta abrir otro archivo para entender lo que viene después: aquí está la idea, la matemática que la sostiene y sus límites. (El mismo texto vive en `theory.md`, que es la fuente desde la que se genera esta guía, junto con la bibliografía del final.)

### De qué trata

Este laboratorio estudia **convoluciones para patrones espaciales** usando `cifar10`, un dataset público real procedente de Torchvision / University of Toronto.

Aplanar una imagen de 32×32×3 y pasarla por un MLP funciona, pero desperdicia la estructura del problema: ignora que los píxeles vecinos están correlacionados y que un objeto es el mismo aunque se desplace unos píxeles. La red convolucional (CNN) incorpora dos sesgos inductivos que encajan con las imágenes: **localidad** (cada neurona mira solo una región pequeña) e **invariancia por traslación** (el mismo filtro se aplica en toda la imagen, compartiendo parámetros). Esto reduce drásticamente el número de pesos y obliga al modelo a aprender detectores de patrones reutilizables.

La consecuencia es una jerarquía de representaciones: las primeras capas aprenden bordes y colores, las intermedias combinan esos bordes en texturas y partes, y las profundas responden a objetos completos. El laboratorio entrena esta jerarquía sobre CIFAR-10 (60.000 fotografías reales de 10 clases) y contrasta contra un clasificador lineal sobre píxeles para hacer patente cuánto aporta explotar la estructura espacial.

### La matemática, paso a paso

La operación central es la **convolución**: un filtro (kernel) de pesos K de tamaño pequeño se desliza sobre la imagen de entrada X y, en cada posición, calcula un producto punto local. Para un filtro de tamaño F×F sobre un mapa de entrada:

Y(i, j) = Σₘ Σₙ X(i+m, j+n)·K(m, n) + b

El mismo K se reutiliza en todas las posiciones (i, j): eso es el **peso compartido** que da la invariancia por traslación y reduce los parámetros de millones (como en una capa densa) a apenas F×F por canal de filtro. Cada filtro produce un **mapa de activación** que señala dónde aparece el patrón que ese filtro detecta. Tras la convolución se aplica una no linealidad (ReLU), sin la cual toda la pila colapsaría a una única convolución lineal.

El tamaño del mapa de salida depende del *stride* s (paso del deslizamiento) y del *padding* p (relleno de bordes): dimensión_salida = ⌊(dimensión_entrada − F + 2p)/s⌋ + 1. El **pooling** (típicamente max-pooling) submuestrea cada región tomando su máximo, lo que reduce la resolución espacial, aporta cierta invariancia a pequeñas traslaciones y amplía el campo receptivo de las capas posteriores sin añadir parámetros.

Un ingrediente decisivo para entrenar redes profundas es la **normalización por lotes** (batch normalization), que estandariza las activaciones de cada canal dentro del minilote:

x̂ = (x − μ_B) / √(σ²_B + ε),    y = γ·x̂ + β

donde μ_B y σ²_B son la media y varianza del lote, ε evita la división por cero, y γ, β son parámetros aprendibles que restauran la capacidad expresiva. Esto estabiliza y acelera el entrenamiento al mantener las activaciones en un rango controlado, reduciendo la sensibilidad a la inicialización.

Tras varias etapas de convolución + ReLU + pooling, los mapas finales se aplanan (o se promedian con global average pooling) y pasan a una cabeza densa que produce los 10 logits, entrenados con entropía cruzada categórica y softmax igual que en el MLP. Una idea arquitectónica que el laboratorio deja como horizonte es la **conexión residual** (ResNet): sumar la entrada de un bloque a su salida, y = F(x) + x, lo que crea un atajo por el que el gradiente fluye sin atenuarse y permite entrenar redes de cientos de capas.

La formulación debe conectarse con cuatro elementos: representación de entrada, función del modelo, función de pérdida y regla de actualización. El notebook muestra las dimensiones de los tensores y conserva la misma implementación que el script de terminal.

### Cuánto ahorra realmente el peso compartido

La frase «reduce drásticamente los parámetros» merece números, porque la magnitud sorprende. Una capa convolucional con C_in canales de entrada, C_out filtros y kernel F×F tiene

|θ|_conv = F²·C_in·C_out + C_out

parámetros, y —esto es lo decisivo— **ese número no depende del tamaño de la imagen**. Una primera capa con 32 filtros de 3×3 sobre las tres bandas de color de CIFAR-10 son 3²·3·32 + 32 = 896 pesos, y serían los mismos 896 sobre una imagen de 1024×1024.

La comparación con una capa densa equivalente es brutal. Aplanar una imagen de 32×32×3 da 3 072 entradas; conectarlas a 32 unidades cuesta 3 072·32 + 32 = 98 336 parámetros, más de cien veces más, y encima esa capa habría perdido toda noción de vecindad: para ella, dos píxeles contiguos y dos píxeles opuestos de la imagen son igual de ajenos.

El costo de cómputo sigue otra ley. Una capa convolucional realiza aproximadamente

FLOPs ≈ H_out · W_out · C_out · F² · C_in

multiplicaciones-acumulaciones, así que **sí** escala con el tamaño de la imagen aunque los parámetros no lo hagan. De ahí una asimetría que conviene tener presente al leer los resultados: en una CNN, la mayor parte de los *parámetros* suele estar en las capas densas finales, mientras que la mayor parte del *tiempo* se va en las capas convolucionales tempranas, que operan sobre mapas grandes. Optimizar el tamaño del modelo y optimizar su latencia no son la misma tarea.

### El campo receptivo: qué ve realmente cada neurona

Una neurona de la primera capa ve una ventana de 3×3 píxeles. La pregunta interesante es cuánto ve una neurona de la quinta capa, y la respuesta la da la recurrencia

R_ℓ = R_(ℓ−1) + (F_ℓ − 1) · Π_(i<ℓ) s_i,

donde F_ℓ es el tamaño de filtro de la capa ℓ y s_i los strides (incluido el del pooling) de las capas anteriores. El producto acumulado es la clave: **cada submuestreo duplica el efecto de todas las convoluciones posteriores**. Con capas 3×3 y un pooling de 2 intercalado cada dos convoluciones, el campo receptivo pasa de 3 a 7, luego a 15, luego a 31 píxeles: en cuatro bloques ya cubre la imagen entera de 32×32.

Esto explica la forma canónica de una CNN. Las primeras capas, con campo receptivo pequeño, solo pueden detectar bordes y colores; las profundas, con campo receptivo comparable a la imagen, pueden responder a objetos completos. Y explica también un error de diseño frecuente: si el campo receptivo final es mucho menor que el objeto que hay que reconocer, ninguna capa llega a «ver» el objeto entero, y añadir filtros no lo arregla —hay que añadir profundidad o submuestreo—.

Sobre el submuestreo hay dos opciones y conviene distinguirlas. El **max-pooling** no tiene parámetros y selecciona el máximo de cada región, quedándose con la evidencia más fuerte de que el patrón está presente e ignorando su posición exacta dentro de la ventana. La **convolución con stride 2** submuestrea aprendiendo cómo combinar la región en vez de imponer el máximo; cuesta parámetros y es la elección de las arquitecturas modernas. Y al final, el **global average pooling** promedia cada mapa completo a un único número, reduciendo un tensor de H×W×C a C valores: elimina de golpe casi todos los parámetros de la cabeza densa y hace la red independiente del tamaño de entrada.

### Normalización por lotes: dos modos y un error clásico

La fórmula de la normalización por lotes esconde una asimetría que causa uno de los fallos más difíciles de diagnosticar. Durante el **entrenamiento**, μ_B y σ²_B se calculan sobre el minilote actual, así que la salida de un ejemplo depende de con qué otros ejemplos comparta lote. Durante la **inferencia** eso sería inaceptable —la predicción no puede depender de quién más esté en el lote— y por eso la capa usa estadísticas acumuladas durante el entrenamiento mediante una media móvil,

μ̂ ← (1 − m)·μ̂ + m·μ_B,   σ̂² ← (1 − m)·σ̂² + m·σ²_B,

con un momento m típico de 0,1. Son parámetros del modelo que no se aprenden por gradiente: se estiman por acumulación.

De ahí el error clásico: evaluar sin poner el modelo en modo evaluación. Si se olvida, la capa sigue normalizando con el lote de test —y si además el lote de test es pequeño o está ordenado por clase, las estadísticas son malísimas— y las métricas salen peores sin que nada falle visiblemente. El mismo interruptor gobierna el dropout, que debe desactivarse en inferencia. Es la razón de que el protocolo de este repositorio evalúe siempre con el modelo congelado y en modo evaluación explícito.

Un efecto secundario que conviene conocer: como las estadísticas del lote introducen ruido en cada ejemplo, la normalización por lotes **regulariza**, y ese efecto se debilita con lotes grandes. Por eso a veces subir el tamaño de lote empeora la generalización aunque el entrenamiento sea más estable, y por eso existen alternativas —normalización de grupo o de capa— para escenarios con lotes muy pequeños.

> **La pregunta que deberías poder responder al terminar:** ¿Qué clases concentran errores visualmente plausibles?

### Qué se mide y con qué se decide

El laboratorio reporta `accuracy`, `balanced_accuracy`, `macro_precision`, `macro_recall`, `macro_f1`. De todas ellas, la que **decide** qué modelo se conserva es `macro_f1`, y se mide siempre sobre `validation`: es la única forma de que `test` siga siendo una estimación honesta de lo que pasará con datos nuevos.

## 🖥️ Los comandos, explicados

Todo el laboratorio se maneja con una sola herramienta de terminal, `neural-labs`, que se instala junto con el paquete (`pip install -e ".[dev,notebooks]"`). Cada subcomando hace **una** cosa del protocolo, y por eso se pueden ejecutar por separado: preparar datos, auditar la partición, entrenar, repetir con varias semillas.

La forma general es siempre la misma:

```bash
neural-labs <subcomando> --lab <identificador> [opciones]
```

| Opción | Valor por defecto | Valores | Qué hace y cuándo cambiarla |
|---|---|---|---|
| `--lab` | `03_cnn_vision` | obligatorio | Qué laboratorio se ejecuta. Solo acepta los identificadores del catálogo. |
| `--quick` | desactivado | — | Usa una fracción real del dataset y pocas épocas. Sirve para comprobar la instalación, no para concluir nada sobre el modelo. |
| `--split-seed N` | `42` | entero | Semilla que decide **qué ejemplo cae en qué partición**. Se mantiene fija al comparar modelos. |
| `--training-seed N` | `42` | entero | Semilla de la inicialización de pesos y del barajado de lotes. Es la que se varía para medir cuánta diferencia es simple azar. |
| `--config` | `baseline` | `baseline` · `improved` | Cuál de las dos configuraciones del laboratorio se usa. |
| `--device` | `auto` | `auto` · `cpu` · `cuda` · `mps` | Dónde entrenar. `auto` elige GPU si está disponible y cae a CPU si no. |
| `--training-seeds A B C` | `41 42 43` | enteros | Solo en `benchmark`: la lista de semillas de entrenamiento que se repiten. |
| `--output-dir` | `runs` | ruta | Dónde se escribe el directorio de la ejecución. |

### El script del laboratorio

`labs/03_cnn_vision/train.py` no es un programa distinto: fija el `--lab` y delega en la misma herramienta, de modo que estas dos líneas hacen exactamente lo mismo.

```bash
python labs/03_cnn_vision/train.py --quick
neural-labs train --lab 03_cnn_vision --quick
```

### Lo mismo desde Python

Si prefieres trabajar en un cuaderno o llamar al laboratorio desde tu propio código, la misma ejecución se lanza así. La función devuelve un objeto con el directorio de la ejecución, las métricas y el historial ya cargados:

```python
from neural_labs.experiments import run_lab

resultado = run_lab(
    "03_cnn_vision",
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

datos = prepare_dataset("03_cnn_vision", quick=True, seed=42)
print(datos.summary)       # tamaño de cada partición y metadatos de la fuente
```

## 🪜 Paso a paso

Cada paso dice qué ocurre por dentro, por qué se hace en ese orden y cómo comprobar que salió bien. El orden no es una convención de estilo: es el que ejecuta el código, y alterarlo invalida el resultado.

### Paso 1 — Traer el dataset real y partirlo

**Qué ocurre.** Descarga `cifar10` desde su proveedor y construye las tres particiones —`train`, `validation` y `test`— con la semilla de partición que le pases.

**Por qué.** La partición se fija **antes** de entrenar y su semilla (`--split-seed`) es independiente de la del entrenamiento (`--training-seed`). Si ambas fueran la misma, no podrías distinguir si un cambio en el resultado viene de haber repartido los datos de otro modo o de haber inicializado los pesos de otro modo.

```bash
neural-labs dataset --lab 03_cnn_vision --quick --split-seed 42
```

**Cómo sabes que salió bien.** El comando termina sin error y deja el dataset en caché. Si no hay red, **falla**: no se rellena con datos inventados (`data/dataset.yaml` declara `fallback_to_generated_data: false`).

### Paso 2 — Comprobar que las particiones no se tocan

**Qué ocurre.** Recorre los identificadores de las tres particiones y verifica que ninguno aparece en dos de ellas.

**Por qué.** Una sola fila compartida entre `train` y `test` basta para que la métrica final quede inflada. Es el error más caro del oficio porque no produce ningún síntoma visible: el modelo simplemente «parece» mejor de lo que es.

```bash
neural-labs audit --lab 03_cnn_vision --quick --split-seed 42
```

**Cómo sabes que salió bien.** La auditoría reporta cero solapamientos. Si reporta alguno, no sigas: el resto del laboratorio no significaría nada.

### Paso 3 — Mirar los datos antes de modelarlos

**Qué ocurre.** Genera el informe de calidad (valores faltantes, balance de clases, rangos) y el de deriva entre particiones.

**Por qué.** Un desbalance fuerte o una diferencia de distribución entre `train` y `test` cambia qué métrica tiene sentido leer. Descubrirlo después de entrenar obliga a repetirlo todo.

```bash
neural-labs quality --lab 03_cnn_vision --quick --split-seed 42
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
python labs/03_cnn_vision/train.py --quick
# o, con control explícito de las dos semillas:
neural-labs train --lab 03_cnn_vision --config baseline --split-seed 42 --training-seed 43
```

**Cómo sabes que salió bien.** En `runs/03_cnn_vision/<ejecución>/` aparecen `history.csv` y `best_model.pt`; la métrica de validación mejora respecto de la primera época.

### Paso 6 — Compararte con la línea base

**Qué ocurre.** El repositorio entrena por su cuenta **Clasificador lineal sobre píxeles** y guarda su resultado, primero sobre `validation` y —solo al final— sobre `test`.

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
neural-labs benchmark --lab 03_cnn_vision --quick --split-seed 42 --training-seeds 41 42 43
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
- **Límite declarado de este dataset.** CIFAR-10 contiene 60.000 imágenes reales de 32×32.

### Riesgos al interpretar los resultados

CIFAR-10 contiene 60.000 imágenes reales de 32×32.

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

- Goodfellow, Bengio & Courville — *Deep Learning* (MIT Press 2016), cap. 9 — redes convolucionales, peso compartido y pooling.
- Géron — *Hands-On Machine Learning* (3.ª ed., O'Reilly 2022), cap. 14 — CNN modernas y visión por computador con frameworks.
- Zhang et al. — *Dive into Deep Learning* (d2l.ai, 2023), cap. 7–8 — convoluciones, arquitecturas clásicas y batch normalization.
- LeCun et al. (1998), *Gradient-based learning applied to document recognition (LeNet)*, Proc. IEEE — primera CNN entrenada de extremo a extremo.
- Krizhevsky, Sutskever & Hinton (2012), *ImageNet Classification with Deep Convolutional Neural Networks (AlexNet)*, NeurIPS — hito que popularizó el aprendizaje profundo en visión.
- He et al. (2016), *Deep Residual Learning for Image Recognition (ResNet)*, CVPR — conexiones residuales para redes muy profundas.
- Fuente del dataset: https://www.cs.toronto.edu/~kriz/cifar.html
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
| [🌀 MLP multiclase](../../labs/02_mlp_nonlinear/README.md) | [Las 31 rutas](../../parts/README.md) | [🔁 RNN para texto](../../labs/04_rnn_sequences/README.md) |

**En este laboratorio:** **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🔵 [Parte 2 — Arquitecturas según la forma del dato](../../parts/02-arquitecturas.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/03_cnn_vision/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
