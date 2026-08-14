# Regularización

<!-- nav-top -->
> 🧭 **Ruta 20 / 31** · 🔴 [Parte 5 — La mecánica fina, ahora en profundidad](../../parts/05-mecanica-fina.md)
>
> [⬅️ ⚙️ Optimizadores y schedulers](../../labs/18_optimizers_and_schedulers/README.md) · [🏠 Índice de rutas](../../parts/README.md) · [🔄 Aumento de datos ➡️](../../labs/20_data_augmentation/README.md)
>
> **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

## 🎯 Qué vas a hacer aquí

Medir dropout, weight decay y batch normalization.

Es la **ruta 20 de 31** del recorrido y pertenece a 🔴 la parte 5, *La mecánica fina, ahora en profundidad*. Llegas desde **Optimizadores y schedulers** y lo que hagas aquí lo da por supuesto **Aumento de datos**.

Trabajarás con el dataset **`fashion_mnist`** (Torchvision / Zalando Research, licencia: MIT), y tendrás que superar la línea base **MLP sin regularización**, decidiendo con la métrica `macro_f1` medida sobre `validation`. Nivel intermedio, unas **6 horas** de dedicación.

**Lo que conviene traer resuelto de las rutas anteriores:** PyTorch básico, particiones train/validation/test, métricas de evaluación.

**Al terminar deberías ser capaz de:**

- Medir dropout, weight decay y batch normalization.
- Preparar y auditar el dataset real fashion_mnist sin fuga de datos.
- Entrenar y evaluar dropout, batch normalization y weight decay.
- Comparar contra la línea base: MLP sin regularización.
- Interpretar intervalos de confianza, errores y limitaciones.

## 🧠 La teoría de este laboratorio

Esta sección es la explicación completa del tema. No hace falta abrir otro archivo para entender lo que viene después: aquí está la idea, la matemática que la sostiene y sus límites. (El mismo texto vive en `theory.md`, que es la fuente desde la que se genera esta guía, junto con la bibliografía del final.)

### De qué trata

Este laboratorio estudia **dropout, batch normalization y weight decay** usando `fashion_mnist`, un dataset público real procedente de Torchvision / Zalando Research.

Una red con suficiente capacidad puede *memorizar* el conjunto de entrenamiento —incluido su ruido— sin aprender el patrón que generaliza. Esa brecha entre el rendimiento en `train` y en `validation` es la señal del **sobreajuste**. La regularización es el conjunto de técnicas que restringen o perturban el modelo para que prefiera soluciones más simples y estables, sacrificando algo de ajuste en `train` a cambio de un mejor comportamiento en datos no vistos. El laboratorio compara tres mecanismos complementarios sobre imágenes reales de prendas (`fashion_mnist`, 28×28 en escala de grises, 10 clases).

La idea que conecta las tres técnicas es que penalizar la complejidad o introducir ruido controlado durante el entrenamiento actúa como un *prior* hacia funciones suaves. **Weight decay** limita la magnitud de los pesos; **dropout** impide que las neuronas dependan de coadaptaciones frágiles; **batch normalization** estabiliza las distribuciones internas y añade un ruido de mini-lote con efecto regularizador. Medimos su impacto observando cómo cambia la brecha train–validation y la exactitud final.

### La matemática, paso a paso

Regularización explícita e implícita; brecha train-validation.

**Weight decay (regularización L2)** añade a la pérdida un término proporcional al cuadrado de la norma de los pesos: ℒ_total = ℒ_datos + (λ/2)·‖θ‖². Su gradiente es λ·θ, de modo que en cada paso los pesos se contraen ligeramente hacia cero: θ ← θ − η(∇ℒ_datos + λ·θ). La intuición es que pesos grandes producen funciones con curvaturas abruptas que se ajustan al ruido; penalizar ‖θ‖² empuja hacia funciones más planas y suaves. El hiperparámetro λ gradúa el compromiso entre ajuste y simplicidad.

**Dropout** apaga aleatoriamente una fracción p de las activaciones en cada paso de entrenamiento. Formalmente, cada activación se multiplica por una máscara Bernoulli: h̃ = h ⊙ m, con mᵢ ~ Bernoulli(1−p), y se reescala por 1/(1−p) para mantener la esperanza. Al forzar a la red a producir la salida correcta con subconjuntos distintos de neuronas, impide que unas pocas unidades formen "conspiraciones" (coadaptaciones) y reparte la representación de forma redundante. Puede leerse como un promedio implícito sobre un número exponencial de subredes que comparten pesos: en inferencia se usa la red completa sin máscara, aproximando ese ensamble.

**Batch normalization** normaliza cada activación dentro del mini-lote antes de la no linealidad. Para una activación x calcula la media μ_B y varianza σ²_B del lote, normaliza x̂ = (x − μ_B)/√(σ²_B + ε), y luego reescala y desplaza con parámetros aprendidos: y = γ·x̂ + β. Al mantener las distribuciones internas con media y varianza estables reduce el *internal covariate shift*, permite tasas de aprendizaje mayores y hace el entrenamiento menos sensible a la inicialización; los parámetros γ, β devuelven a la red la libertad de recuperar cualquier escala útil. Además, como μ_B y σ²_B dependen del mini-lote, inyectan un ruido estocástico que actúa como regularizador implícito. En inferencia se sustituyen por estadísticas acumuladas durante el entrenamiento, para que la predicción de un ejemplo no dependa de sus compañeros de lote.

La lectura conjunta: weight decay actúa sobre la *magnitud* de los pesos, dropout sobre la *estructura* de las representaciones y batch norm sobre la *escala de las activaciones*. Ninguno elimina el sobreajuste por decreto; cada uno desplaza el equilibrio sesgo–varianza, y el laboratorio mide empíricamente cuál reduce la brecha train–validation sin caer en el subajuste.

### Weight decay: qué supuesto está imponiendo

Añadir (λ/2)·‖θ‖² a la pérdida no es un truco: es exactamente lo que sale de hacer inferencia bayesiana con una **distribución previa gaussiana** sobre los pesos. Maximizando la probabilidad posterior,

log p(θ | D) = log p(D | θ) + log p(θ) + const,

y con p(θ) = 𝒩(0, σ²I), el segundo término es −‖θ‖²/(2σ²), es decir, el término L2 con λ = 1/σ². La lectura es que regularizar equivale a declarar una creencia previa: **los pesos pequeños son más probables que los grandes**, y λ mide cuánta evidencia hace falta para abandonar esa creencia.

Su efecto sobre el gradiente es un encogimiento multiplicativo, θ ← (1 − η·λ)·θ − η·g, que empuja continuamente hacia cero y solo se contrarresta donde los datos lo exigen. Dos consecuencias prácticas: los **sesgos no se regularizan** —desplazan la función, no controlan su complejidad, y encogerlos solo introduce error—, y en Adam hay que usar la forma desacoplada de AdamW por la razón que explica la ruta 18.

### Dropout: por qué se escala y qué apaga exactamente

Durante el entrenamiento, el dropout multiplica cada activación por una máscara de Bernoulli que la anula con probabilidad p. Eso cambia la magnitud esperada de la salida: si h tenía esperanza 𝔼[h], tras el apagado pasa a (1 − p)·𝔼[h]. Si en inferencia —donde no se apaga nada— no se corrigiera, la red recibiría activaciones sistemáticamente mayores que las vistas durante el entrenamiento.

La implementación estándar es el **dropout invertido**: se divide por (1 − p) ya en el entrenamiento,

h̃ = (h ⊙ m) / (1 − p),   con m ~ Bernoulli(1 − p),

de modo que 𝔼[h̃] = 𝔼[h] y la inferencia no necesita corrección alguna: basta con desactivar la capa. Es la razón, otra vez, de que el modo evaluación sea obligatorio al medir.

Conceptualmente, entrenar con dropout equivale a entrenar un **conjunto exponencial** de subredes que comparten pesos —2^n máscaras posibles para n unidades—, y evaluar sin dropout aproxima el promedio de todas ellas. De ahí que sea un método de conjunto barato. Su efecto concreto es impedir la **coadaptación**: como ninguna unidad puede contar con que otra esté presente, cada una debe aportar señal útil por sí sola, y la representación resultante es redundante y más robusta.

Dónde ponerlo importa. En capas densas, con p entre 0,2 y 0,5, funciona bien. En capas convolucionales el dropout puntual es poco efectivo, porque los píxeles vecinos de un mapa de activación están muy correlacionados y apagar uno no elimina la información —la aporta su vecino—: la variante útil es el dropout **por canal**, que apaga mapas de características completos.

### La interacción entre dropout y normalización por lotes

Combinar ambas cosas es tan habitual como problemático, y conviene saber por qué.

El dropout modifica la **varianza** de las activaciones, y la modifica de forma distinta en entrenamiento y en inferencia. La normalización por lotes, por su parte, acumula estadísticas durante el entrenamiento —cuando el dropout está activo— para usarlas en inferencia, cuando ya no lo está. Las estadísticas acumuladas no corresponden entonces a la distribución que la capa ve al evaluar, y esa discrepancia de varianza degrada el resultado. Es la razón de que muchas arquitecturas modernas prescindan del dropout en los bloques convolucionales normalizados y lo reserven para la cabeza densa, o de que se coloque siempre **después** de la normalización y no antes.

Conviene además recordar que la normalización por lotes ya regulariza por sí sola, porque el ruido de las estadísticas del minilote actúa como perturbación estocástica. Ese efecto **se debilita con lotes grandes**, así que la cantidad de regularización efectiva de un modelo depende del tamaño de lote: cambiarlo altera el equilibrio y puede exigir reajustar λ y p. Es una interacción que el diseño experimental debe controlar.

### Cómo se mide si la regularización funcionó

La cifra que hay que mirar no es la métrica de validación sino la **brecha** entre entrenamiento y validación. Un modelo que acierta el 99 % en entrenamiento y el 78 % en validación está memorizando; uno que acierta 84 % y 82 % ha generalizado, aunque su cifra de entrenamiento sea peor. Regularizar consiste precisamente en aceptar peor ajuste a cambio de menor brecha, y ese intercambio debe verse en los números.

De ahí que el experimento correcto no sea «activar dropout y comprobar que mejora», sino barrer p y λ observando **las dos curvas a la vez**. Hay tres desenlaces posibles y todos informan: si la brecha es grande, falta regularización; si ambas curvas son bajas y cercanas, sobra —el modelo está subajustado y la regularización le impide aprender—; y si la brecha ya era pequeña de entrada, el modelo no tenía capacidad excedente y la regularización solo puede empeorarlo.

La **parada temprana** merece contarse como parte del mismo conjunto de herramientas, porque es regularización implícita: detener el entrenamiento cuando la validación deja de mejorar limita cuánto pueden crecer los pesos y, en modelos lineales, se puede demostrar equivalente a una penalización L2 con λ dependiente del número de pasos. Es también la más barata, y por eso el repositorio la aplica por defecto mediante `patience`.

> **La pregunta que deberías poder responder al terminar:** ¿Qué técnica reduce sobreajuste sin subajustar?

### Qué se mide y con qué se decide

El laboratorio reporta `accuracy`, `macro_f1`, `generalization_gap`. De todas ellas, la que **decide** qué modelo se conserva es `macro_f1`, y se mide siempre sobre `validation`: es la única forma de que `test` siga siendo una estimación honesta de lo que pasará con datos nuevos.

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
jupyter lab labs/19_regularization_dropout_batchnorm/notebook.ipynb
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
| `--lab` | `19_regularization_dropout_batchnorm` | obligatorio | Qué laboratorio se ejecuta. Solo acepta los identificadores del catálogo. |
| `--quick` | desactivado | — | Usa una fracción real del dataset y pocas épocas. Sirve para comprobar la instalación, no para concluir nada sobre el modelo. |
| `--split-seed N` | `42` | entero | Semilla que decide **qué ejemplo cae en qué partición**. Se mantiene fija al comparar modelos. |
| `--training-seed N` | `42` | entero | Semilla de la inicialización de pesos y del barajado de lotes. Es la que se varía para medir cuánta diferencia es simple azar. |
| `--config` | `baseline` | `baseline` · `improved` | Cuál de las dos configuraciones del laboratorio se usa. |
| `--device` | `auto` | `auto` · `cpu` · `cuda` · `mps` | Dónde entrenar. `auto` elige GPU si está disponible y cae a CPU si no. |
| `--training-seeds A B C` | `41 42 43` | enteros | Solo en `benchmark`: la lista de semillas de entrenamiento que se repiten. |
| `--output-dir` | `runs` | ruta | Dónde se escribe el directorio de la ejecución. |

### El script del laboratorio

`labs/19_regularization_dropout_batchnorm/train.py` no es un programa distinto: fija el `--lab` y delega en la misma herramienta, de modo que estas dos líneas hacen exactamente lo mismo.

```bash
python labs/19_regularization_dropout_batchnorm/train.py --quick
neural-labs train --lab 19_regularization_dropout_batchnorm --quick
```

### Lo mismo desde Python

Si prefieres trabajar en un cuaderno o llamar al laboratorio desde tu propio código, la misma ejecución se lanza así. La función devuelve un objeto con el directorio de la ejecución, las métricas y el historial ya cargados:

```python
from neural_labs.experiments import run_lab

resultado = run_lab(
    "19_regularization_dropout_batchnorm",
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

datos = prepare_dataset("19_regularization_dropout_batchnorm", quick=True, seed=42)
print(datos.summary)       # tamaño de cada partición y metadatos de la fuente
```

## 🪜 Paso a paso

Cada paso dice qué ocurre por dentro, por qué se hace en ese orden y cómo comprobar que salió bien. El orden no es una convención de estilo: es el que ejecuta el código, y alterarlo invalida el resultado.

### Paso 1 — Traer el dataset real y partirlo

**Qué ocurre.** Descarga `fashion_mnist` desde su proveedor y construye las tres particiones —`train`, `validation` y `test`— con la semilla de partición que le pases.

**Por qué.** La partición se fija **antes** de entrenar y su semilla (`--split-seed`) es independiente de la del entrenamiento (`--training-seed`). Si ambas fueran la misma, no podrías distinguir si un cambio en el resultado viene de haber repartido los datos de otro modo o de haber inicializado los pesos de otro modo.

```bash
neural-labs dataset --lab 19_regularization_dropout_batchnorm --quick --split-seed 42
```

**Cómo sabes que salió bien.** El comando termina sin error y deja el dataset en caché. Si no hay red, **falla**: no se rellena con datos inventados (`data/dataset.yaml` declara `fallback_to_generated_data: false`).

### Paso 2 — Comprobar que las particiones no se tocan

**Qué ocurre.** Recorre los identificadores de las tres particiones y verifica que ninguno aparece en dos de ellas.

**Por qué.** Una sola fila compartida entre `train` y `test` basta para que la métrica final quede inflada. Es el error más caro del oficio porque no produce ningún síntoma visible: el modelo simplemente «parece» mejor de lo que es.

```bash
neural-labs audit --lab 19_regularization_dropout_batchnorm --quick --split-seed 42
```

**Cómo sabes que salió bien.** La auditoría reporta cero solapamientos. Si reporta alguno, no sigas: el resto del laboratorio no significaría nada.

### Paso 3 — Mirar los datos antes de modelarlos

**Qué ocurre.** Genera el informe de calidad (valores faltantes, balance de clases, rangos) y el de deriva entre particiones.

**Por qué.** Un desbalance fuerte o una diferencia de distribución entre `train` y `test` cambia qué métrica tiene sentido leer. Descubrirlo después de entrenar obliga a repetirlo todo.

```bash
neural-labs quality --lab 19_regularization_dropout_batchnorm --quick --split-seed 42
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
python labs/19_regularization_dropout_batchnorm/train.py --quick
# o, con control explícito de las dos semillas:
neural-labs train --lab 19_regularization_dropout_batchnorm --config baseline --split-seed 42 --training-seed 43
```

**Cómo sabes que salió bien.** En `runs/19_regularization_dropout_batchnorm/<ejecución>/` aparecen `history.csv` y `best_model.pt`; la métrica de validación mejora respecto de la primera época.

### Paso 6 — Compararte con la línea base

**Qué ocurre.** El repositorio entrena por su cuenta **MLP sin regularización** y guarda su resultado, primero sobre `validation` y —solo al final— sobre `test`.

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
neural-labs benchmark --lab 19_regularization_dropout_batchnorm --quick --split-seed 42 --training-seeds 41 42 43
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
| `variant_comparison.json` | **Propio de esta ruta.** Una fila por variante comparada, con su métrica de validación. |

## ⚠️ Dónde suele perderse la gente

- **`--quick` no es una versión pequeña del resultado, es una prueba de que todo corre.** En esta ruta recorta a 1024 ejemplos de entrenamiento · 256 de validación · 256 de test · 2 épocas. Sirve para comprobar la instalación y la descarga; cualquier conclusión sobre el modelo exige la ejecución completa.
- **Cambiar algo después de ver `test` invalida la comparación.** Si al mirar el resultado final se te ocurre una mejora, la ruta correcta es volver a `validation`, decidir allí, y sellar de nuevo.
- **Las dos semillas no son intercambiables.** `--split-seed` cambia *qué datos* caen en cada partición; `--training-seed` cambia *cómo se inicializa y baraja* el entrenamiento. Para comparar modelos se fija la primera y se varía la segunda.
- **Límite declarado de este dataset.** Prendas reales normalizadas en 28×28.

### Riesgos al interpretar los resultados

Prendas reales normalizadas en 28×28.

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

- Goodfellow, Bengio y Courville — *Deep Learning* (MIT Press, 2016), cap. 7 — marco general de la regularización en aprendizaje profundo: penalizaciones de norma, dropout y estrategias de generalización.
- Géron — *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* (3.ª ed., O'Reilly, 2022), cap. 11 — técnicas prácticas para entrenar redes profundas, incluidas normalización por lotes y regularización.
- Srivastava et al. (2014), *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*, JMLR — formulación original de dropout y su interpretación como ensamble implícito.
- Ioffe y Szegedy (2015), *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*, ICML — definición de batch normalization y su efecto sobre la estabilidad del entrenamiento.
- Fuente del dataset: https://github.com/zalandoresearch/fashion-mnist
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
| [⚙️ Optimizadores y schedulers](../../labs/18_optimizers_and_schedulers/README.md) | [Las 31 rutas](../../parts/README.md) | [🔄 Aumento de datos](../../labs/20_data_augmentation/README.md) |

**En este laboratorio:** **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🔴 [Parte 5 — La mecánica fina, ahora en profundidad](../../parts/05-mecanica-fina.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/19_regularization_dropout_batchnorm/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
