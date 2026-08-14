# Incertidumbre y calibración

<!-- nav-top -->
> 🧭 **Ruta 23 / 31** · ⚫ [Parte 6 — Confiar en el modelo y sacarlo del cuaderno](../../parts/06-confianza-y-despliegue.md)
>
> [⬅️ 🔍 Explicabilidad](../../labs/21_explainability/README.md) · [🏠 Índice de rutas](../../parts/README.md) · [📦 Exportación e inferencia ➡️](../../labs/23_model_export_and_inference/README.md)
>
> **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

## 🎯 Qué vas a hacer aquí

Medir confianza, Brier score, ECE y temperature scaling.

Es la **ruta 23 de 31** del recorrido y pertenece a ⚫ la parte 6, *Confiar en el modelo y sacarlo del cuaderno*. Llegas desde **Explicabilidad** y lo que hagas aquí lo da por supuesto **Exportación e inferencia**.

Trabajarás con el dataset **`breast_cancer_wisconsin`** (UCI, licencia: CC BY 4.0), y tendrás que superar la línea base **Regresión logística calibrada**, decidiendo con la métrica `f1` medida sobre `validation`. Nivel avanzado, unas **8 horas** de dedicación.

**Lo que conviene traer resuelto de las rutas anteriores:** PyTorch intermedio, optimización, lectura de artículos técnicos.

**Al terminar deberías ser capaz de:**

- Medir confianza, Brier score, ECE y temperature scaling.
- Preparar y auditar el dataset real breast_cancer_wisconsin sin fuga de datos.
- Entrenar y evaluar calibración probabilística.
- Comparar contra la línea base: Regresión logística calibrada.
- Interpretar intervalos de confianza, errores y limitaciones.

## 🧠 La teoría de este laboratorio

Esta sección es la explicación completa del tema. No hace falta abrir otro archivo para entender lo que viene después: aquí está la idea, la matemática que la sostiene y sus límites. (El mismo texto vive en `theory.md`, que es la fuente desde la que se genera esta guía, junto con la bibliografía del final.)

### De qué trata

Este laboratorio estudia **calibración probabilística** usando `breast_cancer_wisconsin`, un dataset público real procedente de UCI.

Un clasificador no solo decide una clase: también emite una **confianza**, la probabilidad que asigna a esa decisión. Un modelo está *calibrado* cuando esa confianza coincide con la frecuencia real de acierto: de todas las predicciones hechas con 80 % de confianza, aproximadamente el 80 % deberían ser correctas. La exactitud responde "¿acierta?"; la calibración responde "¿son creíbles sus probabilidades?", y son cosas distintas: una red puede acertar mucho y aun así ser sistemáticamente *sobreconfiada*, gritando 99 % cuando debería decir 70 %.

Esta distinción es crítica cuando la probabilidad alimenta una decisión posterior —fijar un umbral, priorizar un caso, cuantificar riesgo—. En un dataset de diagnóstico como `breast_cancer_wisconsin`, una confianza mal calibrada puede inducir una falsa sensación de certeza. El laboratorio mide la calidad probabilística con Brier score y ECE, y corrige la sobreconfianza con *temperature scaling*, ajustado en validación y evaluado una sola vez en test.

### La matemática, paso a paso

Calibrar logits z/T en validación y evaluar una vez en test.

Una red de clasificación produce **logits** z (puntuaciones sin normalizar) que se convierten en probabilidades con softmax: pₖ = e^{zₖ} / Σⱼ e^{zⱼ}. Las redes profundas modernas tienden a la **sobreconfianza**: el entrenamiento con entropía cruzada empuja los logits a valores extremos —porque acercarse a probabilidad 1 en la clase correcta reduce la pérdida indefinidamente— y el resultado son probabilidades más agudas de lo que la evidencia justifica. Calibrar no cambia qué clase se predice; cambia cuán *afiladas* son las probabilidades.

El **Brier score** mide el error cuadrático medio entre la probabilidad predicha y el resultado real: BS = (1/N)·Σₙ Σₖ (p_{n,k} − y_{n,k})², donde y es el vector one-hot de la etiqueta. Penaliza a la vez errores de clasificación y de confianza: predecir 0.9 en la clase correcta cuesta menos que predecir 0.6, pero predecir 0.9 en la clase *equivocada* cuesta mucho. Es una *proper scoring rule*: se minimiza reportando las probabilidades verdaderas.

El **Expected Calibration Error (ECE)** cuantifica directamente el desajuste entre confianza y acierto. Se agrupan las predicciones en B intervalos según su confianza; en cada intervalo b se comparan la exactitud observada acc(b) y la confianza media conf(b), y se promedia la brecha ponderando por el número de ejemplos: ECE = Σ_b (|Bᵦ|/N) · |acc(b) − conf(b)|. Un modelo perfectamente calibrado tiene ECE = 0; un valor alto revela sobreconfianza (conf > acc) o subconfianza (conf < acc), visible en el *reliability diagram*.

El **temperature scaling** es la corrección más simple y efectiva: divide todos los logits por un único escalar T > 0 antes del softmax, p̂ₖ = e^{zₖ/T} / Σⱼ e^{zⱼ/T}. Con T > 1 las probabilidades se suavizan (baja la confianza), con T < 1 se agudizan; T = 1 no cambia nada. Como el mismo T multiplica todos los logits, el orden relativo se conserva y por tanto **la exactitud y el ranking (AUC) no cambian**: solo se recalibra la confianza. El valor óptimo T\* se ajusta minimizando la entropía cruzada (o el NLL) sobre el conjunto de **validación**, nunca sobre test —ajustar sobre test contaminaría la evaluación—. Luego se evalúan Brier y ECE una sola vez en test con ese T\* congelado. Es importante entender que temperature scaling captura la incertidumbre *aleatórica* (ruido inherente); no distingue lo que el modelo *no sabe* (incertidumbre epistémica), para lo cual se recurre a MC Dropout o ensambles.

### Qué significa exactamente «estar calibrado»

La definición es precisa y conviene tenerla escrita, porque de ella salen todas las métricas del laboratorio. Un modelo está **perfectamente calibrado** si

P( y = 1 | p̂ = q ) = q   para todo q ∈ [0, 1],

es decir: entre todos los casos a los que asigna probabilidad 0,7, exactamente el 70 % resulta positivo. Es una propiedad de la **frecuencia a largo plazo**, no de los casos individuales, y por eso solo se puede evaluar por grupos.

De ahí sale el **ECE**: como no hay dos predicciones con exactamente el mismo valor, se agrupan en M intervalos y se compara, dentro de cada uno, la confianza media con la exactitud observada:

ECE = Σ_(m=1..M) (|B_m| / N) · | exactitud(B_m) − confianza(B_m) |.

Su punto débil está en la palabra «intervalos»: el resultado depende de M y del criterio de agrupación —anchura fija o igual número de muestras—, así que **un ECE solo es comparable con otro calculado igual**. Reportar el valor sin decir cuántos intervalos se usaron lo hace incomparable, y por eso el laboratorio lo declara junto a la cifra.

El **Brier score** mide otra cosa y por eso se reporta junto al ECE: es el error cuadrático medio sobre probabilidades, Σ(p̂ᵢ − yᵢ)²/N, y admite la descomposición de Murphy en tres términos —fiabilidad, resolución e incertidumbre— donde la fiabilidad es justamente la calibración y la resolución es la capacidad de separar clases. Su consecuencia práctica es la que importa: **un modelo puede mejorar su calibración y empeorar su Brier**, si al aplanar sus probabilidades pierde poder de discriminación. Mirar las dos cifras a la vez es lo que evita esa trampa.

Conviene además distinguir dos cosas que se confunden. La **discriminación** —ordenar bien los casos, que es lo que miden ROC-AUC o F1— y la **calibración** —que los números signifiquen lo que dicen— son independientes: un modelo puede ordenar perfectamente y estar mal calibrado, y viceversa. Por eso este laboratorio decide con `f1` y reporta la calibración aparte: son dos preguntas distintas sobre el mismo modelo.

### Por qué temperature scaling funciona sin estropear nada

La corrección es de un solo parámetro: se divide el logit por T > 0 antes del softmax o la sigmoide,

p̂ = σ(z / T),

y se ajusta T minimizando la log-verosimilitud negativa **sobre `validation`**, con los pesos del modelo congelados.

Su propiedad clave es que **no altera el orden** de las predicciones. Como z/T es una transformación monótona creciente de z para todo T > 0, si un caso tenía mayor puntuación que otro la sigue teniendo. Por tanto ROC-AUC, ranking y —fijando el umbral en la escala correspondiente— la matriz de confusión no cambian: la exactitud y el F1 se conservan, y solo se mueven las probabilidades. Ese es el motivo de que sea la técnica por defecto: mejora la calibración sin poner en riesgo el desempeño ya medido.

La dirección del ajuste también informa. Un T > 1 aplana las probabilidades hacia ½, corrigiendo **sobreconfianza**, que es el defecto habitual de las redes neuronales modernas —entrenadas hasta pérdida muy baja, aprenden a asignar probabilidades extremas—. Un T < 1 las agudiza, corrigiendo subconfianza, más raro. Reportar el valor de T ajustado es reportar cuán descalibrado estaba el modelo original.

Hay dos exigencias sin las cuales el método deja de ser válido. La primera: T se ajusta con un conjunto **que el modelo no usó para entrenar**; ajustarlo con `train` no corrige nada, porque ahí el modelo ya está sobreajustado y sus probabilidades no reflejan el error real. La segunda: la evaluación de la calibración se hace sobre `test`, después del sellado, igual que cualquier otra métrica. Ajustar T mirando `test` produciría una calibración excelente y falsa.

Frente a alternativas, temperature scaling es el caso más simple del **escalado de Platt** —que ajusta a·z + b, dos parámetros— y frente a la **regresión isotónica**, que ajusta una función monótona por tramos y es mucho más flexible, tiene la ventaja de no sobreajustar con conjuntos de validación pequeños: un solo parámetro es difícil de sobreajustar, una función por tramos no.

### Para qué sirve realmente una probabilidad calibrada

La calibración importa cuando la probabilidad **entra en una decisión**, y ahí su valor es concreto.

El caso claro es el umbral óptimo bajo costos asimétricos. Si un falso negativo cuesta c_FN y un falso positivo c_FP, la decisión que minimiza el costo esperado es actuar cuando

p̂ > c_FP / (c_FP + c_FN),

una fórmula que **solo tiene sentido si p̂ es una probabilidad real**. Con un modelo descalibrado, ese umbral se aplica sobre una puntuación que no significa lo que dice, y el costo resultante no es el mínimo. Lo mismo ocurre al combinar el modelo con otras fuentes de información, al derivar casos dudosos a revisión humana por debajo de cierta confianza, o al agregar predicciones de varios modelos: todas esas operaciones suponen que los números son probabilidades.

Y conviene cerrar con lo que la calibración **no** da. Corrige el nivel de confianza sobre la distribución en que se ajustó, no fuera de ella: ante un cambio de distribución, un modelo bien calibrado vuelve a descalibrarse, y por eso la calibración se revisa periódicamente junto con la deriva. Tampoco distingue entre incertidumbre **aleatoria** —ruido irreducible del problema, dos casos idénticos con etiquetas distintas— e incertidumbre **epistémica** —ignorancia del modelo por falta de datos en esa región—. La primera no se puede reducir con más datos; la segunda sí, y separarlas requiere métodos bayesianos o de conjunto que quedan fuera de esta ruta.

> **La pregunta que deberías poder responder al terminar:** ¿Una mayor accuracy implica probabilidades confiables?

### Qué se mide y con qué se decide

El laboratorio reporta `accuracy`, `f1`, `roc_auc`, `brier`, `ece`. De todas ellas, la que **decide** qué modelo se conserva es `f1`, y se mide siempre sobre `validation`: es la única forma de que `test` siga siendo una estimación honesta de lo que pasará con datos nuevos.

## 🖥️ Los comandos, explicados

Todo el laboratorio se maneja con una sola herramienta de terminal, `neural-labs`, que se instala junto con el paquete (`pip install -e ".[dev,notebooks]"`). Cada subcomando hace **una** cosa del protocolo, y por eso se pueden ejecutar por separado: preparar datos, auditar la partición, entrenar, repetir con varias semillas.

La forma general es siempre la misma:

```bash
neural-labs <subcomando> --lab <identificador> [opciones]
```

| Opción | Valor por defecto | Valores | Qué hace y cuándo cambiarla |
|---|---|---|---|
| `--lab` | `22_uncertainty_calibration` | obligatorio | Qué laboratorio se ejecuta. Solo acepta los identificadores del catálogo. |
| `--quick` | desactivado | — | Usa una fracción real del dataset y pocas épocas. Sirve para comprobar la instalación, no para concluir nada sobre el modelo. |
| `--split-seed N` | `42` | entero | Semilla que decide **qué ejemplo cae en qué partición**. Se mantiene fija al comparar modelos. |
| `--training-seed N` | `42` | entero | Semilla de la inicialización de pesos y del barajado de lotes. Es la que se varía para medir cuánta diferencia es simple azar. |
| `--config` | `baseline` | `baseline` · `improved` | Cuál de las dos configuraciones del laboratorio se usa. |
| `--device` | `auto` | `auto` · `cpu` · `cuda` · `mps` | Dónde entrenar. `auto` elige GPU si está disponible y cae a CPU si no. |
| `--training-seeds A B C` | `41 42 43` | enteros | Solo en `benchmark`: la lista de semillas de entrenamiento que se repiten. |
| `--output-dir` | `runs` | ruta | Dónde se escribe el directorio de la ejecución. |

### El script del laboratorio

`labs/22_uncertainty_calibration/train.py` no es un programa distinto: fija el `--lab` y delega en la misma herramienta, de modo que estas dos líneas hacen exactamente lo mismo.

```bash
python labs/22_uncertainty_calibration/train.py --quick
neural-labs train --lab 22_uncertainty_calibration --quick
```

### Lo mismo desde Python

Si prefieres trabajar en un cuaderno o llamar al laboratorio desde tu propio código, la misma ejecución se lanza así. La función devuelve un objeto con el directorio de la ejecución, las métricas y el historial ya cargados:

```python
from neural_labs.experiments import run_lab

resultado = run_lab(
    "22_uncertainty_calibration",
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

datos = prepare_dataset("22_uncertainty_calibration", quick=True, seed=42)
print(datos.summary)       # tamaño de cada partición y metadatos de la fuente
```

## 🪜 Paso a paso

Cada paso dice qué ocurre por dentro, por qué se hace en ese orden y cómo comprobar que salió bien. El orden no es una convención de estilo: es el que ejecuta el código, y alterarlo invalida el resultado.

### Paso 1 — Traer el dataset real y partirlo

**Qué ocurre.** Descarga `breast_cancer_wisconsin` desde su proveedor y construye las tres particiones —`train`, `validation` y `test`— con la semilla de partición que le pases.

**Por qué.** La partición se fija **antes** de entrenar y su semilla (`--split-seed`) es independiente de la del entrenamiento (`--training-seed`). Si ambas fueran la misma, no podrías distinguir si un cambio en el resultado viene de haber repartido los datos de otro modo o de haber inicializado los pesos de otro modo.

```bash
neural-labs dataset --lab 22_uncertainty_calibration --quick --split-seed 42
```

**Cómo sabes que salió bien.** El comando termina sin error y deja el dataset en caché. Si no hay red, **falla**: no se rellena con datos inventados (`data/dataset.yaml` declara `fallback_to_generated_data: false`).

### Paso 2 — Comprobar que las particiones no se tocan

**Qué ocurre.** Recorre los identificadores de las tres particiones y verifica que ninguno aparece en dos de ellas.

**Por qué.** Una sola fila compartida entre `train` y `test` basta para que la métrica final quede inflada. Es el error más caro del oficio porque no produce ningún síntoma visible: el modelo simplemente «parece» mejor de lo que es.

```bash
neural-labs audit --lab 22_uncertainty_calibration --quick --split-seed 42
```

**Cómo sabes que salió bien.** La auditoría reporta cero solapamientos. Si reporta alguno, no sigas: el resto del laboratorio no significaría nada.

### Paso 3 — Mirar los datos antes de modelarlos

**Qué ocurre.** Genera el informe de calidad (valores faltantes, balance de clases, rangos) y el de deriva entre particiones.

**Por qué.** Un desbalance fuerte o una diferencia de distribución entre `train` y `test` cambia qué métrica tiene sentido leer. Descubrirlo después de entrenar obliga a repetirlo todo.

```bash
neural-labs quality --lab 22_uncertainty_calibration --quick --split-seed 42
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
python labs/22_uncertainty_calibration/train.py --quick
# o, con control explícito de las dos semillas:
neural-labs train --lab 22_uncertainty_calibration --config baseline --split-seed 42 --training-seed 43
```

**Cómo sabes que salió bien.** En `runs/22_uncertainty_calibration/<ejecución>/` aparecen `history.csv` y `best_model.pt`; la métrica de validación mejora respecto de la primera época.

### Paso 6 — Compararte con la línea base

**Qué ocurre.** El repositorio entrena por su cuenta **Regresión logística calibrada** y guarda su resultado, primero sobre `validation` y —solo al final— sobre `test`.

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
neural-labs benchmark --lab 22_uncertainty_calibration --quick --split-seed 42 --training-seeds 41 42 43
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
| `calibration.json` | **Propio de esta ruta.** Temperatura ajustada y error de calibración esperado (ECE). |

## ⚠️ Dónde suele perderse la gente

- **`--quick` no es una versión pequeña del resultado, es una prueba de que todo corre.** En esta ruta recorta a 1024 ejemplos de entrenamiento · 256 de validación · 256 de test · 2 épocas. Sirve para comprobar la instalación y la descarga; cualquier conclusión sobre el modelo exige la ejecución completa.
- **Cambiar algo después de ver `test` invalida la comparación.** Si al mirar el resultado final se te ocurre una mejora, la ruta correcta es volver a `validation`, decidir allí, y sellar de nuevo.
- **Las dos semillas no son intercambiables.** `--split-seed` cambia *qué datos* caen en cada partición; `--training-seed` cambia *cómo se inicializa y baraja* el entrenamiento. Para comparar modelos se fija la primera y se varía la segunda.
- **Límite declarado de este dataset.** No constituye una herramienta clínica ni consejo médico.

### Riesgos al interpretar los resultados

No constituye una herramienta clínica ni consejo médico.

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

- Guo et al. (2017), *On Calibration of Modern Neural Networks*, ICML — evidencia de la sobreconfianza de las redes profundas y propuesta de temperature scaling; define ECE y reliability diagrams.
- Gal y Ghahramani (2016), *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning (MC Dropout)*, ICML — estimación de incertidumbre epistémica manteniendo dropout activo en inferencia.
- Lakshminarayanan, Pritzel y Blundell (2017), *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*, NeurIPS — ensambles como estimador robusto de incertidumbre predictiva.
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
| [🔍 Explicabilidad](../../labs/21_explainability/README.md) | [Las 31 rutas](../../parts/README.md) | [📦 Exportación e inferencia](../../labs/23_model_export_and_inference/README.md) |

**En este laboratorio:** **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

⚫ [Parte 6 — Confiar en el modelo y sacarlo del cuaderno](../../parts/06-confianza-y-despliegue.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/22_uncertainty_calibration/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
