# Aprendizaje federado por participante

<!-- nav-top -->
> 🧭 **Ruta 16 / 31** · 🟠 [Parte 4 — Entrenar mejor, más barato y sin centralizar datos](../../parts/04-entrenamiento-eficiente.md)
>
> [⬅️ ⚗️ Destilación de conocimiento](../../labs/14_knowledge_distillation/README.md) · [🏠 Índice de rutas](../../parts/README.md) · [∂ Backpropagation manual ➡️](../../labs/16_backpropagation_manual/README.md)
>
> **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

## 🎯 Qué vas a hacer aquí

Aplicar FedAvg usando participantes reales como clientes naturales.

Es la **ruta 16 de 31** del recorrido y pertenece a 🟠 la parte 4, *Entrenar mejor, más barato y sin centralizar datos*. Llegas desde **Destilación de conocimiento** y lo que hagas aquí lo da por supuesto **Backpropagation manual**.

Trabajarás con el dataset **`uci_har_subjects`** (UCI, licencia: CC BY 4.0), y tendrás que superar la línea base **Entrenamiento centralizado**, decidiendo con la métrica `macro_f1` medida sobre `validation`. Nivel avanzado, unas **8 horas** de dedicación.

**Lo que conviene traer resuelto de las rutas anteriores:** PyTorch intermedio, optimización, lectura de artículos técnicos.

**Al terminar deberías ser capaz de:**

- Aplicar FedAvg usando participantes reales como clientes naturales.
- Preparar y auditar el dataset real uci_har_subjects sin fuga de datos.
- Entrenar y evaluar agregación federada de clientes reales.
- Comparar contra la línea base: Entrenamiento centralizado.
- Interpretar intervalos de confianza, errores y limitaciones.

## 🧠 La teoría de este laboratorio

Esta sección es la explicación completa del tema. No hace falta abrir otro archivo para entender lo que viene después: aquí está la idea, la matemática que la sostiene y sus límites. (El mismo texto vive en `theory.md`, que es la fuente desde la que se genera esta guía, junto con la bibliografía del final.)

### De qué trata

Este laboratorio estudia **agregación federada de clientes reales** usando `uci_har_subjects`, un dataset público real procedente de UCI. El aprendizaje federado responde a una tensión práctica: los datos viven distribuidos entre muchos dispositivos o personas (aquí, cada participante del estudio de actividad humana), y por razones de privacidad, ancho de banda o regulación no se pueden centralizar en un servidor. La pregunta es cómo entrenar un modelo global sin mover los datos crudos fuera de su origen.

La respuesta que implementa el laboratorio es **FedAvg** (Federated Averaging). En cada ronda, el servidor envía el modelo actual a un conjunto de clientes; cada cliente entrena localmente unas cuantas épocas con *sus propios* datos y devuelve solo los pesos resultantes (no los datos). El servidor promedia esos pesos, ponderando por la cantidad de datos de cada cliente, y obtiene el nuevo modelo global. Se repite el ciclo. Lo que viaja por la red son parámetros, no ejemplos, lo que reduce la exposición de información sensible.

Una decisión metodológica importante es usar el **identificador real de cada sujeto como cliente natural**, en lugar de trocear los datos al azar. Esto preserva la heterogeneidad genuina: cada persona camina, se sienta y sube escaleras de forma ligeramente distinta, por lo que las distribuciones locales son **no-IID** (no idénticamente distribuidas). Esa heterogeneidad es precisamente lo que hace difícil el aprendizaje federado, y estudiarla con clientes reales es más honesto que fabricar particiones artificiales. La pregunta crítica —qué clientes quedan perjudicados por la agregación— apunta a que un promedio global puede favorecer a la mayoría y degradar a los participantes atípicos.

### La matemática, paso a paso

Hay K clientes; el cliente k posee n_k ejemplos y el total es n = Σₖ n_k. Cada cliente define una pérdida local promedio sobre sus datos, F_k(w). El objetivo global es la pérdida ponderada por tamaño de dataset:

  F(w) = Σₖ (n_k / n) · F_k(w)

FedAvg optimiza F(w) sin acceder a los datos crudos. En la ronda t, partiendo del modelo global w_t:

1. El servidor envía w_t a los clientes seleccionados.
2. Cada cliente hace E épocas de descenso de gradiente local, w_k ← w_k − η · ∇ F_k(w_k), partiendo de w_k = w_t, y obtiene w_k^{(t+1)}.
3. El servidor agrega por promedio ponderado:

  w_{t+1} = Σₖ (n_k / n) · w_k^{(t+1)}

La ponderación n_k/n hace que un cliente con más datos influya proporcionalmente más en el modelo global, lo que equivale a tratar por igual a cada *ejemplo* aunque estén repartidos entre clientes. Un caso límite ilumina la fórmula: si cada cliente diera un solo paso de gradiente completo (E = 1, batch = todos sus datos), el promedio de sus actualizaciones locales coincide exactamente con un paso de gradiente sobre F(w) centralizada. Con E > 1, los clientes se alejan localmente antes de promediar; ese **desvío del cliente** (client drift) es mayor cuanto más no-IID son los datos, y explica por qué FedAvg puede converger más lento o de forma menos estable que el entrenamiento centralizado.

Por eso la línea base natural es el entrenamiento centralizado, y una métrica clave es la *dispersión* de la exactitud entre clientes (client_accuracy_std): no basta con una buena media global si algunos participantes quedan sistemáticamente mal servidos. La formulación conecta cuatro elementos: representación de entrada, función del modelo, función de pérdida local F_k y regla de actualización (SGD local + agregación con Σ). El notebook muestra las dimensiones de los tensores y conserva la misma implementación que el script de terminal.

### Por qué los datos no-IID rompen el promedio

FedAvg parece inofensivo: cada cliente entrena localmente y el servidor promedia los pesos. Funciona bien cuando los clientes tienen datos parecidos, y se degrada exactamente en la medida en que no lo son. Conviene ver por qué.

Si todos los clientes tuvieran datos extraídos de la misma distribución (caso **IID**), el gradiente local sería un estimador insesgado del gradiente global y promediar actualizaciones sería casi equivalente a entrenar de forma centralizada. Pero en este dataset cada cliente es una **persona**, y las personas no son intercambiables: una realiza sobre todo actividades sedentarias, otra camina mucho, y sus etiquetas están desbalanceadas de formas distintas. Cada cliente optimiza entonces un objetivo local F_k(θ) cuyo mínimo está en un lugar distinto del mínimo global F(θ) = Σ_k (n_k/n)·F_k(θ).

La consecuencia se llama **desvío del cliente**: con E épocas locales, cada modelo se aleja hacia *su* óptimo antes de que el servidor promedie. Cuanto mayor es E, más lejos llegan y más se contradicen las direcciones al juntarlas; el modelo promediado puede quedar peor que cualquiera de los locales, y en el caso extremo el entrenamiento oscila sin converger. Ahí está el compromiso central de FedAvg, y es contraintuitivo: **más cómputo local no es mejor**. Aumentar E ahorra rondas de comunicación —que es el recurso caro— pero incrementa el desvío. El número de épocas locales es, por tanto, un hiperparámetro experimental, no un detalle de implementación.

El promedio, además, es **ponderado por el tamaño del cliente**:

θ_(t+1) = Σ_k (n_k / n) · θ_k^(t),

lo que reproduce el objetivo global correcto pero da voz proporcional al número de ejemplos. Un participante con muchos datos domina el modelo resultante, y uno con pocos apenas influye: es una decisión de diseño con consecuencias de equidad, no solo de optimización.

### Qué hay que medir además del promedio

La métrica global de un modelo federado esconde justo lo que el enfoque debería vigilar. Un modelo puede alcanzar un buen promedio y funcionar muy mal para un subconjunto de participantes —típicamente aquellos cuya distribución se aleja más de la mayoritaria—, y ese fallo es invisible en la cifra agregada.

Por eso este laboratorio reporta el desempeño **por participante** y no solo el global. Las cifras que conviene mirar son la dispersión entre clientes, el peor cliente y la brecha entre el mejor y el peor: son la traducción operativa de «¿a quién le funciona esto?». Una mejora del promedio conseguida a costa de empeorar al peor cliente rara vez es aceptable en un despliegue real.

La comparación honesta necesita además dos referencias. Arriba, el **modelo centralizado** entrenado con todos los datos juntos: marca el techo, y la diferencia con él es el precio de no centralizar. Abajo, los **modelos puramente locales**, uno por cliente entrenado solo con sus datos: si el federado no los supera, el participante no gana nada colaborando, y la propuesta se cae. Entre esos dos límites es donde el resultado significa algo.

### Lo que la federación protege y lo que no

Es importante ser preciso en esto, porque es la motivación del enfoque y también su malentendido más común. Que los datos crudos no salgan del dispositivo **no equivale a privacidad**. Lo que se transmite —actualizaciones de pesos o gradientes— es una función de los datos y filtra información sobre ellos: se han demostrado ataques de inferencia de pertenencia, que determinan si un ejemplo concreto estuvo en el entrenamiento, y ataques de reconstrucción que recuperan aproximaciones de las entradas a partir de los gradientes.

Las defensas existen y tienen un costo explícito. La **privacidad diferencial** añade ruido calibrado a las actualizaciones y ofrece una garantía formal con un presupuesto ε, a cambio de exactitud. La **agregación segura** impide que el servidor vea las actualizaciones individuales y solo le permite obtener la suma, a cambio de protocolo criptográfico y coordinación. Ninguna es gratis, y este laboratorio no las implementa: se limita a mostrar el mecanismo de FedAvg, y por eso su alcance debe declararse tal cual —una demostración del algoritmo, no un sistema con garantías de privacidad—.

El otro costo que conviene contabilizar es la **comunicación**. Cada ronda transmite el modelo completo en ambos sentidos, así que el tráfico total es del orden de 2 · |θ| · K · R bytes para K clientes y R rondas. Con modelos grandes es el cuello de botella dominante, muy por encima del cómputo, y es lo que motiva las técnicas de compresión y cuantización de actualizaciones. Reportar el número de rondas hasta alcanzar cierta exactitud es, en federado, tan relevante como reportar la exactitud misma.

> **La pregunta que deberías poder responder al terminar:** ¿Qué clientes quedan perjudicados por la agregación?

### Qué se mide y con qué se decide

El laboratorio reporta `accuracy`, `macro_f1`, `client_accuracy_std`. De todas ellas, la que **decide** qué modelo se conserva es `macro_f1`, y se mide siempre sobre `validation`: es la única forma de que `test` siga siendo una estimación honesta de lo que pasará con datos nuevos.

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
jupyter lab labs/15_federated_learning/notebook.ipynb
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
| `--lab` | `15_federated_learning` | obligatorio | Qué laboratorio se ejecuta. Solo acepta los identificadores del catálogo. |
| `--quick` | desactivado | — | Usa una fracción real del dataset y pocas épocas. Sirve para comprobar la instalación, no para concluir nada sobre el modelo. |
| `--split-seed N` | `42` | entero | Semilla que decide **qué ejemplo cae en qué partición**. Se mantiene fija al comparar modelos. |
| `--training-seed N` | `42` | entero | Semilla de la inicialización de pesos y del barajado de lotes. Es la que se varía para medir cuánta diferencia es simple azar. |
| `--config` | `baseline` | `baseline` · `improved` | Cuál de las dos configuraciones del laboratorio se usa. |
| `--device` | `auto` | `auto` · `cpu` · `cuda` · `mps` | Dónde entrenar. `auto` elige GPU si está disponible y cae a CPU si no. |
| `--training-seeds A B C` | `41 42 43` | enteros | Solo en `benchmark`: la lista de semillas de entrenamiento que se repiten. |
| `--output-dir` | `runs` | ruta | Dónde se escribe el directorio de la ejecución. |

### El script del laboratorio

`labs/15_federated_learning/train.py` no es un programa distinto: fija el `--lab` y delega en la misma herramienta, de modo que estas dos líneas hacen exactamente lo mismo.

```bash
python labs/15_federated_learning/train.py --quick
neural-labs train --lab 15_federated_learning --quick
```

### Lo mismo desde Python

Si prefieres trabajar en un cuaderno o llamar al laboratorio desde tu propio código, la misma ejecución se lanza así. La función devuelve un objeto con el directorio de la ejecución, las métricas y el historial ya cargados:

```python
from neural_labs.experiments import run_lab

resultado = run_lab(
    "15_federated_learning",
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

datos = prepare_dataset("15_federated_learning", quick=True, seed=42)
print(datos.summary)       # tamaño de cada partición y metadatos de la fuente
```

## 🪜 Paso a paso

Cada paso dice qué ocurre por dentro, por qué se hace en ese orden y cómo comprobar que salió bien. El orden no es una convención de estilo: es el que ejecuta el código, y alterarlo invalida el resultado.

### Paso 1 — Traer el dataset real y partirlo

**Qué ocurre.** Descarga `uci_har_subjects` desde su proveedor y construye las tres particiones —`train`, `validation` y `test`— con la semilla de partición que le pases.

**Por qué.** La partición se fija **antes** de entrenar y su semilla (`--split-seed`) es independiente de la del entrenamiento (`--training-seed`). Si ambas fueran la misma, no podrías distinguir si un cambio en el resultado viene de haber repartido los datos de otro modo o de haber inicializado los pesos de otro modo.

```bash
neural-labs dataset --lab 15_federated_learning --quick --split-seed 42
```

**Cómo sabes que salió bien.** El comando termina sin error y deja el dataset en caché. Si no hay red, **falla**: no se rellena con datos inventados (`data/dataset.yaml` declara `fallback_to_generated_data: false`).

### Paso 2 — Comprobar que las particiones no se tocan

**Qué ocurre.** Recorre los identificadores de las tres particiones y verifica que ninguno aparece en dos de ellas.

**Por qué.** Una sola fila compartida entre `train` y `test` basta para que la métrica final quede inflada. Es el error más caro del oficio porque no produce ningún síntoma visible: el modelo simplemente «parece» mejor de lo que es.

```bash
neural-labs audit --lab 15_federated_learning --quick --split-seed 42
```

**Cómo sabes que salió bien.** La auditoría reporta cero solapamientos. Si reporta alguno, no sigas: el resto del laboratorio no significaría nada.

### Paso 3 — Mirar los datos antes de modelarlos

**Qué ocurre.** Genera el informe de calidad (valores faltantes, balance de clases, rangos) y el de deriva entre particiones.

**Por qué.** Un desbalance fuerte o una diferencia de distribución entre `train` y `test` cambia qué métrica tiene sentido leer. Descubrirlo después de entrenar obliga a repetirlo todo.

```bash
neural-labs quality --lab 15_federated_learning --quick --split-seed 42
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
python labs/15_federated_learning/train.py --quick
# o, con control explícito de las dos semillas:
neural-labs train --lab 15_federated_learning --config baseline --split-seed 42 --training-seed 43
```

**Cómo sabes que salió bien.** En `runs/15_federated_learning/<ejecución>/` aparecen `history.csv` y `best_model.pt`; la métrica de validación mejora respecto de la primera época.

### Paso 6 — Compararte con la línea base

**Qué ocurre.** El repositorio entrena por su cuenta **Entrenamiento centralizado** y guarda su resultado, primero sobre `validation` y —solo al final— sobre `test`.

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
neural-labs benchmark --lab 15_federated_learning --quick --split-seed 42 --training-seeds 41 42 43
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
- **Límite declarado de este dataset.** No crea clientes espaciales artificiales; conserva identificadores reales de sujetos.

### Riesgos al interpretar los resultados

No crea clientes espaciales artificiales; conserva identificadores reales de sujetos.

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

- Kairouz et al. (2021), *Advances and Open Problems in Federated Learning*, Foundations and Trends in Machine Learning — monografía de referencia sobre el marco federado, datos no-IID, privacidad y problemas abiertos.
- McMahan et al. (2017), *Communication-Efficient Learning of Deep Networks from Decentralized Data (FedAvg)*, AISTATS — artículo que introduce el algoritmo FedAvg implementado en este laboratorio.
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
| [⚗️ Destilación de conocimiento](../../labs/14_knowledge_distillation/README.md) | [Las 31 rutas](../../parts/README.md) | [∂ Backpropagation manual](../../labs/16_backpropagation_manual/README.md) |

**En este laboratorio:** **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🟠 [Parte 4 — Entrenar mejor, más barato y sin centralizar datos](../../parts/04-entrenamiento-eficiente.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/15_federated_learning/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
