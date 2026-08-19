# Transformer para noticias

<!-- nav-top -->
> 🧭 **Ruta 8 / 31** · 🔵 [Parte 2 — Arquitecturas según la forma del dato](../../parts/02-arquitecturas.md)
>
> [⬅️ 🧬 Autoencoder para fraude](../../labs/06_autoencoder_anomaly/README.md) · [🏠 Índice de rutas](../../parts/README.md) · [🎨 GAN generativa ➡️](../../labs/08_gan_generation/README.md)
>
> **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

## 🎯 Qué vas a hacer aquí

Aplicar atención multi-cabeza a clasificación de noticias reales.

Es la **ruta 8 de 31** del recorrido y pertenece a 🔵 la parte 2, *Arquitecturas según la forma del dato*. Llegas desde **Autoencoder para fraude** y lo que hagas aquí lo da por supuesto **GAN generativa**.

Trabajarás con el dataset **`ag_news`** (Hugging Face, licencia: Consultar dataset card), y tendrás que superar la línea base **TF-IDF + regresión logística**, decidiendo con la métrica `macro_f1` medida sobre `validation`. Nivel avanzado, unas **8 horas** de dedicación.

**Lo que conviene traer resuelto de las rutas anteriores:** PyTorch intermedio, optimización, lectura de artículos técnicos.

**Al terminar deberías ser capaz de:**

- Aplicar atención multi-cabeza a clasificación de noticias reales.
- Preparar y auditar el dataset real ag_news sin fuga de datos.
- Entrenar y evaluar autoatención para clasificación de texto.
- Comparar contra la línea base: TF-IDF + regresión logística.
- Interpretar intervalos de confianza, errores y limitaciones.

## 🧠 La teoría de este laboratorio

Esta sección es la explicación completa del tema. No hace falta abrir otro archivo para entender lo que viene después: aquí está la idea, la matemática que la sostiene y sus límites. (El mismo texto vive en `theory.md`, que es la fuente desde la que se genera esta guía, junto con la bibliografía del final.)

### De qué trata

Este laboratorio estudia **autoatención para clasificación de texto** usando `ag_news`, un dataset público real procedente de Hugging Face.

La intuición central es que el significado de una palabra depende de su contexto, y ese contexto puede estar lejos en la secuencia. Los modelos recurrentes procesan el texto token a token y arrastran la información en un estado que se degrada con la distancia. La **autoatención** rompe con eso: permite que cada token mire a *todos* los demás en un solo paso y pondere cuánto atender a cada uno según su relevancia. Para clasificar una noticia como "Deportes", "Mundo", "Negocios" o "Ciencia/Tecnología", el modelo aprende a concentrar la atención en las palabras discriminantes (nombres de equipos, términos financieros, etc.) sin importar en qué posición aparezcan.

Cada token se proyecta en tres roles: una **consulta** (query) que expresa "qué busco", una **clave** (key) que expresa "qué ofrezco" y un **valor** (value) que es la información a transmitir. La compatibilidad entre la consulta de un token y las claves de los demás determina los pesos con que se combinan los valores. Al apilar varias "cabezas" de atención en paralelo, el modelo puede atender simultáneamente a distintos tipos de relación (sintáctica, semántica, de correferencia). Este laboratorio construye el transformer desde cero para ver cómo estas piezas producen una representación contextual de toda la noticia.

### La matemática, paso a paso

El bloque de atención escalada por producto punto se define sobre matrices Q ∈ ℝ^{n×d_k}, K ∈ ℝ^{n×d_k} y V ∈ ℝ^{n×d_v}, donde n es el número de tokens:

    Attention(Q, K, V) = softmax( Q Kᵀ / √d_k ) V

La matriz Q Kᵀ contiene, en su entrada (i, j), el producto punto entre la consulta del token i y la clave del token j: un puntaje de compatibilidad. La división por √d_k es esencial —no cosmética—: cuando d_k es grande, los productos punto crecen en magnitud proporcionalmente a √d_k, y sin escalar empujarían al softmax a regiones de gradiente casi nulo (saturación), frenando el aprendizaje. Dividir por √d_k mantiene la varianza de los puntajes controlada. El **softmax** por filas convierte cada fila en una distribución de pesos que suma 1, y al multiplicar por V se obtiene, para cada token, un promedio ponderado de los valores de toda la secuencia: su nueva representación contextual.

La **atención multi-cabeza** ejecuta h proyecciones distintas en paralelo. Con matrices aprendidas W_iᵠ, W_iᴷ, W_iⱽ para cada cabeza i, se calcula headᵢ = Attention(Q W_iᵠ, K W_iᴷ, V W_iⱽ), y luego se concatenan y se proyectan: MultiHead(Q,K,V) = Concat(head₁, …, head_h) Wᴼ. Cada cabeza opera en un subespacio de dimensión d_k = d_model / h, de modo que el coste total es comparable al de una sola cabeza pero con mayor capacidad de representar relaciones diversas. Como la atención es invariante al orden (permutar los tokens permuta las filas pero no cambia las relaciones), se suma una **codificación posicional** a los embeddings de entrada para inyectar la noción de posición; sin ella el modelo no distinguiría "el perro muerde al hombre" de "el hombre muerde al perro".

Conectando con los cuatro elementos del laboratorio: la **representación de entrada** es la secuencia de embeddings de tokens más su codificación posicional; la **función del modelo** apila bloques de atención multi-cabeza + red feed-forward con conexiones residuales y normalización de capa, y agrega la secuencia en un vector para clasificar; la **función de pérdida** es la entropía cruzada categórica sobre las cuatro clases, ℒ = −Σ_c y_c log ŷ_c, donde ŷ = softmax de los logits; y la **regla de actualización** es descenso de gradiente (Adam) con θ ← θ − η ∇_θ ℒ. El notebook muestra las dimensiones de los tensores (n, d_model, h, d_k) en cada etapa y conserva la misma implementación que el script de terminal.

### Por qué se divide por √d_k, con el cálculo delante

El factor 1/√d_k es la parte de la fórmula que más se copia sin entender, y tiene una justificación estadística exacta. Supóngase que las componentes de q y k son variables aleatorias independientes de media 0 y varianza 1. Su producto escalar es q·k = Σ_(i=1..d_k) qᵢ·kᵢ, una suma de d_k términos independientes de media 0 y varianza 1, de modo que

𝔼[q·k] = 0,   Var(q·k) = d_k,   desviación típica = √d_k.

Es decir, **la magnitud típica del producto escalar crece con √d_k**. Con d_k = 64, los logits de atención tendrían una desviación típica de 8: valores que el softmax convierte casi en un one-hot, concentrando toda la atención en un único token. Y un softmax saturado tiene gradiente prácticamente nulo —la misma patología de la sigmoide en la ruta 00—, así que la atención dejaría de aprender a quién mirar. Dividir por √d_k devuelve la varianza a 1 y mantiene el softmax en su zona sensible, independientemente de la dimensión elegida.

De ahí también se entiende para qué sirven varias **cabezas**. Con d_model = 256 se podría hacer una sola atención de d_k = 256, pero se prefieren, por ejemplo, 8 cabezas de d_k = 32 cada una. El costo en parámetros es idéntico —las proyecciones suman lo mismo—, y a cambio el modelo obtiene ocho relaciones distintas en subespacios distintos, que luego concatena y mezcla con W_O. Una cabeza sola tiene que comprometer una única distribución de atención para todos los tipos de relación; ocho cabezas pueden especializarse, y en la práctica se observa que unas siguen la posición contigua, otras enlazan sujeto y verbo, otras marcan tokens raros.

### Lo que el transformer gana y lo que paga frente a la recurrencia

La comparación con la ruta 04 se puede hacer con dos números, y explica el cambio de paradigma completo.

**Camino de información.** En una RNN, la señal entre las posiciones i y j debe atravesar |i − j| pasos recurrentes, multiplicándose por otras tantas matrices —de ahí el desvanecimiento—. En la atención, cualquier par de posiciones está conectado por **un solo** producto escalar: el camino máximo es O(1). Esa es la razón de fondo por la que los transformers capturan dependencias largas que a una RNN se le escapan, y no una cuestión de tamaño.

**Costo.** La contrapartida es que construir la matriz de atención exige comparar todos los pares: complejidad **O(n²·d)** en tiempo y en memoria, frente a la O(n·d²) de la recurrencia. Para los titulares cortos de este laboratorio n es pequeño y el cuadrado no duele; para documentos de miles de tokens, esa cuadrática es la razón de ser de toda una línea de investigación en atención eficiente.

**Paralelismo.** Una RNN debe calcular hₜ antes que hₜ₊₁: su cómputo es secuencial por construcción y no se puede repartir en el tiempo. La atención procesa todas las posiciones a la vez, y por eso aprovecha el hardware paralelo de forma que la recurrencia nunca podrá. Esta ventaja práctica —no una superioridad teórica— es la que decidió la adopción masiva de la arquitectura.

Hay un precio conceptual: al mirar todas las posiciones simultáneamente, **la atención no sabe nada del orden**. Es una operación permutación-equivariante, así que sin información posicional «el perro mordió al hombre» y «el hombre mordió al perro» tendrían representaciones idénticas para el conjunto de tokens. La codificación posicional que se suma a los embeddings existe únicamente para romper esa simetría, y por eso no es un adorno: sin ella el modelo pierde toda la sintaxis.

### Qué es y qué no es un mapa de atención

Los mapas de atención se visualizan en este laboratorio y conviene interpretarlos con precisión. Cada fila de softmax(Q·Kᵀ/√d_k) es una distribución de probabilidad: suma 1 y dice qué mezcla de valores V construye la representación de esa posición. Eso es todo lo que dice.

En particular, **no es una explicación de la decisión**. Un peso alto significa que ese token contribuyó a la mezcla en esa capa y esa cabeza, no que la predicción dependa causalmente de él: la información puede haber viajado por la conexión residual, haber sido reescrita por la FFN, o repartirse entre varias cabezas que se compensan. Se han construido modelos con mapas de atención muy distintos y predicciones idénticas, que es la prueba de que la atención no identifica de forma única la causa. Para afirmar dependencia causal hacen falta las técnicas de la ruta 21 —perturbar la entrada y medir el cambio en la salida—, no leer los pesos.

Históricamente la atención nació como mecanismo de *alineamiento* en traducción (Bahdanau et al., 2015), donde el decodificador aprendía a qué palabras de la frase origen mirar en cada paso. La contribución de Vaswani et al. (2017) fue mostrar que la atención por sí sola —sin recurrencia ni convolución— basta para modelar secuencias, lo que además desbloquea el paralelismo masivo que hizo posibles los modelos de lenguaje actuales.

> **La pregunta que deberías poder responder al terminar:** ¿La atención observada coincide con evidencia útil para la clase?

### Qué se mide y con qué se decide

El laboratorio reporta `accuracy`, `balanced_accuracy`, `macro_precision`, `macro_recall`, `macro_f1`. De todas ellas, la que **decide** qué modelo se conserva es `macro_f1`, y se mide siempre sobre `validation`: es la única forma de que `test` siga siendo una estimación honesta de lo que pasará con datos nuevos.

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
jupyter lab labs/07_transformer_attention/notebook.ipynb
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
| `--lab` | `07_transformer_attention` | obligatorio | Qué laboratorio se ejecuta. Solo acepta los identificadores del catálogo. |
| `--quick` | desactivado | — | Usa una fracción real del dataset y pocas épocas. Sirve para comprobar la instalación, no para concluir nada sobre el modelo. |
| `--split-seed N` | `42` | entero | Semilla que decide **qué ejemplo cae en qué partición**. Se mantiene fija al comparar modelos. |
| `--training-seed N` | `42` | entero | Semilla de la inicialización de pesos y del barajado de lotes. Es la que se varía para medir cuánta diferencia es simple azar. |
| `--config` | `baseline` | `baseline` · `improved` | Cuál de las dos configuraciones del laboratorio se usa. |
| `--device` | `auto` | `auto` · `cpu` · `cuda` · `mps` | Dónde entrenar. `auto` elige GPU si está disponible y cae a CPU si no. |
| `--training-seeds A B C` | `41 42 43` | enteros | Solo en `benchmark`: la lista de semillas de entrenamiento que se repiten. |
| `--output-dir` | `runs` | ruta | Dónde se escribe el directorio de la ejecución. |

### El script del laboratorio

`labs/07_transformer_attention/train.py` no es un programa distinto: fija el `--lab` y delega en la misma herramienta, de modo que estas dos líneas hacen exactamente lo mismo.

```bash
python labs/07_transformer_attention/train.py --quick
neural-labs train --lab 07_transformer_attention --quick
```

### Lo mismo desde Python

Si prefieres trabajar en un cuaderno o llamar al laboratorio desde tu propio código, la misma ejecución se lanza así. La función devuelve un objeto con el directorio de la ejecución, las métricas y el historial ya cargados:

```python
from neural_labs.experiments import run_lab

resultado = run_lab(
    "07_transformer_attention",
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

datos = prepare_dataset("07_transformer_attention", quick=True, seed=42)
print(datos.summary)       # tamaño de cada partición y metadatos de la fuente
```

## 🪜 Paso a paso

Cada paso dice qué ocurre por dentro, por qué se hace en ese orden y cómo comprobar que salió bien. El orden no es una convención de estilo: es el que ejecuta el código, y alterarlo invalida el resultado.

### Paso 1 — Traer el dataset real y partirlo

**Qué ocurre.** Descarga `ag_news` desde su proveedor y construye las tres particiones —`train`, `validation` y `test`— con la semilla de partición que le pases.

**Por qué.** La partición se fija **antes** de entrenar y su semilla (`--split-seed`) es independiente de la del entrenamiento (`--training-seed`). Si ambas fueran la misma, no podrías distinguir si un cambio en el resultado viene de haber repartido los datos de otro modo o de haber inicializado los pesos de otro modo.

```bash
neural-labs dataset --lab 07_transformer_attention --quick --split-seed 42
```

**Cómo sabes que salió bien.** El comando termina sin error y deja el dataset en caché. Si no hay red, **falla**: no se rellena con datos inventados (`data/dataset.yaml` declara `fallback_to_generated_data: false`).

### Paso 2 — Comprobar que las particiones no se tocan

**Qué ocurre.** Recorre los identificadores de las tres particiones y verifica que ninguno aparece en dos de ellas.

**Por qué.** Una sola fila compartida entre `train` y `test` basta para que la métrica final quede inflada. Es el error más caro del oficio porque no produce ningún síntoma visible: el modelo simplemente «parece» mejor de lo que es.

```bash
neural-labs audit --lab 07_transformer_attention --quick --split-seed 42
```

**Cómo sabes que salió bien.** La auditoría reporta cero solapamientos. Si reporta alguno, no sigas: el resto del laboratorio no significaría nada.

### Paso 3 — Mirar los datos antes de modelarlos

**Qué ocurre.** Genera el informe de calidad (valores faltantes, balance de clases, rangos) y el de deriva entre particiones.

**Por qué.** Un desbalance fuerte o una diferencia de distribución entre `train` y `test` cambia qué métrica tiene sentido leer. Descubrirlo después de entrenar obliga a repetirlo todo.

```bash
neural-labs quality --lab 07_transformer_attention --quick --split-seed 42
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
python labs/07_transformer_attention/train.py --quick
# o, con control explícito de las dos semillas:
neural-labs train --lab 07_transformer_attention --config baseline --split-seed 42 --training-seed 43
```

**Cómo sabes que salió bien.** En `runs/07_transformer_attention/<ejecución>/` aparecen `history.csv` y `best_model.pt`; la métrica de validación mejora respecto de la primera época.

### Paso 6 — Compararte con la línea base

**Qué ocurre.** El repositorio entrena por su cuenta **TF-IDF + regresión logística** y guarda su resultado, primero sobre `validation` y —solo al final— sobre `test`.

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
neural-labs benchmark --lab 07_transformer_attention --quick --split-seed 42 --training-seeds 41 42 43
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
- **Límite declarado de este dataset.** Noticias reales en cuatro categorías con particiones públicas.

### Riesgos al interpretar los resultados

Noticias reales en cuatro categorías con particiones públicas.

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

- Goodfellow, Bengio & Courville — *Deep Learning* (MIT Press, 2016), cap. 10 — modelado de secuencias y mecanismos de atención sobre redes recurrentes.
- Géron — *Hands-On Machine Learning* (3.ª ed., O'Reilly), cap. 16 — procesamiento de secuencias con RNN y atención, transformer paso a paso.
- Prince — *Understanding Deep Learning* (MIT Press, 2024), cap. 12 — desarrollo moderno y didáctico de la autoatención y la arquitectura transformer.
- Bahdanau, Cho & Bengio (2015), *Neural Machine Translation by Jointly Learning to Align and Translate*, ICLR — introdujo la atención como alineamiento suave en secuencia-a-secuencia.
- Vaswani et al. (2017), *Attention Is All You Need*, NeurIPS — la arquitectura transformer basada íntegramente en autoatención multi-cabeza.
- Fuente del dataset: https://huggingface.co/datasets/fancyzhx/ag_news — **AG News Topic Classification Dataset** (Distribuido por Hugging Face Datasets, La ficha de Hugging Face declara `unknown`); procedencia, versión y SHA-256 en el registro de fuentes, entrada `ag-news` — esta clase la usa para aplicar atención multi-cabeza a la clasificación de titulares reales en cuatro categorías.
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
| [🧬 Autoencoder para fraude](../../labs/06_autoencoder_anomaly/README.md) | [Las 31 rutas](../../parts/README.md) | [🎨 GAN generativa](../../labs/08_gan_generation/README.md) |

**En este laboratorio:** **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🔵 [Parte 2 — Arquitecturas según la forma del dato](../../parts/02-arquitecturas.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/07_transformer_attention/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
