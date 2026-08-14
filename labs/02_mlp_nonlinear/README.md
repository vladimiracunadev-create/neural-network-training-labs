# MLP multiclase

<!-- nav-top -->
> 🧭 **Ruta 3 / 31** · 🟢 [Parte 1 — Fundamentos: de la derivada a la primera red](../../parts/01-fundamentos.md)
>
> [⬅️ 🧩 Perceptrón con PyTorch](../../labs/01_pytorch_perceptron/README.md) · [🏠 Índice de rutas](../../parts/README.md) · [🖼️ CNN para visión ➡️](../../labs/03_cnn_vision/README.md)
>
> **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

## 🎯 Qué vas a hacer aquí

Resolver clasificación no lineal con capas densas, activaciones y regularización.

Es la **ruta 3 de 31** del recorrido y pertenece a 🟢 la parte 1, *Fundamentos: de la derivada a la primera red*. Llegas desde **Perceptrón con PyTorch** y lo que hagas aquí lo da por supuesto **CNN para visión**.

Trabajarás con el dataset **`dry_bean`** (UCI, licencia: CC BY 4.0), y tendrás que superar la línea base **Regresión logística multinomial y Random Forest**, decidiendo con la métrica `macro_f1` medida sobre `validation`. Nivel fundamentos, unas **4 horas** de dedicación.

**Lo que conviene traer resuelto de las rutas anteriores:** Python básico, NumPy, álgebra lineal elemental.

**Al terminar deberías ser capaz de:**

- Resolver clasificación no lineal con capas densas, activaciones y regularización.
- Preparar y auditar el dataset real dry_bean sin fuga de datos.
- Entrenar y evaluar red multicapa para relaciones no lineales.
- Comparar contra la línea base: Regresión logística multinomial y Random Forest.
- Interpretar intervalos de confianza, errores y limitaciones.

## 🧠 La teoría de este laboratorio

Esta sección es la explicación completa del tema. No hace falta abrir otro archivo para entender lo que viene después: aquí está la idea, la matemática que la sostiene y sus límites. (El mismo texto vive en `theory.md`, que es la fuente desde la que se genera esta guía, junto con la bibliografía del final.)

### De qué trata

Este laboratorio estudia **red multicapa para relaciones no lineales** usando `dry_bean`, un dataset público real procedente de UCI.

Un clasificador lineal solo puede trazar hiperplanos: falla cuando las clases se entrelazan de forma no lineal. El perceptrón multicapa (MLP) resuelve esto apilando capas de neuronas separadas por **funciones de activación no lineales**. La clave conceptual es que sin esas no linealidades, componer varias capas lineales sería inútil —el producto de matrices sigue siendo una matriz, es decir, otro modelo lineal—. La activación (aquí ReLU) es lo que permite que cada capa doble y pliegue el espacio de representación, de modo que clases inseparables en el espacio original se vuelvan separables en el espacio aprendido.

El problema —clasificar 13.611 granos en siete variedades a partir de 16 atributos de forma— es genuinamente multiclase y no lineal, ideal para observar cómo una capa oculta supera a la regresión logística multinomial. El laboratorio también introduce la regularización (dropout, weight decay) como respuesta al mayor riesgo de sobreajuste que trae la capacidad adicional.

### La matemática, paso a paso

Una red de una capa oculta calcula su predicción en dos etapas. Primero proyecta la entrada a un espacio oculto y aplica una no linealidad; luego proyecta ese espacio oculto a los logits de las clases:

h = ReLU(x·W₁ + b₁)    con    ReLU(a) = max(0, a)

logits = h·W₂ + b₂

La función **ReLU** (Rectified Linear Unit) es engañosamente simple: deja pasar los valores positivos y anula los negativos. Su derivada es 1 para a > 0 y 0 para a < 0, lo que la hace barata de calcular y, sobre todo, evita el problema del **desvanecimiento del gradiente** que sufren la sigmoide y la tanh, cuyas derivadas se aproximan a 0 en sus extremos y frenan el aprendizaje en redes profundas. El "codo" no lineal en a = 0 es lo que aporta la capacidad expresiva: cada neurona ReLU introduce un pliegue lineal por tramos, y su combinación aproxima superficies de decisión arbitrariamente complejas.

Este poder no es una intuición vaga sino un resultado formal: el **teorema de aproximación universal** (Cybenko 1989 para sigmoides; Hornik 1991 para activaciones generales) demuestra que una red con una sola capa oculta y suficientes neuronas puede aproximar cualquier función continua sobre un conjunto compacto con el error que se desee. El teorema garantiza la *existencia* de los pesos, no que el descenso de gradiente los encuentre fácilmente; en la práctica, apilar más capas suele ser más eficiente en parámetros que ensanchar una sola.

Para clasificación multiclase, los logits se convierten en una distribución de probabilidad con **softmax**, que normaliza exponenciales para que sumen 1:

softmax(z)ₖ = e^(zₖ) / Σⱼ e^(zⱼ)

y se entrena minimizando la **entropía cruzada categórica**, L = −(1/N) Σᵢ ln( p_{i, yᵢ} ), donde p_{i, yᵢ} es la probabilidad que el modelo asigna a la clase verdadera del ejemplo i. En PyTorch, `CrossEntropyLoss` fusiona softmax y log-verosimilitud de forma numéricamente estable, por lo que la última capa entrega logits crudos. Los gradientes fluyen hacia atrás por retropropagación: ∂L/∂W₂ se calcula directamente y, por la regla de la cadena a través de la ReLU, el error se propaga a la capa oculta (∂L/∂W₁), donde la máscara de la ReLU bloquea el gradiente en las neuronas que estaban inactivas.

La formulación debe conectarse con cuatro elementos: representación de entrada, función del modelo, función de pérdida y regla de actualización. El notebook muestra las dimensiones de los tensores y conserva la misma implementación que el script de terminal.

### Las ecuaciones de la retropropagación en esta red

Con dos capas, la retropropagación cabe en cuatro líneas y conviene tenerlas escritas, porque son el esqueleto de todo lo que viene después. Llamando a = x·W₁ + b₁ a la preactivación oculta, h = ReLU(a), z = h·W₂ + b₂ a los logits y p = softmax(z):

δ² = p − Y,   ∂L/∂W₂ = hᵀ·δ² / N,   ∂L/∂b₂ = Σ_filas δ² / N,

δ¹ = (δ²·W₂ᵀ) ⊙ 𝟙[a > 0],   ∂L/∂W₁ = xᵀ·δ¹ / N,   ∂L/∂b₁ = Σ_filas δ¹ / N,

donde ⊙ es el producto elemento a elemento y 𝟙[a > 0] la máscara de la ReLU. Vale la pena leer la segunda línea despacio: el error de la capa de salida viaja hacia atrás multiplicado por W₂ᵀ —la misma matriz del paso hacia adelante, transpuesta— y luego se **apaga** en las posiciones donde la neurona estaba inactiva. Una neurona que no participó en la predicción tampoco recibe corrección.

Que δ² = p − Y no es evidente, y es el mismo regalo que aparecía en la ruta 00. Derivando la entropía cruzada categórica respecto de los logits, el término del softmax ∂pₖ/∂z_j = pₖ·(δ_kj − p_j) se combina con ∂L/∂pₖ = −y_k/pₖ y todo se simplifica a p − y. Softmax con entropía cruzada, igual que sigmoide con entropía cruzada binaria, están emparejadas para que el gradiente sea el error puro.

Sobre el softmax hay una propiedad que se usa en toda implementación seria: es **invariante a desplazamientos**, softmax(z + c) = softmax(z) para cualquier constante c. Restar el máximo, softmax(z − max z), no cambia el resultado y garantiza que el mayor exponente sea e⁰ = 1, evitando el desbordamiento de e^z con logits grandes. Es lo que `CrossEntropyLoss` hace internamente, y la razón de que la última capa deba devolver logits crudos.

### Cuántos parámetros hay y cómo inicializarlos

Contar los parámetros es inmediato y conviene hacerlo antes de entrenar. Cada capa densa de m entradas y n salidas aporta m·n pesos más n sesgos, así que para una pila de anchuras (H₁, H₂, …) entre d características y C clases:

|θ| = (d·H₁ + H₁) + (H₁·H₂ + H₂) + … + (H_L·C + C).

El modelo tabular de este repositorio usa por defecto dos capas ocultas de 128 y 64 unidades. Con las 16 características de forma del dataset y sus 7 variedades, la cuenta es 16·128 + 128 + 128·64 + 64 + 64·7 + 7 = **10 887 parámetros**. Frente a los 13 611 granos del conjunto completo, la red tiene casi un parámetro por ejemplo: es exactamente la situación en la que memorizar es una estrategia disponible, y la que justifica el dropout y el weight decay que se estudian en la ruta 19.

La **inicialización** no es un detalle. Si todos los pesos se ponen a cero, todas las neuronas ocultas calculan lo mismo, reciben el mismo gradiente y siguen siendo idénticas para siempre: la red se comporta como si tuviera una sola neurona oculta. Es el problema de **simetría**, y por eso los pesos se inicializan al azar. Pero la escala de ese azar importa: si la varianza es alta, las preactivaciones crecen capa a capa y saturan; si es baja, se encogen y la señal se apaga.

La receta que usan las redes con ReLU es la de **He**: muestrear W de una normal de varianza 2/fan_in, donde fan_in es el número de entradas de la capa. El factor 2 compensa exactamente que la ReLU anula la mitad de las activaciones y por tanto reduce la varianza a la mitad. Para activaciones simétricas como tanh, la inicialización de **Glorot** usa 2/(fan_in + fan_out), que equilibra la propagación en ambos sentidos. Es la primera aparición de una idea que la ruta 18 desarrolla: gran parte del arte del entrenamiento consiste en mantener la varianza de las activaciones y de los gradientes dentro de un rango sano de extremo a extremo de la red.

De ahí sale también el fallo característico de la ReLU: si una neurona recibe una actualización que deja su preactivación negativa para **todos** los ejemplos, su gradiente es cero permanentemente y no vuelve a aprender nunca. Es la **ReLU muerta**, y su causa habitual es una tasa de aprendizaje demasiado alta. Las variantes Leaky ReLU y GELU, que se comparan en la ruta 17, existen precisamente para dejar pasar algo de gradiente en la zona negativa.

### Por qué la profundidad gana a la anchura

El teorema de aproximación universal garantiza que basta una capa oculta, pero no dice cuántas neuronas hacen falta, y ahí está la trampa: para muchas funciones ese número crece exponencialmente. Contar **regiones lineales** lo hace concreto. Una red ReLU es una función lineal a trozos, y el número de regiones en que divide el espacio de entrada mide su capacidad de plegar la frontera de decisión. Una red de una capa con H neuronas sobre d entradas genera del orden de O(H^d) regiones; una red profunda de L capas con H neuronas cada una alcanza del orden de Ω((H/d)^(d(L−1))·H^d), es decir, **exponencial en la profundidad** y solo polinómico en la anchura.

La lectura práctica es que añadir una capa multiplica la capacidad expresiva mucho más barato que ensanchar la existente, y es la justificación de que este laboratorio compare profundidad frente a anchura en vez de dar por buena la primera arquitectura que funcione. La contrapartida —redes profundas más difíciles de optimizar por gradientes que se desvanecen— es lo que motivará las conexiones residuales y la normalización en rutas posteriores.

> **La pregunta que deberías poder responder al terminar:** ¿La complejidad adicional supera de forma estable a la línea base?

### Qué se mide y con qué se decide

El laboratorio reporta `accuracy`, `balanced_accuracy`, `macro_precision`, `macro_recall`, `macro_f1`. De todas ellas, la que **decide** qué modelo se conserva es `macro_f1`, y se mide siempre sobre `validation`: es la única forma de que `test` siga siendo una estimación honesta de lo que pasará con datos nuevos.

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
jupyter lab labs/02_mlp_nonlinear/notebook.ipynb
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
| `--lab` | `02_mlp_nonlinear` | obligatorio | Qué laboratorio se ejecuta. Solo acepta los identificadores del catálogo. |
| `--quick` | desactivado | — | Usa una fracción real del dataset y pocas épocas. Sirve para comprobar la instalación, no para concluir nada sobre el modelo. |
| `--split-seed N` | `42` | entero | Semilla que decide **qué ejemplo cae en qué partición**. Se mantiene fija al comparar modelos. |
| `--training-seed N` | `42` | entero | Semilla de la inicialización de pesos y del barajado de lotes. Es la que se varía para medir cuánta diferencia es simple azar. |
| `--config` | `baseline` | `baseline` · `improved` | Cuál de las dos configuraciones del laboratorio se usa. |
| `--device` | `auto` | `auto` · `cpu` · `cuda` · `mps` | Dónde entrenar. `auto` elige GPU si está disponible y cae a CPU si no. |
| `--training-seeds A B C` | `41 42 43` | enteros | Solo en `benchmark`: la lista de semillas de entrenamiento que se repiten. |
| `--output-dir` | `runs` | ruta | Dónde se escribe el directorio de la ejecución. |

### El script del laboratorio

`labs/02_mlp_nonlinear/train.py` no es un programa distinto: fija el `--lab` y delega en la misma herramienta, de modo que estas dos líneas hacen exactamente lo mismo.

```bash
python labs/02_mlp_nonlinear/train.py --quick
neural-labs train --lab 02_mlp_nonlinear --quick
```

### Lo mismo desde Python

Si prefieres trabajar en un cuaderno o llamar al laboratorio desde tu propio código, la misma ejecución se lanza así. La función devuelve un objeto con el directorio de la ejecución, las métricas y el historial ya cargados:

```python
from neural_labs.experiments import run_lab

resultado = run_lab(
    "02_mlp_nonlinear",
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

datos = prepare_dataset("02_mlp_nonlinear", quick=True, seed=42)
print(datos.summary)       # tamaño de cada partición y metadatos de la fuente
```

## 🪜 Paso a paso

Cada paso dice qué ocurre por dentro, por qué se hace en ese orden y cómo comprobar que salió bien. El orden no es una convención de estilo: es el que ejecuta el código, y alterarlo invalida el resultado.

### Paso 1 — Traer el dataset real y partirlo

**Qué ocurre.** Descarga `dry_bean` desde su proveedor y construye las tres particiones —`train`, `validation` y `test`— con la semilla de partición que le pases.

**Por qué.** La partición se fija **antes** de entrenar y su semilla (`--split-seed`) es independiente de la del entrenamiento (`--training-seed`). Si ambas fueran la misma, no podrías distinguir si un cambio en el resultado viene de haber repartido los datos de otro modo o de haber inicializado los pesos de otro modo.

```bash
neural-labs dataset --lab 02_mlp_nonlinear --quick --split-seed 42
```

**Cómo sabes que salió bien.** El comando termina sin error y deja el dataset en caché. Si no hay red, **falla**: no se rellena con datos inventados (`data/dataset.yaml` declara `fallback_to_generated_data: false`).

### Paso 2 — Comprobar que las particiones no se tocan

**Qué ocurre.** Recorre los identificadores de las tres particiones y verifica que ninguno aparece en dos de ellas.

**Por qué.** Una sola fila compartida entre `train` y `test` basta para que la métrica final quede inflada. Es el error más caro del oficio porque no produce ningún síntoma visible: el modelo simplemente «parece» mejor de lo que es.

```bash
neural-labs audit --lab 02_mlp_nonlinear --quick --split-seed 42
```

**Cómo sabes que salió bien.** La auditoría reporta cero solapamientos. Si reporta alguno, no sigas: el resto del laboratorio no significaría nada.

### Paso 3 — Mirar los datos antes de modelarlos

**Qué ocurre.** Genera el informe de calidad (valores faltantes, balance de clases, rangos) y el de deriva entre particiones.

**Por qué.** Un desbalance fuerte o una diferencia de distribución entre `train` y `test` cambia qué métrica tiene sentido leer. Descubrirlo después de entrenar obliga a repetirlo todo.

```bash
neural-labs quality --lab 02_mlp_nonlinear --quick --split-seed 42
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
python labs/02_mlp_nonlinear/train.py --quick
# o, con control explícito de las dos semillas:
neural-labs train --lab 02_mlp_nonlinear --config baseline --split-seed 42 --training-seed 43
```

**Cómo sabes que salió bien.** En `runs/02_mlp_nonlinear/<ejecución>/` aparecen `history.csv` y `best_model.pt`; la métrica de validación mejora respecto de la primera época.

### Paso 6 — Compararte con la línea base

**Qué ocurre.** El repositorio entrena por su cuenta **Regresión logística multinomial y Random Forest** y guarda su resultado, primero sobre `validation` y —solo al final— sobre `test`.

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
neural-labs benchmark --lab 02_mlp_nonlinear --quick --split-seed 42 --training-seeds 41 42 43
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
- **Límite declarado de este dataset.** 13.611 granos de siete variedades reales y 16 atributos de forma.

### Riesgos al interpretar los resultados

13.611 granos de siete variedades reales y 16 atributos de forma.

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

- Goodfellow, Bengio & Courville — *Deep Learning* (MIT Press 2016), cap. 6 — redes hacia adelante, ReLU, softmax y retropropagación.
- Géron — *Hands-On Machine Learning* (3.ª ed., O'Reilly 2022), cap. 10 — diseño e implementación de MLP y regularización.
- Prince — *Understanding Deep Learning* (MIT Press 2024), cap. 3–4 — redes superficiales y profundas con activaciones lineales por tramos.
- He et al. (2015), *Delving Deep into Rectifiers*, ICCV — la inicialización de varianza 2/fan_in para redes con ReLU.
- Glorot & Bengio (2010), *Understanding the difficulty of training deep feedforward neural networks*, AISTATS — la inicialización que equilibra la varianza en ambos sentidos.
- Montúfar et al. (2014), *On the Number of Linear Regions of Deep Neural Networks*, NeurIPS — el conteo de regiones lineales que cuantifica la ventaja de la profundidad.
- Cybenko (1989), *Approximation by superpositions of a sigmoidal function*, Math. Control Signals Systems — teorema de aproximación universal para sigmoides.
- Hornik (1991), *Approximation capabilities of multilayer feedforward networks*, Neural Networks — generalización del teorema a activaciones arbitrarias.
- Fuente del dataset: https://archive.ics.uci.edu/dataset/602/dry+bean+dataset
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
| [🧩 Perceptrón con PyTorch](../../labs/01_pytorch_perceptron/README.md) | [Las 31 rutas](../../parts/README.md) | [🖼️ CNN para visión](../../labs/03_cnn_vision/README.md) |

**En este laboratorio:** **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🟢 [Parte 1 — Fundamentos: de la derivada a la primera red](../../parts/01-fundamentos.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/02_mlp_nonlinear/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
