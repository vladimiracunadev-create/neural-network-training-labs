# GAN generativa

<!-- nav-top -->
> 🧭 **Ruta 9 / 31** · 🟣 [Parte 3 — Familias especializadas: generar, decidir, relacionar](../../parts/03-familias-especializadas.md)
>
> [⬅️ 🔭 Transformer para noticias](../../labs/07_transformer_attention/README.md) · [🏠 Índice de rutas](../../parts/README.md) · [🕸️ GNN sobre red de citas ➡️](../../labs/09_gnn_graphs/README.md)
>
> **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

## 🎯 Qué vas a hacer aquí

Generar prendas a partir de imágenes reales de Fashion-MNIST.

Es la **ruta 9 de 31** del recorrido y pertenece a 🟣 la parte 3, *Familias especializadas: generar, decidir, relacionar*. Llegas desde **Transformer para noticias** y lo que hagas aquí lo da por supuesto **GNN sobre red de citas**.

Trabajarás con el dataset **`fashion_mnist`** (Torchvision / Zalando Research, licencia: MIT), y tendrás que superar la línea base **PCA generativa y distribución real de referencia**, decidiendo con la métrica `f1` medida sobre `validation`. Nivel avanzado, unas **8 horas** de dedicación.

**Lo que conviene traer resuelto de las rutas anteriores:** PyTorch intermedio, optimización, lectura de artículos técnicos.

**Al terminar deberías ser capaz de:**

- Generar prendas a partir de imágenes reales de Fashion-MNIST.
- Preparar y auditar el dataset real fashion_mnist sin fuga de datos.
- Entrenar y evaluar aprendizaje adversarial generativo.
- Comparar contra la línea base: PCA generativa y distribución real de referencia.
- Interpretar intervalos de confianza, errores y limitaciones.

## 🧠 La teoría de este laboratorio

Esta sección es la explicación completa del tema. No hace falta abrir otro archivo para entender lo que viene después: aquí está la idea, la matemática que la sostiene y sus límites. (El mismo texto vive en `theory.md`, que es la fuente desde la que se genera esta guía, junto con la bibliografía del final.)

### De qué trata

Este laboratorio estudia **aprendizaje adversarial generativo** usando `fashion_mnist`, un dataset público real procedente de Torchvision / Zalando Research.

Una **red generativa adversarial (GAN)** plantea el aprendizaje como un juego entre dos redes con objetivos opuestos. El **generador** G toma ruido aleatorio z y trata de producir imágenes que parezcan prendas reales. El **discriminador** D es un clasificador que recibe una imagen y estima la probabilidad de que sea real (proveniente del dataset) y no falsa (generada por G). Ambos se entrenan a la vez: D mejora en distinguir real de falso, y G mejora en engañar a D. La metáfora habitual es la del falsificador (G) y el detective (D): cada uno fuerza al otro a mejorar, y en el equilibrio ideal el falsificador produce prendas indistinguibles de las auténticas.

Lo elegante es que G nunca ve las imágenes reales directamente ni recibe una pérdida de reconstrucción píxel a píxel; aprende **solo a través del gradiente que le pasa D**. En vez de decirle a G "copia esta imagen", D le dice "esto todavía se nota falso por aquí", y ese señal guía a G hacia la variedad de imágenes plausibles. Este laboratorio usa una **DCGAN** (GAN convolucional profunda), donde G usa convoluciones transpuestas para expandir el ruido hasta una imagen de 28×28 y D usa convoluciones para clasificarla; esta receta convolucional es la que estabilizó el entrenamiento de GANs sobre imágenes.

### La matemática, paso a paso

El objetivo original es un juego minimax de suma cero sobre el valor V(D, G):

    min_G max_D  V(D, G) = 𝔼_{x∼p_data}[ log D(x) ] + 𝔼_{z∼p_z}[ log(1 − D(G(z))) ]

Leámoslo por partes. El discriminador D quiere **maximizar** V: para muestras reales x quiere D(x) → 1 (así log D(x) → 0, su máximo), y para muestras falsas G(z) quiere D(G(z)) → 0 (así log(1 − D(G(z))) → 0). El generador G quiere **minimizar** V respecto al segundo término: busca que D(G(z)) → 1, es decir, engañar a D. z se muestrea de una distribución simple p_z (típicamente 𝒩(0, I)) y G la transforma en la distribución generada p_g. El entrenamiento alterna pasos: se congela G y se da un paso de ascenso de gradiente en θ_D, luego se congela D y se da un paso de descenso en θ_G.

¿Por qué este juego produce imágenes realistas? Goodfellow et al. probaron que, para un G fijo, el discriminador óptimo es D*(x) = p_data(x) / (p_data(x) + p_g(x)). Sustituyendo D* en V, el objetivo de G se vuelve equivalente a minimizar la **divergencia de Jensen–Shannon** entre la distribución real p_data y la generada p_g (salvo constantes): min_G V = 2·D_JS(p_data ‖ p_g) − log 4. El mínimo global se alcanza cuando p_g = p_data, es decir, cuando el generador reproduce exactamente la distribución de las prendas reales y D no puede hacer mejor que responder ½ en todo. Ese es el sentido preciso de "generar imágenes indistinguibles".

En la práctica, el término log(1 − D(G(z))) tiene gradiente casi nulo justo cuando G es malo (al inicio, D lo detecta con facilidad), así que se suele entrenar G maximizando 𝔼_z[ log D(G(z)) ] —el truco del "gradiente no saturante"— que apunta al mismo óptimo pero da señal fuerte desde el principio. Conectando con los cuatro elementos: la **representación de entrada** es el vector de ruido z para G y la imagen (28×28) para D; la **función del modelo** son las dos redes convolucionales G y D; la **función de pérdida** es la entropía cruzada binaria derivada de V (una para D, otra para G); y la **regla de actualización** son los dos pasos de gradiente alternados θ_D ← θ_D + η ∇_{θ_D} V y θ_G ← θ_G − η ∇_{θ_G} V. El notebook muestra las dimensiones de los tensores en cada capa y conserva la misma implementación que el script de terminal.

El riesgo técnico característico es el **colapso de modos** (mode collapse): G descubre unas pocas imágenes que engañan a D y las produce siempre, perdiendo diversidad aunque la pérdida parezca buena. Por eso este laboratorio no se conforma con las curvas de pérdida y mide diversidad, distancia al vecino real más cercano y discrepancia de momentos: distinguir *diversidad real* de *ruido visual* o de un puñado de prototipos repetidos es exactamente el reto de evaluar una GAN.

> **La pregunta que deberías poder responder al terminar:** ¿Cómo se distingue diversidad real de ruido visual?

### Qué se mide y con qué se decide

El laboratorio reporta `generator_loss`, `discriminator_loss`, `mmd_rbf`, `diversity`, `nearest_real_distance`, `moment_distance`. De todas ellas, la que **decide** qué modelo se conserva es `f1`, y se mide siempre sobre `validation`: es la única forma de que `test` siga siendo una estimación honesta de lo que pasará con datos nuevos.

## 🖥️ Los comandos, explicados

Todo el laboratorio se maneja con una sola herramienta de terminal, `neural-labs`, que se instala junto con el paquete (`pip install -e ".[dev,notebooks]"`). Cada subcomando hace **una** cosa del protocolo, y por eso se pueden ejecutar por separado: preparar datos, auditar la partición, entrenar, repetir con varias semillas.

La forma general es siempre la misma:

```bash
neural-labs <subcomando> --lab <identificador> [opciones]
```

| Opción | Valor por defecto | Valores | Qué hace y cuándo cambiarla |
|---|---|---|---|
| `--lab` | `08_gan_generation` | obligatorio | Qué laboratorio se ejecuta. Solo acepta los identificadores del catálogo. |
| `--quick` | desactivado | — | Usa una fracción real del dataset y pocas épocas. Sirve para comprobar la instalación, no para concluir nada sobre el modelo. |
| `--split-seed N` | `42` | entero | Semilla que decide **qué ejemplo cae en qué partición**. Se mantiene fija al comparar modelos. |
| `--training-seed N` | `42` | entero | Semilla de la inicialización de pesos y del barajado de lotes. Es la que se varía para medir cuánta diferencia es simple azar. |
| `--config` | `baseline` | `baseline` · `improved` | Cuál de las dos configuraciones del laboratorio se usa. |
| `--device` | `auto` | `auto` · `cpu` · `cuda` · `mps` | Dónde entrenar. `auto` elige GPU si está disponible y cae a CPU si no. |
| `--training-seeds A B C` | `41 42 43` | enteros | Solo en `benchmark`: la lista de semillas de entrenamiento que se repiten. |
| `--output-dir` | `runs` | ruta | Dónde se escribe el directorio de la ejecución. |

### El script del laboratorio

`labs/08_gan_generation/train.py` no es un programa distinto: fija el `--lab` y delega en la misma herramienta, de modo que estas dos líneas hacen exactamente lo mismo.

```bash
python labs/08_gan_generation/train.py --quick
neural-labs train --lab 08_gan_generation --quick
```

### Lo mismo desde Python

Si prefieres trabajar en un cuaderno o llamar al laboratorio desde tu propio código, la misma ejecución se lanza así. La función devuelve un objeto con el directorio de la ejecución, las métricas y el historial ya cargados:

```python
from neural_labs.experiments import run_lab

resultado = run_lab(
    "08_gan_generation",
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

datos = prepare_dataset("08_gan_generation", quick=True, seed=42)
print(datos.summary)       # tamaño de cada partición y metadatos de la fuente
```

## 🪜 Paso a paso

Cada paso dice qué ocurre por dentro, por qué se hace en ese orden y cómo comprobar que salió bien. El orden no es una convención de estilo: es el que ejecuta el código, y alterarlo invalida el resultado.

### Paso 1 — Traer el dataset real y partirlo

**Qué ocurre.** Descarga `fashion_mnist` desde su proveedor y construye las tres particiones —`train`, `validation` y `test`— con la semilla de partición que le pases.

**Por qué.** La partición se fija **antes** de entrenar y su semilla (`--split-seed`) es independiente de la del entrenamiento (`--training-seed`). Si ambas fueran la misma, no podrías distinguir si un cambio en el resultado viene de haber repartido los datos de otro modo o de haber inicializado los pesos de otro modo.

```bash
neural-labs dataset --lab 08_gan_generation --quick --split-seed 42
```

**Cómo sabes que salió bien.** El comando termina sin error y deja el dataset en caché. Si no hay red, **falla**: no se rellena con datos inventados (`data/dataset.yaml` declara `fallback_to_generated_data: false`).

### Paso 2 — Comprobar que las particiones no se tocan

**Qué ocurre.** Recorre los identificadores de las tres particiones y verifica que ninguno aparece en dos de ellas.

**Por qué.** Una sola fila compartida entre `train` y `test` basta para que la métrica final quede inflada. Es el error más caro del oficio porque no produce ningún síntoma visible: el modelo simplemente «parece» mejor de lo que es.

```bash
neural-labs audit --lab 08_gan_generation --quick --split-seed 42
```

**Cómo sabes que salió bien.** La auditoría reporta cero solapamientos. Si reporta alguno, no sigas: el resto del laboratorio no significaría nada.

### Paso 3 — Mirar los datos antes de modelarlos

**Qué ocurre.** Genera el informe de calidad (valores faltantes, balance de clases, rangos) y el de deriva entre particiones.

**Por qué.** Un desbalance fuerte o una diferencia de distribución entre `train` y `test` cambia qué métrica tiene sentido leer. Descubrirlo después de entrenar obliga a repetirlo todo.

```bash
neural-labs quality --lab 08_gan_generation --quick --split-seed 42
```

**Cómo sabes que salió bien.** Obtienes `data_quality.json` y `drift_report.json`; ábrelos antes de decidir la configuración.

### Paso 4 — Estudiar la teoría del laboratorio

**Qué ocurre.** Leer [`theory.md`](theory.md): la idea central, el desarrollo matemático, los riesgos de interpretación y la bibliografía de la que sale todo eso.

**Por qué.** Sin esto, el entrenamiento es una caja que devuelve números. La teoría es lo que te permite decidir qué mirar y reconocer cuándo un resultado es sospechoso.

**Cómo sabes que salió bien.** Puedes responder, con tus palabras, qué calcula el modelo y por qué esa arquitectura encaja con la tarea `generation`.

### Paso 5 — Entrenar y seleccionar con `validation`

**Qué ocurre.** El entrenamiento recorre las épocas midiendo en `validation` después de cada una, y conserva el checkpoint con el mejor valor de `f1`.

**Por qué.** El conjunto de validación existe para tomar decisiones —arquitectura, hiperparámetros, cuándo parar—. Si esas decisiones se tomaran mirando `test`, `test` dejaría de ser una estimación de lo que pasará con datos nuevos y pasaría a ser parte del entrenamiento.

```bash
python labs/08_gan_generation/train.py --quick
# o, con control explícito de las dos semillas:
neural-labs train --lab 08_gan_generation --config baseline --split-seed 42 --training-seed 43
```

**Cómo sabes que salió bien.** En `runs/08_gan_generation/<ejecución>/` aparecen `history.csv` y `best_model.pt`; la métrica de validación mejora respecto de la primera época.

### Paso 6 — Compararte con la línea base

**Qué ocurre.** El repositorio entrena por su cuenta **PCA generativa y distribución real de referencia** y guarda su resultado, primero sobre `validation` y —solo al final— sobre `test`.

**Por qué.** Una métrica sola no dice si el modelo aporta algo. Puede que un método mucho más simple llegue igual de lejos, y entonces la complejidad añadida no está justificada. Esta comparación es la que convierte un número en un argumento.

**Cómo sabes que salió bien.** Comparas `metrics.json` con `baseline_metrics.json`. Si tu modelo no supera la línea base, el resultado del laboratorio es exactamente ese, y hay que reportarlo.

### Paso 7 — El sellado: `experiment.lock.json`

**Qué ocurre.** Antes de tocar `test`, el código escribe un archivo que fija el laboratorio, las dos semillas, la configuración, la métrica de selección, el checkpoint elegido y el hash del dataset.

**Por qué.** Es la frontera del experimento. A partir de ahí, cualquier ajuste que hagas mirando `test` queda a la vista: el sello dice qué habías decidido *antes* de ver el resultado final. Sin ese archivo, nadie —incluido tú dentro de un mes— puede distinguir una predicción de una racionalización.

**Cómo sabes que salió bien.** El archivo existe y su contenido coincide con lo que creías haber ejecutado.

### Paso 8 — Evaluar `test` una sola vez y medir la incertidumbre

**Qué ocurre.** Con el checkpoint congelado se evalúa `test`. En esta ruta la tarea es `generation`, así que el resultado se resume en las métricas propias de ese régimen y no en una predicción por ejemplo.

**Por qué.** Un número puntual esconde cuánto podría moverse. Por eso el paso siguiente —repetir con varias semillas— no es opcional aquí: es la única forma de saber cuánta de la diferencia observada es señal.

**Cómo sabes que salió bien.** Tienes `metrics.json` con el resultado final, y sabes que la comparación honesta llega con las repeticiones del paso siguiente.

### Paso 9 — Repetir con varias semillas de entrenamiento

**Qué ocurre.** Se repite el entrenamiento manteniendo **fija** la partición y cambiando solo la semilla de entrenamiento.

**Por qué.** Dos ejecuciones idénticas salvo por la inicialización pueden diferir bastante. Si no mides esa dispersión, corres el riesgo de celebrar una mejora que era una semilla afortunada.

```bash
neural-labs benchmark --lab 08_gan_generation --quick --split-seed 42 --training-seeds 41 42 43
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
| `model_spec.json` · `inference_contract.json` | Qué entrada espera el modelo y qué devuelve: lo que necesita quien lo despliegue. |
| `model_card.md` · `report.md` | La ficha del modelo y el informe legible de la ejecución. |
| `generated_samples.png` | **Propio de esta ruta.** Rejilla de muestras generadas: la evidencia visual de si hay diversidad o colapso. |

## ⚠️ Dónde suele perderse la gente

- **`--quick` no es una versión pequeña del resultado, es una prueba de que todo corre.** En esta ruta recorta a 1024 ejemplos de entrenamiento · 256 de validación · 256 de test · 2 épocas. Sirve para comprobar la instalación y la descarga; cualquier conclusión sobre el modelo exige la ejecución completa.
- **Cambiar algo después de ver `test` invalida la comparación.** Si al mirar el resultado final se te ocurre una mejora, la ruta correcta es volver a `validation`, decidir allí, y sellar de nuevo.
- **Las dos semillas no son intercambiables.** `--split-seed` cambia *qué datos* caen en cada partición; `--training-seed` cambia *cómo se inicializa y baraja* el entrenamiento. Para comparar modelos se fija la primera y se varía la segunda.
- **Aquí no vas a ver `predictions.csv` ni `confusion_matrix.png`, y no es un error.** La tarea es `generation`, y el código solo genera esos archivos cuando hay una predicción por ejemplo comparable contra una etiqueta.
- **Límite declarado de este dataset.** No usa anillos ni puntos inventados; entrena con prendas reales etiquetadas.

### Riesgos al interpretar los resultados

No usa anillos ni puntos inventados; entrena con prendas reales etiquetadas.

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

- Foster — *Generative Deep Learning* (2.ª ed., O'Reilly) — tratamiento práctico de GANs, DCGAN y evaluación de modelos generativos.
- Goodfellow, Bengio & Courville — *Deep Learning* (MIT Press, 2016), cap. 20 — modelos generativos profundos y fundamentos del marco adversarial.
- Goodfellow et al. (2014), *Generative Adversarial Nets*, NeurIPS — formulación original del juego minimax y prueba del óptimo p_g = p_data.
- Radford, Metz & Chintala (2016), *Unsupervised Representation Learning with Deep Convolutional GANs (DCGAN)*, ICLR — arquitectura convolucional que estabilizó el entrenamiento de GANs sobre imágenes.
- Fuente del dataset: https://github.com/zalandoresearch/fashion-mnist
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
| [🔭 Transformer para noticias](../../labs/07_transformer_attention/README.md) | [Las 31 rutas](../../parts/README.md) | [🕸️ GNN sobre red de citas](../../labs/09_gnn_graphs/README.md) |

**En este laboratorio:** **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🟣 [Parte 3 — Familias especializadas: generar, decidir, relacionar](../../parts/03-familias-especializadas.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/08_gan_generation/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
