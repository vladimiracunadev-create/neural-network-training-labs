# Segmentación semántica con U-Net

<!-- nav-top -->
> 🧭 **Ruta 27 / 31** · 🔬 [Parte 7 — Especializaciones avanzadas](../../parts/07-especializaciones-avanzadas.md)
>
> [⬅️ 🔧 Fine-tuning eficiente de transformer](../../advanced_labs/25_transformer_finetuning/README.md) · [🏠 Índice de rutas](../../parts/README.md) · [🎙️ Clasificación de audio con SpeechCommands ➡️](../../advanced_labs/27_audio_speechcommands/README.md)
>
> **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

## 🎯 Qué vas a hacer aquí

Segmentar mascota, fondo y contorno con IoU por clase.

Es la **ruta 27 de 31** del recorrido y pertenece a 🔬 la parte 7, *Especializaciones avanzadas*. Llegas desde **Fine-tuning eficiente de transformer** y lo que hagas aquí lo da por supuesto **Clasificación de audio con SpeechCommands**.

Trabajarás con el dataset **`oxford_iiit_pet_segmentation`** (Torchvision / University of Oxford, licencia: Consultar términos Oxford-IIIT Pet), y tendrás que superar la línea base **Máscara de clase mayoritaria**, decidiendo con la métrica `mean_iou` medida sobre `validation`. Nivel avanzado.

**Qué recibe el modelo como entrada:** imagen RGB y máscara trimap.

**Lo que conviene traer resuelto de las rutas anteriores:** CNN, visión, métricas por píxel.

**Al terminar deberías ser capaz de:**

- Segmentar mascota, fondo y contorno con IoU por clase.
- Interpretar mean_iou, iou_per_class
- Aplicar sellado de test y reproducibilidad

## 🧠 La teoría de este laboratorio

Esta sección es la explicación completa del tema. No hace falta abrir otro archivo para entender lo que viene después: aquí está la idea, la matemática que la sostiene y sus límites. (El mismo texto vive en `theory.md`, que es la fuente desde la que se genera esta guía, junto con la bibliografía del final.)

### De qué trata

Una CNN de clasificación responde «hay un gato en esta foto». La segmentación responde algo mucho más exigente: «este píxel es gato, este es fondo, y este es el borde entre ambos». La salida deja de ser una etiqueta y pasa a ser **una imagen de etiquetas**, del mismo tamaño que la entrada. Ese cambio de forma es el que obliga a cambiar de arquitectura.

El problema de fondo es un conflicto entre dos necesidades opuestas. Para decidir *qué* hay en una región hace falta contexto amplio: un parche de 8×8 píxeles de pelaje no distingue un gato de una alfombra. Ganar contexto significa reducir la resolución con submuestreos sucesivos, y cada submuestreo destruye información sobre *dónde* estaba exactamente cada cosa. Al final de un encoder típico la red sabe muy bien qué hay en la imagen y muy mal en qué píxel empieza. La segmentación necesita las dos respuestas a la vez.

La U-Net resuelve ese conflicto sin renunciar a ninguna de las dos: baja la resolución para ganar contexto y luego la sube para recuperar el detalle, pero en cada nivel de subida **reinyecta** los mapas de alta resolución que el encoder había producido antes de perderlos. Esas son las conexiones skip, y son la idea entera de la arquitectura. Su efecto se ve mejor en la clase más difícil de este laboratorio, el contorno de la mascota: son franjas de pocos píxeles que el submuestreo borra por completo y que solo el camino directo desde el encoder puede restituir.

El desbalance es el segundo protagonista. En una foto de mascota, el fondo ocupa la mayoría de los píxeles y el contorno una fracción mínima. Un modelo que prediga «fondo» en todas partes obtiene una exactitud por píxel alta y es completamente inútil: por eso la métrica no es la exactitud, sino la intersección sobre unión desglosada por clase, y por eso la pérdida necesita un término que no se deje dominar por la clase mayoritaria.

### La matemática, paso a paso

La segmentación semántica asigna a *cada* píxel de la imagen una etiqueta de clase. Formalmente, dada una entrada X ∈ ℝ^(H×W×3) se busca una función que produzca un mapa de probabilidades ŷ ∈ ℝ^(H×W×C), donde C es el número de clases (aquí: mascota, fondo y contorno). Es una clasificación densa: en lugar de una etiqueta por imagen, se predice una por posición espacial. Las **redes totalmente convolucionales** (FCN) de Long, Shelhamer y Darrell hicieron esto viable al sustituir las capas densas finales de una CNN por convoluciones, permitiendo salidas del tamaño de la imagen mediante *upsampling* (convoluciones transpuestas).

La **U-Net** refina esta idea con una arquitectura simétrica en forma de U. El **encoder** (camino de contracción) aplica bloques de convolución seguidos de submuestreo (max-pooling), reduciendo la resolución espacial y aumentando la profundidad de canales: captura el *qué* (contexto semántico) pero pierde el *dónde* (detalle espacial). El **decoder** (camino de expansión) revierte el proceso con upsampling progresivo hasta recuperar la resolución original. La clave son las **conexiones skip**: en cada nivel, los mapas de características del encoder se concatenan con los del decoder de igual resolución. Así se reinyecta la información espacial de alta frecuencia (bordes, contornos finos) que el submuestreo había diluido, resolviendo el compromiso entre contexto y localización. Esto es decisivo para la clase "contorno", que ocupa franjas delgadas de pocos píxeles.

El entrenamiento minimiza una **pérdida por píxel**, típicamente la entropía cruzada promediada sobre todas las posiciones:

ℒ_CE = −(1 / (H·W)) · Σ_(i,j) Σ_(c=1..C) y_(i,j,c) · log ŷ_(i,j,c),

donde la probabilidad por clase se obtiene con un softmax sobre el eje de canales, ŷ_(i,j,c) = e^(z_(i,j,c)) / Σ_k e^(z_(i,j,k)). Como las clases suelen estar desbalanceadas (el fondo domina), se complementa con la **pérdida Dice**, ℒ_Dice = 1 − (2·Σ ŷ·y + ε) / (Σ ŷ + Σ y + ε), donde ε > 0 evita división por cero; Dice premia el solape directo y es más robusta al desbalance.

La métrica principal es la **intersección sobre unión** (IoU), o índice de Jaccard, definida por clase como

IoU = |A ∩ B| / |A ∪ B| = TP / (TP + FP + FN),

siendo A la máscara predicha y B la real. Vale 1 si coinciden perfectamente y 0 si no se solapan; el *mean IoU* promedia sobre clases y es el estándar de la segmentación semántica. La línea base "máscara de clase mayoritaria" predice siempre la clase más frecuente: fija un piso trivial que la U-Net debe superar ampliamente para demostrar que aprende estructura real y no solo la proporción de píxeles de fondo.

### La aritmética de la U: cuánto se pierde y cuánto se recupera

Cada nivel del encoder aplica convoluciones y un max-pooling de factor 2. Tras L niveles, un mapa de H×W queda en (H/2^L)×(W/2^L): con L = 4 y una entrada de 128×128, el cuello de botella mide 8×8. Ese es el precio explícito del contexto.

Lo que se gana a cambio se mide con el **campo receptivo**, la región de la entrada que influye en una sola activación. Para una pila de capas, crece según

R_ℓ = R_(ℓ−1) + (F_ℓ − 1) · Π_(i<ℓ) s_i,

donde F_ℓ es el tamaño del filtro de la capa ℓ y s_i los strides acumulados de las anteriores. La consecuencia es que cada submuestreo **duplica** el efecto de las convoluciones posteriores: una convolución 3×3 tras cuatro poolings ve una ventana de 48 píxeles de la imagen original, mientras que la misma convolución al principio ve solo 3. Por eso el fondo se clasifica bien en las capas profundas y el borde no.

La subida usa **convoluciones transpuestas**, cuya dimensión de salida invierte la fórmula de la convolución normal:

dimensión_salida = (dimensión_entrada − 1)·s − 2p + F.

Con s = 2, F = 2 y p = 0 se duplica exactamente la resolución. Tras cada subida se concatena el mapa del encoder de igual resolución —la conexión skip— de modo que el decoder recibe C_dec + C_enc canales: la información semántica que trae de abajo y la información espacial que nunca pasó por el cuello de botella. Es una concatenación, no una suma, y esa distinción importa: la red aprende con sus pesos cómo combinar ambas fuentes en vez de imponerles el mismo peso por construcción.

### Cómo se combate el desbalance en la pérdida

La entropía cruzada por píxel trata todas las posiciones por igual, así que en una imagen con 80 % de fondo el gradiente está dominado por píxeles fáciles. Hay dos correcciones habituales, y el laboratorio usa ambas.

La primera es **ponderar las clases** en la entropía cruzada, ℒ_CE^w = −(1/(H·W)) · Σ_(i,j) Σ_c w_c · y_(i,j,c) · log ŷ_(i,j,c), con pesos inversamente proporcionales a la frecuencia, típicamente w_c ∝ 1 / f_c o w_c ∝ 1 / √f_c, siendo f_c la fracción de píxeles de la clase c. La raíz cuadrada modera la corrección: la ponderación inversa pura suele desestabilizar el entrenamiento porque dispara el gradiente de clases con muy pocos píxeles.

La segunda es sumar el término **Dice**, que no mide aciertos por píxel sino solape entre conjuntos y por tanto es insensible al tamaño del fondo. La pérdida total queda

ℒ = ℒ_CE^w + λ · ℒ_Dice,   con   ℒ_Dice = 1 − (1/C) · Σ_c (2·Σ_(i,j) ŷ_(i,j,c)·y_(i,j,c) + ε) / (Σ_(i,j) ŷ_(i,j,c) + Σ_(i,j) y_(i,j,c) + ε).

El Dice se calcula sobre las probabilidades ŷ sin binarizar, lo que lo hace diferenciable —una versión "blanda" del coeficiente clásico— y permite optimizarlo por descenso de gradiente. Los dos términos se complementan: la entropía cruzada da gradiente denso y estable desde el primer paso; el Dice orienta el entrenamiento hacia la métrica que de verdad se reporta.

### Por qué Dice e IoU no son la misma cifra

Ambos miden solape y se confunden con frecuencia, pero no coinciden. Con TP, FP y FN contados sobre píxeles,

IoU = TP / (TP + FP + FN),   Dice = 2·TP / (2·TP + FP + FN),

y están ligados por una relación monótona exacta:

Dice = 2·IoU / (1 + IoU),   equivalentemente   IoU = Dice / (2 − Dice).

Como Dice ≥ IoU siempre (con igualdad solo en 0 y en 1), **el Dice siempre se ve mejor**. Un IoU de 0,50 es un Dice de 0,67; un IoU de 0,80, un Dice de 0,89. Reportar uno creyendo que es el otro infla el resultado sin que nada falle visiblemente, y es la razón de que este laboratorio fije el `mean_iou` como métrica de selección y exija además el desglose `iou_per_class`: un mean IoU alto puede convivir con un IoU de contorno cercano a cero, que es exactamente el fallo que la arquitectura pretendía evitar.

### Qué conviene graficar

Imagen, máscara real, máscara predicha, IoU por clase y mapas intermedios. Los mapas intermedios muestran cómo el encoder abstrae el contexto y las conexiones skip recuperan el detalle; el IoU por clase expone en qué categoría (mascota, fondo o contorno) falla más el modelo.

### Qué se mide y con qué se decide

El laboratorio reporta `mean_iou`, `iou_per_class`, `pixel_accuracy`, `dice`. De todas ellas, la que **decide** qué modelo se conserva es `mean_iou`, y se mide siempre sobre `validation`: es la única forma de que `test` siga siendo una estimación honesta de lo que pasará con datos nuevos.

## 📓 Los tres cuadernos

El laboratorio incluye tres cuadernos Jupyter de **21 celdas** cada uno, de las cuales **10 son de código ejecutable**. Los tres recorren el mismo camino —descargar el dataset real, auditar la partición, entrenar, sellar el experimento y evaluar `test` una vez— y se diferencian en cuánto viene resuelto:

| Cuaderno | Qué trae | Cuándo usarlo |
|---|---|---|
| [📓 `notebook.ipynb`](notebook.ipynb) | El recorrido completo con **todo el código escrito y ejecutable**, celda a celda, intercalado con las explicaciones. | Para leer y ejecutar de principio a fin, entendiendo qué hace cada paso. |
| [✏️ `notebook_student.ipynb`](notebook_student.ipynb) | El mismo recorrido con **3 celdas vaciadas**, marcadas con `# YOUR CODE HERE`, que hay que completar. | Para practicar: se ejecuta igual, pero falla hasta que completas los huecos. |
| [✅ `notebook_solution.ipynb`](notebook_solution.ipynb) | Las celdas anteriores ya resueltas, marcadas con `# SOLUCIÓN DE REFERENCIA`. | Para contrastar tu respuesta después de intentarlo. |

> **Aviso honesto sobre el estado actual.** Hoy `notebook.ipynb` y `notebook_solution.ipynb` tienen **el mismo contenido**, y los ejercicios que los separan del cuaderno de estudiante son **3**. Es decir: el código del laboratorio está completo y es ejecutable en los tres, pero la versión de estudiante todavía no propone una práctica extensa. Está anotado en el [roadmap](../../ROADMAP.md) y se dice aquí para que nadie descubra el límite después de abrir el archivo.

### Cómo abrirlos

Los cuadernos necesitan el extra `notebooks`, que instala Jupyter junto con el paquete:

```bash
pip install -e ".[dev,notebooks]"
jupyter lab advanced_labs/26_segmentation_unet/notebook.ipynb
```

También se abren desde VS Code —con la extensión de Jupyter— haciendo doble clic en el archivo, o desde la interfaz clásica con `jupyter notebook`. El primer arranque descarga el dataset real desde su proveedor, así que la primera ejecución tarda más y **requiere conexión**.

Si prefieres ejecutar sin abrir un cuaderno, `train.py` hace exactamente lo mismo desde la terminal, y la sección de comandos de arriba explica cada opción.

## 🖥️ Los comandos, explicados

Todo el laboratorio se maneja con una sola herramienta de terminal, `neural-labs`, que se instala junto con el paquete (`pip install -e ".[dev,notebooks]"`). Cada subcomando hace **una** cosa del protocolo, y por eso se pueden ejecutar por separado: preparar datos, auditar la partición, entrenar, repetir con varias semillas.

La forma general es siempre la misma:

```bash
neural-labs <subcomando> --track <identificador> [opciones]
```

| Opción | Valor por defecto | Valores | Qué hace y cuándo cambiarla |
|---|---|---|---|
| `--track` | `26_segmentation_unet` | obligatorio | Qué especialización se entrena. Solo acepta los seis identificadores existentes. |
| `--quick` | desactivado | — | Reduce datos y épocas para comprobar que la ruta corre de extremo a extremo. |
| `--split-seed N` | `42` | entero | Semilla que decide **qué ejemplo cae en qué partición**. Se mantiene fija al comparar. |
| `--training-seed N` | `42` | entero | Semilla de la inicialización de pesos y del barajado. Es la que se varía para medir dispersión. |
| `--device` | `auto` | `auto` · `cpu` · `cuda` · `mps` | Dónde entrenar. `auto` elige GPU si la hay. |
| `--output-dir` | `runs-advanced` | ruta | Dónde se escribe el directorio de la ejecución. |

### Lo mismo desde Python

```python
from neural_labs.advanced.training import train_advanced

resultado = train_advanced(
    "26_segmentation_unet",
    quick=True,
    split_seed=42,
    training_seed=43,
)

print(resultado["run_dir"])
print(resultado["metrics"])
```

## 🪜 Paso a paso

Cada paso dice qué ocurre por dentro, por qué se hace en ese orden y cómo comprobar que salió bien. El orden no es una convención de estilo: es el que ejecuta el código, y alterarlo invalida el resultado.

### Paso 1 — Estudiar la teoría antes de ejecutar nada

**Qué ocurre.** Leer [`theory.md`](theory.md), que desarrolla Arquitectura encoder-decoder, conexiones skip, pérdida por píxel e intersección sobre unión. y cita las obras y papers de los que procede.

**Por qué.** Estas rutas usan arquitecturas donde un error de comprensión no se manifiesta como un fallo, sino como un número plausible pero equivocado.

**Cómo sabes que salió bien.** Puedes explicar qué mide `mean_iou` y por qué es la métrica de selección aquí.

### Paso 2 — Ejecutar la versión rápida

**Qué ocurre.** Descarga el dataset y los pesos preentrenados desde su proveedor, entrena una versión reducida y escribe la ejecución en `runs-advanced/`.

**Por qué.** Antes de gastar horas de cómputo conviene comprobar que la descarga, el entorno y la ruta completa funcionan de extremo a extremo.

```bash
neural-labs train-advanced --track 26_segmentation_unet --quick
```

**Cómo sabes que salió bien.** Termina sin error y deja `metrics.json`, `history.json` y `best_model.pt` en el directorio de la ejecución.

### Paso 3 — Entrenar en serio y seleccionar con `validation`

**Qué ocurre.** Se entrena el modelo completo conservando el checkpoint con el mejor valor de `mean_iou` en validación, y se sella el experimento antes de evaluar `test`.

**Por qué.** Igual que en las rutas centrales: `validation` decide, `test` solo confirma, y el sello deja por escrito qué se había decidido antes de mirar.

```bash
neural-labs train-advanced --track 26_segmentation_unet --split-seed 42 --training-seed 43
```

**Cómo sabes que salió bien.** Existe `experiment.lock.json` y `metrics.json` incluye tanto el valor de validación como el de test.

### Paso 4 — Repetir con otra semilla de entrenamiento

**Qué ocurre.** Se repite el entrenamiento con la misma partición y distinta semilla de entrenamiento.

**Por qué.** Estas arquitecturas —adversariales, contrastivas, de difusión— son especialmente sensibles a la inicialización: una sola ejecución no permite distinguir una mejora de una casualidad.

```bash
neural-labs train-advanced --track 26_segmentation_unet --split-seed 42 --training-seed 44
```

**Cómo sabes que salió bien.** Puedes reportar el rango entre ejecuciones, no un único número.

### Paso 5 — Documentar los límites

**Qué ocurre.** Registrar el resultado junto con la limitación declarada de la ruta y responder [`assessment.md`](assessment.md).

**Por qué.** En generación y aprendizaje autosupervisado las métricas son aproximaciones: sin declarar qué NO demuestran, invitan a conclusiones que los números no sostienen.

**Cómo sabes que salió bien.** Tu reporte dice qué mejoró, cuánto costó y en qué condiciones no esperarías el mismo resultado.

## 🔍 Cómo leer lo que produce la ejecución

Cada ejecución escribe su propio directorio con nombre único, de modo que dos corridas nunca se pisan. Esto es lo que encontrarás dentro:

| Archivo | Qué contiene y qué mirar |
|---|---|
| `config.json` | Track, semillas, dispositivo y opciones con las que se lanzó. |
| `dataset_manifest.json` | Fuente, licencia y número de ejemplos por partición. |
| `best_model.pt` | El checkpoint seleccionado por validación. |
| `experiment.lock.json` | El sello: qué se decidió antes de abrir `test`. |
| `history.json` | La métrica de validación época a época. |
| `metrics.json` | Resultado de validación y de test, ya con el modelo congelado. |

## ⚠️ Dónde suele perderse la gente

- **Cambiar algo después de ver `test` invalida la comparación.** Si al mirar el resultado final se te ocurre una mejora, la ruta correcta es volver a `validation`, decidir allí, y sellar de nuevo.
- **Las dos semillas no son intercambiables.** `--split-seed` cambia *qué datos* caen en cada partición; `--training-seed` cambia *cómo se inicializa y baraja* el entrenamiento. Para comparar modelos se fija la primera y se varía la segunda.
- **Límite declarado de este dataset.** Las imágenes se concentran en mascotas y fondos cotidianos; no generaliza a segmentación médica o industrial.

### Riesgos al interpretar los resultados

Las imágenes se concentran en mascotas y fondos cotidianos; no generaliza a segmentación médica o industrial. Un IoU global alto puede ocultar mal desempeño en clases minoritarias como el contorno, por lo que conviene leer siempre el IoU desglosado por clase.

## ✅ Antes de darlo por terminado

Y cuando tienes estos entregables:

- [ ] notebook ejecutado
- [ ] reporte experimental
- [ ] model card

El plan experimental con la tabla que hay que completar está en `experiments.md`, y las preguntas con su rúbrica, en `assessment.md`. Ambos documentos se abren desde la barra de navegación de arriba.

### Para ir más lejos

- Cambia una decisión experimental y justifícala con el resultado en `validation`, no con la intuición.
- Analiza los errores por clase o por segmento: casi siempre se concentran en un subconjunto reconocible.
- Compara costo, precisión y latencia; el mejor modelo no siempre es el que gana por décimas.
- Documenta sesgos, limitaciones y usos para los que **no** recomendarías este modelo.

## 📚 Fuentes

La teoría de arriba no es original de este repositorio: se apoya en la literatura de referencia del área y en los papers originales de cada arquitectura. Estas son las obras concretas, y lo que aporta cada una:

> Las referencias apuntan a las obras; no se reproduce su contenido, la redacción es original.

- Long, Shelhamer & Darrell (2015), *Fully Convolutional Networks for Semantic Segmentation*, CVPR — funda la segmentación densa reemplazando capas densas por convoluciones y upsampling.
- Ronneberger, Fischer & Brox (2015), *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI — encoder-decoder simétrico con conexiones skip para localización precisa.
- Goodfellow, Bengio & Courville — *Deep Learning* (MIT Press, 2016), cap. 9 — fundamentos de las redes convolucionales que sustentan el encoder-decoder.

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

Y fuera de la carpeta, tres referencias que esta guía usa: el catálogo `configs/advanced_tracks.yaml` —de donde salen el objetivo, la línea base y las métricas—, el código `src/neural_labs/advanced/training.py` —que define el orden de los pasos y los archivos que escribe cada ejecución— y `docs/experiment-protocol.md`, con la regla general del protocolo.

Los datasets se descargan de su proveedor original y conservan su licencia; este repositorio no los redistribuye ni sustituye una descarga fallida por datos generados.

<!-- nav-bottom -->
## 🧭 Navegación del recorrido

| ⬅️ Laboratorio anterior | 🏠 Índice | Laboratorio siguiente ➡️ |
|---|:---:|---|
| [🔧 Fine-tuning eficiente de transformer](../../advanced_labs/25_transformer_finetuning/README.md) | [Las 31 rutas](../../parts/README.md) | [🎙️ Clasificación de audio con SpeechCommands](../../advanced_labs/27_audio_speechcommands/README.md) |

**En este laboratorio:** **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🔬 [Parte 7 — Especializaciones avanzadas](../../parts/07-especializaciones-avanzadas.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/26_segmentation_unet/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
