# Clasificación de audio con SpeechCommands

<!-- nav-top -->
> 🧭 **Ruta 28 / 31** · 🔬 [Parte 7 — Especializaciones avanzadas](../../parts/07-especializaciones-avanzadas.md)
>
> [⬅️ 🧷 Segmentación semántica con U-Net](../../advanced_labs/26_segmentation_unet/README.md) · [🏠 Índice de rutas](../../parts/README.md) · [🖌️ WGAN-GP sobre Fashion-MNIST ➡️](../../advanced_labs/28_wgan_gp/README.md)
>
> **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

## 🎯 Qué vas a hacer aquí

Clasificar comandos hablados desde waveform y log-mel spectrograms.

Es la **ruta 28 de 31** del recorrido y pertenece a 🔬 la parte 7, *Especializaciones avanzadas*. Llegas desde **Segmentación semántica con U-Net** y lo que hagas aquí lo da por supuesto **WGAN-GP sobre Fashion-MNIST**.

Trabajarás con el dataset **`speechcommands_v0.02`** (Torchaudio / Google Speech Commands, licencia: Creative Commons BY 4.0), y tendrás que superar la línea base **MFCC + regresión logística**, decidiendo con la métrica `accuracy` medida sobre `validation`. Nivel avanzado.

**Qué recibe el modelo como entrada:** audio mono de un segundo a 16 kHz.

**Lo que conviene traer resuelto de las rutas anteriores:** CNN, señales, transformada tiempo-frecuencia.

**Al terminar deberías ser capaz de:**

- Clasificar comandos hablados desde waveform y log-mel spectrograms.
- Interpretar accuracy, macro_f1
- Aplicar sellado de test y reproducibilidad

## 🧠 La teoría de este laboratorio

Esta sección es la explicación completa del tema. No hace falta abrir otro archivo para entender lo que viene después: aquí está la idea, la matemática que la sostiene y sus límites. (El mismo texto vive en `theory.md`, que es la fuente desde la que se genera esta guía, junto con la bibliografía del final.)

### De qué trata

Un clip de un segundo a 16 kHz son 16 000 números. Puestos en fila, una red densa necesitaría decenas de millones de pesos solo en su primera capa, y aun así no aprendería gran cosa: la información que distingue «arriba» de «abajo» no está en el valor de cada muestra individual, sino en **cómo cambia el contenido en frecuencia a lo largo del tiempo**. Dos grabaciones de la misma palabra, desplazadas unos milisegundos o dichas con otro tono, tienen formas de onda numéricamente distintas y contenido fonético idéntico.

La idea del laboratorio es cambiar de representación antes de modelar. La transformada de Fourier de corto tiempo convierte la señal unidimensional en una imagen bidimensional —tiempo en un eje, frecuencia en el otro, energía como intensidad— donde los rasgos que importan se vuelven **patrones locales visibles**: las bandas horizontales de los formantes vocálicos, las ráfagas anchas de las consonantes oclusivas, las transiciones diagonales entre fonemas. Una vez que el problema tiene forma de imagen, la herramienta correcta es la que ya se estudió en la ruta 03: una CNN 2D con pesos compartidos, que detecta esos patrones aparezcan donde aparezcan.

Sobre esa representación se aplican dos ajustes que no vienen de las matemáticas sino de la fisiología del oído: comprimir el eje de frecuencia según la escala mel, porque distinguimos mucho mejor entre 300 y 400 Hz que entre 8 000 y 8 100, y tomar el logaritmo de la energía, porque percibimos la intensidad de forma aproximadamente logarítmica. El resultado —el espectrograma log-mel— no es una elección arbitraria: es un preprocesamiento que descarta justo la información que el oído humano tampoco usa para reconocer palabras.

El laboratorio se cierra con la pregunta que separa un modelo de laboratorio de uno usable: qué pasa cuando el micrófono capta ruido de fondo. Un modelo que aprendió rasgos fonéticos estables degrada suavemente; uno que memorizó particularidades del set limpio se derrumba. Esa diferencia no aparece en la exactitud sobre datos limpios, y por eso se mide aparte.

### La matemática, paso a paso

El audio crudo es una **forma de onda** (waveform): una señal x[t] de amplitud muestreada en el tiempo, aquí a 16 kHz durante un segundo (≈16 000 muestras por clip). Clasificar directamente sobre esa secuencia larga y unidimensional es difícil, porque la información fonética relevante vive en el *contenido frecuencial* que varía a lo largo del tiempo. Por eso se transforma la señal a una representación tiempo-frecuencia mediante la **transformada de Fourier de corto tiempo** (STFT): se divide la onda en ventanas solapadas y se calcula el espectro de cada una,

X(m, k) = Σ_(n) x[n] · w[n − m] · e^(−j·2π·k·n / N),

donde w es una ventana (p. ej. Hann), m indexa el tiempo y k la frecuencia. El **espectrograma** es la magnitud al cuadrado |X(m, k)|²: una "imagen" 2D donde un eje es tiempo y el otro frecuencia, y la intensidad es la energía.

Sobre ese espectrograma se aplican dos transformaciones inspiradas en la percepción auditiva humana. Primero, la **escala mel** comprime el eje de frecuencia con un banco de filtros triangulares espaciados según mel(f) = 2595 · log₁₀(1 + f/700), que da más resolución a las frecuencias bajas —donde el oído distingue mejor— y agrupa las altas. Segundo, se toma el **logaritmo** de la energía, log(mel-energía + ε), imitando que percibimos la intensidad de forma aproximadamente logarítmica y comprimiendo el enorme rango dinámico; ε > 0 evita log(0). El resultado es el **espectrograma log-mel**, la entrada del modelo. Los **MFCC** (coeficientes cepstrales en frecuencias mel) van un paso más allá aplicando una transformada coseno discreta que decorrelaciona los canales mel; se usan en la línea base MFCC + regresión logística.

Como el log-mel es una imagen 2D, el modelo natural es una **CNN 2D**. Cada capa convolucional desliza filtros aprendidos K sobre la entrada, (X * K)(i, j) = Σ_(a,b) X(i+a, j+b) · K(a, b), detectando patrones locales tiempo-frecuencia (formantes, transiciones, ráfagas de energía) con pesos compartidos; el apilamiento con submuestreo construye representaciones cada vez más abstractas hasta una capa densa con softmax que produce p(clase | audio). El entrenamiento minimiza la entropía cruzada ℒ = −Σ_c y_c · log ŷ_c, y esta idea de tratar el audio como imagen espectral es la que Hershey et al. escalaron a clasificación de audio a gran escala.

La **robustez ante ruido** es central: en uso real el micrófono capta fondo, reverberación y solapamientos. Se evalúa perturbando la entrada, por ejemplo x̃ = x + n con ruido n de una relación señal-ruido dada (SNR = 10·log₁₀(P_señal / P_ruido) dB), y midiendo cuánto cae la accuracy. Un modelo que aprende rasgos fonéticos estables degrada poco; uno que memoriza artefactos del set limpio colapsa. Esto también motiva aumentaciones (desplazamiento temporal, ruido, enmascarado de bandas) durante el entrenamiento.

### El compromiso que decide el tamaño de la ventana

La STFT obliga a una elección que no tiene solución óptima, solo compromisos. Con una ventana de N muestras a frecuencia de muestreo f_s, la resolución en frecuencia es

Δf = f_s / N,

y la resolución temporal es la duración de la propia ventana, Δt = N / f_s. Las dos están ligadas por Δf · Δt = 1: **mejorar una empeora la otra exactamente en la misma proporción**. Es el principio de incertidumbre aplicado al análisis de señales, y no se puede eludir eligiendo mejor la ventana.

Con los valores habituales en voz —f_s = 16 000 Hz y una ventana de 25 ms, es decir N = 400 muestras— resulta Δf = 40 Hz. Suficiente para separar formantes, que distan cientos de hercios, y lo bastante corta para que el contenido fonético no cambie apreciablemente dentro de la ventana. Ventanas mucho más largas mezclarían dos fonemas en un mismo espectro; mucho más cortas no resolverían la estructura armónica de una vocal.

Las ventanas se solapan con un **salto** (hop) de unos 10 ms, así que el número de tramas de un clip de un segundo es

T = ⌊(longitud − N) / hop⌋ + 1 ≈ ⌊(16 000 − 400) / 160⌋ + 1 = 98.

Con 64 bandas mel, la entrada de la red es un tensor de 64×98: una imagen pequeña, comparable a las de la ruta 03. Esa es la razón práctica de que el laboratorio corra en CPU. El solape no es opcional: sin él, un fonema que caiga en la frontera entre dos ventanas quedaría partido y atenuado por la ventana de Hann, que pesa poco los extremos. El solape garantiza que todo instante quede bien representado en al menos una trama.

### El banco mel, escrito como una multiplicación de matrices

El paso de espectrograma lineal a mel no es una operación exótica: es un producto matricial. Se construye un banco de M filtros triangulares —aquí M = 64— cuyos centros están **igualmente espaciados en la escala mel**, no en hercios. Con

mel(f) = 2595 · log₁₀(1 + f / 700),   y su inversa   f(m) = 700 · (10^(m/2595) − 1),

se reparten M + 2 puntos uniformemente entre mel(f_min) y mel(f_max), se convierten de vuelta a hercios y cada terna consecutiva define un triángulo. El filtro m vale 0 fuera de su intervalo, sube linealmente hasta su centro y baja de nuevo:

H_m(k) = (k − k_(m−1)) / (k_m − k_(m−1)) si k ∈ [k_(m−1), k_m];   (k_(m+1) − k) / (k_(m+1) − k_m) si k ∈ [k_m, k_(m+1)];   0 en el resto.

Como los centros están espaciados en mel, los triángulos son **estrechos en frecuencias bajas y anchos en altas**: ahí está toda la compresión perceptual. Agrupando los filtros en una matriz H ∈ ℝ^(M×K), el espectrograma mel es simplemente S_mel = H · |X|², y el log-mel, log(S_mel + ε). La operación reduce K ≈ 201 bins de frecuencia a 64 bandas: una reducción de dimensión de más de tres veces que descarta precisamente la resolución que el oído no aprovecha.

Los **MFCC** de la línea base añaden un paso más, una transformada coseno discreta sobre las bandas log-mel:

c_n = Σ_(m=1..M) log(S_mel[m]) · cos( π·n·(m − ½) / M ),   n = 0, …, M−1,

quedándose con los primeros 13 coeficientes. La DCT decorrelaciona las bandas —que están muy correlacionadas entre vecinas— y concentra la información en pocas dimensiones, lo que es imprescindible para un clasificador lineal. Pero al hacerlo **destruye la estructura local** de la representación: dos coeficientes cepstrales contiguos ya no describen frecuencias contiguas, y por tanto una convolución sobre ellos no tiene sentido. Esa es la razón técnica de que la línea base use MFCC con regresión logística y la red use log-mel: cada representación está hecha para un modelo distinto, y comparar ambas mide cuánto aporta conservar la vecindad tiempo-frecuencia.

### Ruido y enmascarado, con números

La relación señal-ruido cuantifica la perturbación:

SNR_dB = 10 · log₁₀( P_señal / P_ruido ),

donde P es la potencia media, P = (1/T)·Σ_t x[t]². Para inyectar ruido a una SNR objetivo se escala el ruido por el factor a = √( P_señal / (P_ruido · 10^(SNR/10)) ) y se suma, x̃ = x + a·n. Evaluar la exactitud a 20, 10 y 0 dB traza una **curva de degradación**: 20 dB es una habitación tranquila, 10 dB un ambiente con conversación de fondo, y 0 dB significa que ruido y voz tienen la misma potencia. La forma de esa curva dice más sobre la utilidad del modelo que la exactitud sobre audio limpio.

El enmascarado de **SpecAugment** actúa directamente sobre el log-mel, sin volver a la onda: se anulan bandas de frecuencia consecutivas [f₀, f₀ + f) y tramos de tiempo [t₀, t₀ + t), con f y t muestreados uniformemente hasta un máximo y las posiciones al azar. Es barato —una máscara sobre un tensor— y obliga al modelo a no depender de una única banda ni de un único instante, que es exactamente la fragilidad que el ruido explota.

### Qué conviene graficar

Waveform, espectrograma, errores por palabra y ruido. Contrastar la onda cruda con su espectrograma log-mel muestra por qué la representación 2D facilita la clasificación; el desglose de errores por palabra revela confusiones entre comandos fonéticamente parecidos, y las pruebas con ruido cuantifican la robustez.

### Qué se mide y con qué se decide

El laboratorio reporta `accuracy`, `macro_f1`, `confusion_matrix`, `noise_robustness`. De todas ellas, la que **decide** qué modelo se conserva es `accuracy`, y se mide siempre sobre `validation`: es la única forma de que `test` siga siendo una estimación honesta de lo que pasará con datos nuevos.

## 📓 Los tres cuadernos

El laboratorio se puede recorrer en Jupyter, y trae tres cuadernos con papeles distintos. Los tres siguen el mismo camino —descargar el dataset real, auditar la partición, entrenar, sellar el experimento y evaluar `test` una vez—; lo que cambia es qué te toca escribir a ti:

| Cuaderno | Qué trae | Cuándo usarlo |
|---|---|---|
| [📓 `notebook.ipynb`](notebook.ipynb) | El **recorrido de referencia**: 22 celdas (10 de código) con **todo el código escrito y ejecutable**, intercalado con las explicaciones. No trae ejercicios. | Para leer y ejecutar de principio a fin. |
| [✏️ `notebook_student.ipynb`](notebook_student.ipynb) | El mismo recorrido más **8 ejercicios evaluables** (37 celdas en total). Las celdas de ejercicio están marcadas con `# YOUR CODE HERE` y debajo de cada una hay una comprobación. | Para practicar. |
| [✅ `notebook_solution.ipynb`](notebook_solution.ipynb) | Los mismos ejercicios **resueltos**, marcados con `# SOLUCIÓN DE REFERENCIA`. Cada solución se ejecuta en la integración continua, así que se sabe que pasa. | Para contrastar después de intentarlo. |

### Qué se practica en los ejercicios

Cinco de ellos no son de arquitectura sino del **contrato experimental**, que es lo que distingue a estos laboratorios de un tutorial: auditar la partición, decidir con `validation`, compararse con la línea base, sellar antes de abrir `test` y dejar el plan por escrito. Se resuelven con Python estándar —**sin descargar el dataset ni entrenar**—, así que se corrigen en segundos y sin GPU, y cada uno está parametrizado con los valores de este laboratorio: su métrica de selección, su línea base y su experimento propio.

### Cómo abrirlos

Los cuadernos necesitan el extra `notebooks`, que instala Jupyter junto con el paquete:

```bash
pip install -e ".[dev,notebooks]"
jupyter lab advanced_labs/27_audio_speechcommands/notebook.ipynb
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
| `--track` | `27_audio_speechcommands` | obligatorio | Qué especialización se entrena. Solo acepta los seis identificadores existentes. |
| `--quick` | desactivado | — | Reduce datos y épocas para comprobar que la ruta corre de extremo a extremo. |
| `--split-seed N` | `42` | entero | Semilla que decide **qué ejemplo cae en qué partición**. Se mantiene fija al comparar. |
| `--training-seed N` | `42` | entero | Semilla de la inicialización de pesos y del barajado. Es la que se varía para medir dispersión. |
| `--device` | `auto` | `auto` · `cpu` · `cuda` · `mps` | Dónde entrenar. `auto` elige GPU si la hay. |
| `--output-dir` | `runs-advanced` | ruta | Dónde se escribe el directorio de la ejecución. |

### Lo mismo desde Python

```python
from neural_labs.advanced.training import train_advanced

resultado = train_advanced(
    "27_audio_speechcommands",
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

**Qué ocurre.** Leer [`theory.md`](theory.md), que desarrolla Waveform, espectrograma log-mel, convolución 2D y robustez ante ruido. y cita las obras y papers de los que procede.

**Por qué.** Estas rutas usan arquitecturas donde un error de comprensión no se manifiesta como un fallo, sino como un número plausible pero equivocado.

**Cómo sabes que salió bien.** Puedes explicar qué mide `accuracy` y por qué es la métrica de selección aquí.

### Paso 2 — Ejecutar la versión rápida

**Qué ocurre.** Descarga el dataset y los pesos preentrenados desde su proveedor, entrena una versión reducida y escribe la ejecución en `runs-advanced/`.

**Por qué.** Antes de gastar horas de cómputo conviene comprobar que la descarga, el entorno y la ruta completa funcionan de extremo a extremo.

```bash
neural-labs train-advanced --track 27_audio_speechcommands --quick
```

**Cómo sabes que salió bien.** Termina sin error y deja `metrics.json`, `history.json` y `best_model.pt` en el directorio de la ejecución.

### Paso 3 — Entrenar en serio y seleccionar con `validation`

**Qué ocurre.** Se entrena el modelo completo conservando el checkpoint con el mejor valor de `accuracy` en validación, y se sella el experimento antes de evaluar `test`.

**Por qué.** Igual que en las rutas centrales: `validation` decide, `test` solo confirma, y el sello deja por escrito qué se había decidido antes de mirar.

```bash
neural-labs train-advanced --track 27_audio_speechcommands --split-seed 42 --training-seed 43
```

**Cómo sabes que salió bien.** Existe `experiment.lock.json` y `metrics.json` incluye tanto el valor de validación como el de test.

### Paso 4 — Repetir con otra semilla de entrenamiento

**Qué ocurre.** Se repite el entrenamiento con la misma partición y distinta semilla de entrenamiento.

**Por qué.** Estas arquitecturas —adversariales, contrastivas, de difusión— son especialmente sensibles a la inicialización: una sola ejecución no permite distinguir una mejora de una casualidad.

```bash
neural-labs train-advanced --track 27_audio_speechcommands --split-seed 42 --training-seed 44
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
- **Límite declarado de este dataset.** Acentos, micrófonos y ambientes no están representados uniformemente.

### Riesgos al interpretar los resultados

Acentos, micrófonos y ambientes no están representados uniformemente. Una accuracy alta en el set limpio no garantiza desempeño con hablantes, dispositivos o entornos distintos a los del corpus, y la robustez debe verificarse explícitamente con ruido antes de confiar en el modelo.

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

- Warden (2018), *Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition*, arXiv — describe el corpus y el protocolo de evaluación de comandos hablados.
- Hershey et al. (2017), *CNN Architectures for Large-Scale Audio Classification*, ICASSP — muestra que arquitecturas convolucionales sobre espectrogramas escalan a la clasificación de audio.
- Davis & Mermelstein (1980), *Comparison of Parametric Representations for Monosyllabic Word Recognition in Continuously Spoken Sentences*, IEEE TASSP — origen de los coeficientes cepstrales en escala mel (MFCC) usados en la línea base.
- Park et al. (2019), *SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition*, Interspeech — enmascarado de bandas de tiempo y frecuencia directamente sobre el espectrograma.

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
| [🧷 Segmentación semántica con U-Net](../../advanced_labs/26_segmentation_unet/README.md) | [Las 31 rutas](../../parts/README.md) | [🖌️ WGAN-GP sobre Fashion-MNIST](../../advanced_labs/28_wgan_gp/README.md) |

**En este laboratorio:** **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🔬 [Parte 7 — Especializaciones avanzadas](../../parts/07-especializaciones-avanzadas.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/27_audio_speechcommands/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
