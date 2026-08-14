# Teoría — Clasificación de audio con SpeechCommands

<!-- nav-top -->
> 🧭 **Ruta 28 / 31** · 🔬 [Parte 7 — Especializaciones avanzadas](../../parts/07-especializaciones-avanzadas.md)
>
> [⬅️ 🧷 Segmentación semántica con U-Net](../../advanced_labs/26_segmentation_unet/theory.md) · [🏠 Índice de rutas](../../parts/README.md) · [🖌️ WGAN-GP sobre Fashion-MNIST ➡️](../../advanced_labs/28_wgan_gp/theory.md)
>
> [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

Waveform, espectrograma log-mel, convolución 2D y robustez ante ruido.

## Idea central

Un clip de un segundo a 16 kHz son 16 000 números. Puestos en fila, una red densa necesitaría decenas de millones de pesos solo en su primera capa, y aun así no aprendería gran cosa: la información que distingue «arriba» de «abajo» no está en el valor de cada muestra individual, sino en **cómo cambia el contenido en frecuencia a lo largo del tiempo**. Dos grabaciones de la misma palabra, desplazadas unos milisegundos o dichas con otro tono, tienen formas de onda numéricamente distintas y contenido fonético idéntico.

La idea del laboratorio es cambiar de representación antes de modelar. La transformada de Fourier de corto tiempo convierte la señal unidimensional en una imagen bidimensional —tiempo en un eje, frecuencia en el otro, energía como intensidad— donde los rasgos que importan se vuelven **patrones locales visibles**: las bandas horizontales de los formantes vocálicos, las ráfagas anchas de las consonantes oclusivas, las transiciones diagonales entre fonemas. Una vez que el problema tiene forma de imagen, la herramienta correcta es la que ya se estudió en la ruta 03: una CNN 2D con pesos compartidos, que detecta esos patrones aparezcan donde aparezcan.

Sobre esa representación se aplican dos ajustes que no vienen de las matemáticas sino de la fisiología del oído: comprimir el eje de frecuencia según la escala mel, porque distinguimos mucho mejor entre 300 y 400 Hz que entre 8 000 y 8 100, y tomar el logaritmo de la energía, porque percibimos la intensidad de forma aproximadamente logarítmica. El resultado —el espectrograma log-mel— no es una elección arbitraria: es un preprocesamiento que descarta justo la información que el oído humano tampoco usa para reconocer palabras.

El laboratorio se cierra con la pregunta que separa un modelo de laboratorio de uno usable: qué pasa cuando el micrófono capta ruido de fondo. Un modelo que aprendió rasgos fonéticos estables degrada suavemente; uno que memorizó particularidades del set limpio se derrumba. Esa diferencia no aparece en la exactitud sobre datos limpios, y por eso se mide aparte.

## Fundamento matemático

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

## Visualización específica

Waveform, espectrograma, errores por palabra y ruido. Contrastar la onda cruda con su espectrograma log-mel muestra por qué la representación 2D facilita la clasificación; el desglose de errores por palabra revela confusiones entre comandos fonéticamente parecidos, y las pruebas con ruido cuantifican la robustez.

## Riesgo de interpretación

Acentos, micrófonos y ambientes no están representados uniformemente. Una accuracy alta en el set limpio no garantiza desempeño con hablantes, dispositivos o entornos distintos a los del corpus, y la robustez debe verificarse explícitamente con ruido antes de confiar en el modelo.

## 🔗 Referencias

> Las referencias apuntan a las obras; no se reproduce su contenido, la redacción es original.

- Warden (2018), *Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition*, arXiv — describe el corpus y el protocolo de evaluación de comandos hablados.
- Hershey et al. (2017), *CNN Architectures for Large-Scale Audio Classification*, ICASSP — muestra que arquitecturas convolucionales sobre espectrogramas escalan a la clasificación de audio.
- Davis & Mermelstein (1980), *Comparison of Parametric Representations for Monosyllabic Word Recognition in Continuously Spoken Sentences*, IEEE TASSP — origen de los coeficientes cepstrales en escala mel (MFCC) usados en la línea base.
- Park et al. (2019), *SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition*, Interspeech — enmascarado de bandas de tiempo y frecuencia directamente sobre el espectrograma.

<!-- nav-bottom -->
## 🧭 Navegación del recorrido

| ⬅️ Laboratorio anterior | 🏠 Índice | Laboratorio siguiente ➡️ |
|---|:---:|---|
| [🧷 Segmentación semántica con U-Net](../../advanced_labs/26_segmentation_unet/README.md) | [Las 31 rutas](../../parts/README.md) | [🖌️ WGAN-GP sobre Fashion-MNIST](../../advanced_labs/28_wgan_gp/README.md) |

**En este laboratorio:** [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🔬 [Parte 7 — Especializaciones avanzadas](../../parts/07-especializaciones-avanzadas.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/27_audio_speechcommands/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
