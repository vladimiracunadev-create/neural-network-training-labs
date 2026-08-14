# Teoría — Clasificación de audio con SpeechCommands

<!-- nav-top -->
> 🧭 **Ruta 28 / 31** · 🔬 [Parte 7 — Especializaciones avanzadas](../../parts/07-especializaciones-avanzadas.md)
>
> [⬅️ 🧷 Segmentación semántica con U-Net](../../advanced_labs/26_segmentation_unet/theory.md) · [🏠 Índice de rutas](../../parts/README.md) · [🖌️ WGAN-GP sobre Fashion-MNIST ➡️](../../advanced_labs/28_wgan_gp/theory.md)
>
> [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

Waveform, espectrograma log-mel, convolución 2D y robustez ante ruido.

## Fundamento matemático

El audio crudo es una **forma de onda** (waveform): una señal x[t] de amplitud muestreada en el tiempo, aquí a 16 kHz durante un segundo (≈16 000 muestras por clip). Clasificar directamente sobre esa secuencia larga y unidimensional es difícil, porque la información fonética relevante vive en el *contenido frecuencial* que varía a lo largo del tiempo. Por eso se transforma la señal a una representación tiempo-frecuencia mediante la **transformada de Fourier de corto tiempo** (STFT): se divide la onda en ventanas solapadas y se calcula el espectro de cada una,

X(m, k) = Σ_(n) x[n] · w[n − m] · e^(−j·2π·k·n / N),

donde w es una ventana (p. ej. Hann), m indexa el tiempo y k la frecuencia. El **espectrograma** es la magnitud al cuadrado |X(m, k)|²: una "imagen" 2D donde un eje es tiempo y el otro frecuencia, y la intensidad es la energía.

Sobre ese espectrograma se aplican dos transformaciones inspiradas en la percepción auditiva humana. Primero, la **escala mel** comprime el eje de frecuencia con un banco de filtros triangulares espaciados según mel(f) = 2595 · log₁₀(1 + f/700), que da más resolución a las frecuencias bajas —donde el oído distingue mejor— y agrupa las altas. Segundo, se toma el **logaritmo** de la energía, log(mel-energía + ε), imitando que percibimos la intensidad de forma aproximadamente logarítmica y comprimiendo el enorme rango dinámico; ε > 0 evita log(0). El resultado es el **espectrograma log-mel**, la entrada del modelo. Los **MFCC** (coeficientes cepstrales en frecuencias mel) van un paso más allá aplicando una transformada coseno discreta que decorrelaciona los canales mel; se usan en la línea base MFCC + regresión logística.

Como el log-mel es una imagen 2D, el modelo natural es una **CNN 2D**. Cada capa convolucional desliza filtros aprendidos K sobre la entrada, (X * K)(i, j) = Σ_(a,b) X(i+a, j+b) · K(a, b), detectando patrones locales tiempo-frecuencia (formantes, transiciones, ráfagas de energía) con pesos compartidos; el apilamiento con submuestreo construye representaciones cada vez más abstractas hasta una capa densa con softmax que produce p(clase | audio). El entrenamiento minimiza la entropía cruzada ℒ = −Σ_c y_c · log ŷ_c, y esta idea de tratar el audio como imagen espectral es la que Hershey et al. escalaron a clasificación de audio a gran escala.

La **robustez ante ruido** es central: en uso real el micrófono capta fondo, reverberación y solapamientos. Se evalúa perturbando la entrada, por ejemplo x̃ = x + n con ruido n de una relación señal-ruido dada (SNR = 10·log₁₀(P_señal / P_ruido) dB), y midiendo cuánto cae la accuracy. Un modelo que aprende rasgos fonéticos estables degrada poco; uno que memoriza artefactos del set limpio colapsa. Esto también motiva aumentaciones (desplazamiento temporal, ruido, enmascarado de bandas) durante el entrenamiento.

## Visualización específica

Waveform, espectrograma, errores por palabra y ruido. Contrastar la onda cruda con su espectrograma log-mel muestra por qué la representación 2D facilita la clasificación; el desglose de errores por palabra revela confusiones entre comandos fonéticamente parecidos, y las pruebas con ruido cuantifican la robustez.

## Riesgo de interpretación

Acentos, micrófonos y ambientes no están representados uniformemente. Una accuracy alta en el set limpio no garantiza desempeño con hablantes, dispositivos o entornos distintos a los del corpus, y la robustez debe verificarse explícitamente con ruido antes de confiar en el modelo.

## 🔗 Referencias

> Las referencias apuntan a las obras; no se reproduce su contenido, la redacción es original.

- Warden (2018), *Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition*, arXiv — describe el corpus y el protocolo de evaluación de comandos hablados.
- Hershey et al. (2017), *CNN Architectures for Large-Scale Audio Classification*, ICASSP — muestra que arquitecturas convolucionales sobre espectrogramas escalan a la clasificación de audio.

<!-- nav-bottom -->
## 🧭 Navegación del recorrido

| ⬅️ Laboratorio anterior | 🏠 Índice | Laboratorio siguiente ➡️ |
|---|:---:|---|
| [🧷 Segmentación semántica con U-Net](../../advanced_labs/26_segmentation_unet/README.md) | [Las 31 rutas](../../parts/README.md) | [🖌️ WGAN-GP sobre Fashion-MNIST](../../advanced_labs/28_wgan_gp/README.md) |

**En este laboratorio:** [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🔬 [Parte 7 — Especializaciones avanzadas](../../parts/07-especializaciones-avanzadas.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/27_audio_speechcommands/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
