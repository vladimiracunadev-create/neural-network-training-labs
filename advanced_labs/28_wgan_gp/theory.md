# Teoría — WGAN-GP sobre Fashion-MNIST

<!-- nav-top -->
> 🧭 **Ruta 29 / 31** · [⬅️ 🎙️ Clasificación de audio con SpeechCommands](../../advanced_labs/27_audio_speechcommands/theory.md) · [🏠 Índice](../../README.md#laboratorios) · [🌫️ Difusión DDPM sobre Fashion-MNIST ➡️](../../advanced_labs/29_diffusion_ddpm/theory.md)
>
> [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

Distancia Wasserstein, crítico sin sigmoide y restricción Lipschitz mediante gradient penalty.

## Fundamento matemático

Una GAN enfrenta dos redes: un **generador** G que transforma ruido z ∼ p_z (típicamente z ∼ 𝒩(0, I)) en muestras G(z), y un discriminador/crítico que juzga qué tan reales parecen. La GAN clásica minimiza una divergencia de Jensen-Shannon entre la distribución real p_r y la generada p_g. El problema es que cuando ambas distribuciones tienen soportes casi disjuntos —lo habitual al inicio— la JS es constante y su gradiente se anula, provocando **entrenamiento inestable** y colapso de modos. La **WGAN** cambia el objetivo por la **distancia de Wasserstein-1** (o "earth mover"):

W(p_r, p_g) = inf_(γ ∈ Π(p_r, p_g)) 𝔼_((x,y)∼γ) ‖x − y‖,

que mide el "coste mínimo de transporte" para convertir una distribución en la otra. A diferencia de la JS, W varía suavemente aunque los soportes no se solapen, dando un gradiente útil en todo momento.

Calcular ese ínfimo es intratable, así que se usa la **dualidad de Kantorovich-Rubinstein**:

W(p_r, p_g) = sup_(‖f‖_L ≤ 1) [ 𝔼_(x∼p_r) f(x) − 𝔼_(x̃∼p_g) f(x̃) ],

donde el supremo se toma sobre todas las funciones **1-Lipschitz** f. Aquí f es el **crítico** (no un clasificador): a diferencia del discriminador clásico, no lleva sigmoide final ni produce una probabilidad, sino un valor escalar real. El generador se entrena para maximizar 𝔼 f(G(z)), es decir, para que sus muestras reciban puntuaciones altas del crítico. El nombre "crítico" en vez de "discriminador" subraya que estima una distancia, no clasifica real/falso.

La condición 1-Lipschitz (‖∇f‖ ≤ 1 en todo punto) es la clave y también la dificultad. La WGAN original la imponía recortando los pesos (*weight clipping*), lo que degrada la capacidad de la red y provoca gradientes que explotan o desaparecen. La mejora **WGAN-GP** (Gulrajani et al.) la sustituye por una **penalización de gradiente**: como una función 1-Lipschitz diferenciable tiene norma de gradiente ≤ 1, se penaliza que se aleje de 1 en puntos interpolados x̂ = ε·x + (1−ε)·x̃, con ε ∼ U[0,1] entre una muestra real x y una generada x̃. La pérdida del crítico queda

ℒ_crítico = 𝔼_(x̃∼p_g) f(x̃) − 𝔼_(x∼p_r) f(x) + λ · 𝔼_(x̂) [ (‖∇_x̂ f(x̂)‖₂ − 1)² ],

donde λ (habitualmente 10) pesa el término de penalización. Este regularizador impone la restricción de forma suave y local, estabilizando el entrenamiento y permitiendo arquitecturas más profundas. En la práctica se actualiza el crítico varias veces por cada paso del generador, para que su estimación de W sea buena antes de mover G. La línea base DCGAN convolucional (GAN clásica con sigmoide y pérdida JS) sirve de contraste directo para apreciar la ganancia en estabilidad y cobertura de modos.

## Visualización específica

Muestras por época, interpolación latente, pérdidas y cobertura de clases mediante clasificador externo. La curva de pérdida del crítico aproxima la distancia de Wasserstein y, a diferencia de la GAN clásica, correlaciona con la calidad visual; la interpolación en z revela si el espacio latente es suave; el clasificador externo estima si se cubren todas las clases o hay colapso de modos.

## Riesgo de interpretación

Las métricas generativas aproximadas no sustituyen evaluación humana ni validación del uso previsto. El estimador de Wasserstein y los proxies de diversidad son indicadores, no garantías de fidelidad; muestras nítidas pueden coexistir con clases faltantes, y clases cubiertas con artefactos sutiles.

## 🔗 Referencias

> Las referencias apuntan a las obras; no se reproduce su contenido, la redacción es original.

- Arjovsky, Chintala & Bottou (2017), *Wasserstein GAN*, ICML — reemplaza la divergencia JS por la distancia de Wasserstein y define el crítico Lipschitz.
- Gulrajani et al. (2017), *Improved Training of Wasserstein GANs*, NeurIPS — introduce la penalización de gradiente (WGAN-GP) en lugar del recorte de pesos.
- Foster — *Generative Deep Learning* (2.ª ed., O'Reilly 2023) — exposición práctica de GANs, WGAN y estabilización del entrenamiento.

<!-- nav-bottom -->
## 🧭 Navegación del recorrido

| ⬅️ Laboratorio anterior | 🏠 Índice | Laboratorio siguiente ➡️ |
|---|:---:|---|
| [🎙️ Clasificación de audio con SpeechCommands](../../advanced_labs/27_audio_speechcommands/README.md) | [Las 31 rutas](../../README.md#laboratorios) | [🌫️ Difusión DDPM sobre Fashion-MNIST](../../advanced_labs/29_diffusion_ddpm/README.md) |

**En este laboratorio:** [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

[🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/28_wgan_gp/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
