# Teoría — Difusión DDPM sobre Fashion-MNIST

<!-- nav-top -->
> 🧭 **Ruta 30 / 31** · 🔬 [Parte 7 — Especializaciones avanzadas](../../parts/07-especializaciones-avanzadas.md)
>
> [⬅️ 🖌️ WGAN-GP sobre Fashion-MNIST](../../advanced_labs/28_wgan_gp/theory.md) · [🏠 Índice de rutas](../../parts/README.md) · [🪞 Aprendizaje autosupervisado SimCLR ➡️](../../advanced_labs/30_self_supervised_simclr/theory.md)
>
> [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

Proceso directo de ruido, predicción de epsilon, cronograma beta y muestreo inverso.

## Idea central

Las dos rutas generativas anteriores atacan el problema de frente: una GAN aprende a producir una imagen completa de un salto, a partir de ruido, y el entrenamiento consiste en un juego adversarial delicado. La difusión hace lo contrario, y esa inversión es la idea entera: en vez de aprender la tarea difícil de una vez, la **descompone en muchos pasos fáciles**.

El razonamiento es este. Destruir una imagen es trivial: se le añade un poco de ruido gaussiano, y repitiendo el gesto mil veces queda ruido puro. Cada paso individual de esa destrucción es tan pequeño que **invertirlo también es fácil**: dada una imagen apenas ruidosa, quitarle ese poco de ruido es un problema de regresión sencillo, no un problema creativo. Y si se sabe invertir cada paso, se sabe invertir la cadena entera: se parte de ruido puro y se deshace la destrucción hasta llegar a una imagen. Generar deja de ser un salto y pasa a ser un descenso gradual.

Eso cambia la naturaleza del entrenamiento por completo. No hay dos redes compitiendo, no hay equilibrio que mantener, no hay colapso de modos: hay **una sola red y un error cuadrático medio**. Se toma una imagen real, se elige un paso al azar, se la corrompe con un ruido conocido, y se entrena la red a adivinar exactamente ese ruido. Es supervisión pura, con la etiqueta generada por uno mismo, y por eso el entrenamiento es tan estable comparado con el de la ruta 28.

El precio aparece en el otro extremo. Una GAN genera con una sola pasada por el generador; la difusión necesita recorrer la cadena entera, es decir **cientos de evaluaciones de la red por muestra**. Esa asimetría —entrenamiento estable y barato por paso, muestreo caro— es la característica que define a la familia, y es lo que este laboratorio mide de forma explícita en la latencia de muestreo.

## Fundamento matemático

Un modelo de difusión aprende a generar imágenes **invirtiendo** un proceso que las destruye. El **proceso directo** (forward) parte de una imagen real x₀ y le añade ruido gaussiano en T pasos pequeños, según un **cronograma** de varianzas βₜ ∈ (0, 1) creciente:

q(xₜ | xₜ₋₁) = 𝒩(xₜ ; √(1 − βₜ)·xₜ₋₁, βₜ·I).

Tras muchos pasos, x_T se vuelve indistinguible de ruido puro 𝒩(0, I). Una propiedad clave permite saltar directamente a cualquier paso t sin iterar: definiendo αₜ = 1 − βₜ y ᾱₜ = Π_(s=1..t) αₛ, se obtiene la forma cerrada

q(xₜ | x₀) = 𝒩(xₜ ; √ᾱₜ·x₀, (1 − ᾱₜ)·I),   equivalente a   xₜ = √ᾱₜ·x₀ + √(1 − ᾱₜ)·ε,  con ε ∼ 𝒩(0, I).

Así, para cualquier t, la imagen ruidosa es una mezcla determinista de la imagen original (peso √ᾱₜ) y ruido (peso √(1 − ᾱₜ)). A medida que t crece, ᾱₜ → 0 y domina el ruido.

Generar equivale a recorrer el **proceso inverso** p_θ(xₜ₋₁ | xₜ), partiendo de ruido y quitándolo paso a paso. El aporte central de DDPM (Ho, Jain y Abbeel) es que, en lugar de predecir directamente la media de esa distribución, la red neuronal ε_θ(xₜ, t) —una U-Net condicionada al paso t— se entrena para **predecir el ruido ε** que se añadió. El objetivo se simplifica a una regresión de error cuadrático medio sorprendentemente simple:

ℒ = 𝔼_(x₀, ε, t) [ ‖ ε − ε_θ(√ᾱₜ·x₀ + √(1 − ᾱₜ)·ε, t) ‖² ].

Es decir: se toma una imagen real, se elige un paso t al azar, se la corrompe con un ε conocido, y la red aprende a adivinar ese ε. Predecir el ruido resulta más estable y efectivo que predecir la media o la imagen limpia directamente, y conecta la difusión con la idea de *score matching* (aprender el gradiente ∇ log p(x) de la densidad de datos).

El **muestreo inverso** usa el ε predicho para dar un paso de denoising. La media del paso anterior se estima como

μ_θ(xₜ, t) = (1 / √αₜ) · ( xₜ − (βₜ / √(1 − ᾱₜ)) · ε_θ(xₜ, t) ),

y se muestrea xₜ₋₁ = μ_θ(xₜ, t) + σₜ·z con z ∼ 𝒩(0, I) (y z = 0 en el último paso). Repitiendo de t = T hasta t = 1 se transforma ruido en una muestra coherente. Esto explica el compromiso característico de la difusión: la calidad es alta, pero el muestreo es **iterativo** y por tanto costoso, pues requiere T evaluaciones de la red. Nichol y Dhariwal mejoraron el método aprendiendo también las varianzas y proponiendo cronogramas βₜ más suaves (p. ej. tipo coseno), que reparten mejor la dificultad entre pasos. La raíz teórica está en Sohl-Dickstein et al., que formularon la generación como difusión inversa inspirada en la termodinámica de no equilibrio. La línea base autoencoder generativo simple ofrece un punto de comparación de menor fidelidad.

### De dónde sale esa pérdida tan simple

La forma final del objetivo parece demasiado buena para ser el resultado de una derivación probabilística, y conviene ver por qué lo es. Como en cualquier modelo de variables latentes, no se puede maximizar log p_θ(x₀) directamente, así que se maximiza una **cota inferior variacional** (ELBO). Descomponiéndola por pasos, queda

ℒ_VLB = 𝔼_q [ D_KL( q(x_T | x₀) ‖ p(x_T) ) + Σ_(t>1) D_KL( q(x_(t−1) | xₜ, x₀) ‖ p_θ(x_(t−1) | xₜ) ) − log p_θ(x₀ | x₁) ].

El primer término no depende de θ —el proceso directo es fijo y x_T es ruido puro por construcción—, así que desaparece de la optimización. Todo el trabajo está en los términos centrales, que comparan el paso inverso aprendido con el **posterior verdadero** del proceso directo. Y ese posterior tiene forma cerrada gaussiana, lo que es la pieza que hace tratable todo el método:

q(x_(t−1) | xₜ, x₀) = 𝒩( x_(t−1) ; μ̃ₜ(xₜ, x₀), β̃ₜ·I ),

con

μ̃ₜ(xₜ, x₀) = (√ᾱ_(t−1)·βₜ / (1 − ᾱₜ))·x₀ + (√αₜ·(1 − ᾱ_(t−1)) / (1 − ᾱₜ))·xₜ,   β̃ₜ = ((1 − ᾱ_(t−1)) / (1 − ᾱₜ))·βₜ.

Nótese que este posterior está condicionado a x₀: sabiendo la imagen original, se sabe exactamente cómo desandar un paso. El modelo no dispone de x₀, y ahí está el problema que la red resuelve.

La KL entre dos gaussianas de igual varianza se reduce a la distancia entre sus medias, así que el término t-ésimo es proporcional a ‖μ̃ₜ(xₜ, x₀) − μ_θ(xₜ, t)‖². Ahora se usa la reparametrización: de xₜ = √ᾱₜ·x₀ + √(1 − ᾱₜ)·ε se despeja x₀ = (xₜ − √(1 − ᾱₜ)·ε) / √ᾱₜ, se sustituye en μ̃ₜ, y tras simplificar aparece justo la forma de μ_θ que se usa en el muestreo. La diferencia entre ambas medias se convierte entonces en la diferencia entre el ruido verdadero y el predicho, y el término queda

ℒ_t = ( βₜ² / (2·σₜ²·αₜ·(1 − ᾱₜ)) ) · ‖ ε − ε_θ(xₜ, t) ‖².

Ho, Jain y Abbeel observaron empíricamente que **descartar ese coeficiente** —fijarlo en 1 para todo t— da mejores muestras que conservarlo. Esa es la ℒ_simple que se implementa. La interpretación es que el peso original da mucha importancia a los pasos con t pequeño, donde queda poco ruido y la tarea es casi trivial; igualar los pesos obliga a la red a dedicar capacidad a los pasos intermedios, que son los que deciden la estructura global de la imagen. Es una decisión práctica que se aparta de la cota teórica y la documenta como tal.

### El cronograma βₜ y qué cambia al elegirlo

El cronograma decide cuánta información se destruye en cada paso, y por tanto cómo se reparte la dificultad. El original es **lineal**, con βₜ creciendo de 10⁻⁴ a 0,02 en T = 1000 pasos. Su defecto es que ᾱₜ cae demasiado rápido al principio: buena parte de los pasos finales operan sobre entradas ya indistinguibles de ruido y aportan poco al aprendizaje.

Nichol y Dhariwal propusieron un **cronograma coseno**, definido directamente sobre ᾱₜ en vez de sobre βₜ:

ᾱₜ = f(t) / f(0),   con   f(t) = cos²( ((t/T + s) / (1 + s)) · (π/2) ),

y un desplazamiento pequeño s ≈ 0,008 que evita que βₜ sea demasiado pequeño cerca de t = 0. La curva resultante destruye información de forma más gradual en los extremos y concentra el cambio en el centro, que es donde la red tiene algo que aprender. La relación inversa βₜ = 1 − ᾱₜ/ᾱ_(t−1) permite recuperar el cronograma de varianzas a partir de cualquier ᾱₜ elegido.

En el muestreo queda por fijar σₜ, la varianza del paso inverso. Hay dos elecciones clásicas con garantía teórica —σₜ² = βₜ y σₜ² = β̃ₜ, que son las cotas superior e inferior del intervalo razonable— y dan resultados similares. Nichol y Dhariwal muestran que **aprender** una interpolación entre ambas mejora la verosimilitud, aunque no siempre la calidad visual percibida.

### El costo del muestreo, en números

Cada muestra exige T evaluaciones de la U-Net, una por paso. Con T = 1000, generar una sola imagen cuesta mil pasadas hacia adelante; una GAN de calidad comparable cuesta una. Esa razón de tres órdenes de magnitud es la desventaja estructural de la difusión y explica toda la línea de investigación posterior sobre muestreadores acelerados, que buscan saltarse pasos sin degradar la muestra.

De ahí se sigue el compromiso que conviene tener presente al leer los resultados del laboratorio: reducir T abarata el muestreo pero engrosa cada paso de denoising, y llegado un punto la aproximación gaussiana del paso inverso deja de ser válida y las muestras se degradan. La `sampling_latency` que se reporta no es un detalle de ingeniería: es la mitad del perfil de esta familia de modelos.

## Visualización específica

Cadena de ruido, denoising, cuadrícula de muestras y costo por pasos. La cadena forward muestra cómo βₜ degrada la imagen; la secuencia de denoising ilustra la reconstrucción progresiva; la curva de costo por número de pasos T evidencia el compromiso entre calidad de muestreo y latencia.

## Riesgo de interpretación

El modelo pequeño sirve para estudio; no debe extrapolarse a generación fotográfica de alta resolución. El error de predicción de ruido (noise_mse) mide qué tan bien se estima ε, pero no equivale directamente a calidad perceptual, y los proxies de diversidad no descartan memorización de ejemplos del entrenamiento.

## 🔗 Referencias

> Las referencias apuntan a las obras; no se reproduce su contenido, la redacción es original.

- Sohl-Dickstein et al. (2015), *Deep Unsupervised Learning using Nonequilibrium Thermodynamics*, ICML — formula la generación como inversión de un proceso de difusión.
- Ho, Jain & Abbeel (2020), *Denoising Diffusion Probabilistic Models*, NeurIPS — establece el objetivo de predicción de ε y el muestreo DDPM.
- Nichol & Dhariwal (2021), *Improved Denoising Diffusion Probabilistic Models*, ICML — varianzas aprendidas y cronogramas βₜ mejorados (coseno).
- Foster — *Generative Deep Learning* (2.ª ed., O'Reilly 2023) — presentación accesible de los modelos de difusión y su implementación.

<!-- nav-bottom -->
## 🧭 Navegación del recorrido

| ⬅️ Laboratorio anterior | 🏠 Índice | Laboratorio siguiente ➡️ |
|---|:---:|---|
| [🖌️ WGAN-GP sobre Fashion-MNIST](../../advanced_labs/28_wgan_gp/README.md) | [Las 31 rutas](../../parts/README.md) | [🪞 Aprendizaje autosupervisado SimCLR](../../advanced_labs/30_self_supervised_simclr/README.md) |

**En este laboratorio:** [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🔬 [Parte 7 — Especializaciones avanzadas](../../parts/07-especializaciones-avanzadas.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/29_diffusion_ddpm/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
