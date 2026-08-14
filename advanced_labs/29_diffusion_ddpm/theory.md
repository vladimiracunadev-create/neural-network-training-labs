# Teoría — Difusión DDPM sobre Fashion-MNIST

<!-- nav-top -->
> 🧭 **Ruta 30 / 31** · [⬅️ 🖌️ WGAN-GP sobre Fashion-MNIST](../../advanced_labs/28_wgan_gp/theory.md) · [🏠 Índice](../../README.md#laboratorios) · [🪞 Aprendizaje autosupervisado SimCLR ➡️](../../advanced_labs/30_self_supervised_simclr/theory.md)
>
> [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

Proceso directo de ruido, predicción de epsilon, cronograma beta y muestreo inverso.

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
| [🖌️ WGAN-GP sobre Fashion-MNIST](../../advanced_labs/28_wgan_gp/README.md) | [Las 31 rutas](../../README.md#laboratorios) | [🪞 Aprendizaje autosupervisado SimCLR](../../advanced_labs/30_self_supervised_simclr/README.md) |

**En este laboratorio:** [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

[🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/29_diffusion_ddpm/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
