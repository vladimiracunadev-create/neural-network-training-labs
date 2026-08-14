# Difusión DDPM sobre Fashion-MNIST

<!-- nav-top -->
> 🧭 **Ruta 30 / 31** · 🔬 [Parte 7 — Especializaciones avanzadas](../../parts/07-especializaciones-avanzadas.md)
>
> [⬅️ 🖌️ WGAN-GP sobre Fashion-MNIST](../../advanced_labs/28_wgan_gp/README.md) · [🏠 Índice de rutas](../../parts/README.md) · [🪞 Aprendizaje autosupervisado SimCLR ➡️](../../advanced_labs/30_self_supervised_simclr/README.md)
>
> **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

## 🎯 Qué vas a hacer aquí

Aprender predicción de ruido y muestreo iterativo sobre imágenes reales.

Es la **ruta 30 de 31** del recorrido y pertenece a 🔬 la parte 7, *Especializaciones avanzadas*. Llegas desde **WGAN-GP sobre Fashion-MNIST** y lo que hagas aquí lo da por supuesto **Aprendizaje autosupervisado SimCLR**.

Trabajarás con el dataset **`fashion_mnist`** (Torchvision / Zalando Research, licencia: MIT para código; consultar dataset), y tendrás que superar la línea base **Autoencoder generativo simple**, decidiendo con la métrica `noise_mse` medida sobre `validation`. Nivel avanzado.

**Qué recibe el modelo como entrada:** imágenes Fashion-MNIST normalizadas.

**Lo que conviene traer resuelto de las rutas anteriores:** CNN, probabilidad, modelos generativos.

**Al terminar deberías ser capaz de:**

- Aprender predicción de ruido y muestreo iterativo sobre imágenes reales.
- Interpretar noise_mse, sample_diversity
- Aplicar sellado de test y reproducibilidad

## 🧠 La teoría de este laboratorio

Esta sección es la explicación completa del tema. No hace falta abrir otro archivo para entender lo que viene después: aquí está la idea, la matemática que la sostiene y sus límites. (El mismo texto vive en `theory.md`, que es la fuente desde la que se genera esta guía, junto con la bibliografía del final.)

### La matemática, paso a paso

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

### Qué conviene graficar

Cadena de ruido, denoising, cuadrícula de muestras y costo por pasos. La cadena forward muestra cómo βₜ degrada la imagen; la secuencia de denoising ilustra la reconstrucción progresiva; la curva de costo por número de pasos T evidencia el compromiso entre calidad de muestreo y latencia.

### Qué se mide y con qué se decide

El laboratorio reporta `noise_mse`, `sample_diversity`, `sampling_latency`, `reconstruction_proxy`. De todas ellas, la que **decide** qué modelo se conserva es `noise_mse`, y se mide siempre sobre `validation`: es la única forma de que `test` siga siendo una estimación honesta de lo que pasará con datos nuevos.

## 🖥️ Los comandos, explicados

Todo el laboratorio se maneja con una sola herramienta de terminal, `neural-labs`, que se instala junto con el paquete (`pip install -e ".[dev,notebooks]"`). Cada subcomando hace **una** cosa del protocolo, y por eso se pueden ejecutar por separado: preparar datos, auditar la partición, entrenar, repetir con varias semillas.

La forma general es siempre la misma:

```bash
neural-labs <subcomando> --track <identificador> [opciones]
```

| Opción | Valor por defecto | Valores | Qué hace y cuándo cambiarla |
|---|---|---|---|
| `--track` | `29_diffusion_ddpm` | obligatorio | Qué especialización se entrena. Solo acepta los seis identificadores existentes. |
| `--quick` | desactivado | — | Reduce datos y épocas para comprobar que la ruta corre de extremo a extremo. |
| `--split-seed N` | `42` | entero | Semilla que decide **qué ejemplo cae en qué partición**. Se mantiene fija al comparar. |
| `--training-seed N` | `42` | entero | Semilla de la inicialización de pesos y del barajado. Es la que se varía para medir dispersión. |
| `--device` | `auto` | `auto` · `cpu` · `cuda` · `mps` | Dónde entrenar. `auto` elige GPU si la hay. |
| `--output-dir` | `runs-advanced` | ruta | Dónde se escribe el directorio de la ejecución. |

### Lo mismo desde Python

```python
from neural_labs.advanced.training import train_advanced

resultado = train_advanced(
    "29_diffusion_ddpm",
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

**Qué ocurre.** Leer [`theory.md`](theory.md), que desarrolla Proceso directo de ruido, predicción de epsilon, cronograma beta y muestreo inverso. y cita las obras y papers de los que procede.

**Por qué.** Estas rutas usan arquitecturas donde un error de comprensión no se manifiesta como un fallo, sino como un número plausible pero equivocado.

**Cómo sabes que salió bien.** Puedes explicar qué mide `noise_mse` y por qué es la métrica de selección aquí.

### Paso 2 — Ejecutar la versión rápida

**Qué ocurre.** Descarga el dataset y los pesos preentrenados desde su proveedor, entrena una versión reducida y escribe la ejecución en `runs-advanced/`.

**Por qué.** Antes de gastar horas de cómputo conviene comprobar que la descarga, el entorno y la ruta completa funcionan de extremo a extremo.

```bash
neural-labs train-advanced --track 29_diffusion_ddpm --quick
```

**Cómo sabes que salió bien.** Termina sin error y deja `metrics.json`, `history.json` y `best_model.pt` en el directorio de la ejecución.

### Paso 3 — Entrenar en serio y seleccionar con `validation`

**Qué ocurre.** Se entrena el modelo completo conservando el checkpoint con el mejor valor de `noise_mse` en validación, y se sella el experimento antes de evaluar `test`.

**Por qué.** Igual que en las rutas centrales: `validation` decide, `test` solo confirma, y el sello deja por escrito qué se había decidido antes de mirar.

```bash
neural-labs train-advanced --track 29_diffusion_ddpm --split-seed 42 --training-seed 43
```

**Cómo sabes que salió bien.** Existe `experiment.lock.json` y `metrics.json` incluye tanto el valor de validación como el de test.

### Paso 4 — Repetir con otra semilla de entrenamiento

**Qué ocurre.** Se repite el entrenamiento con la misma partición y distinta semilla de entrenamiento.

**Por qué.** Estas arquitecturas —adversariales, contrastivas, de difusión— son especialmente sensibles a la inicialización: una sola ejecución no permite distinguir una mejora de una casualidad.

```bash
neural-labs train-advanced --track 29_diffusion_ddpm --split-seed 42 --training-seed 44
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
- **Límite declarado de este dataset.** El modelo pequeño sirve para estudio; no debe extrapolarse a generación fotográfica de alta resolución.

### Riesgos al interpretar los resultados

El modelo pequeño sirve para estudio; no debe extrapolarse a generación fotográfica de alta resolución. El error de predicción de ruido (noise_mse) mide qué tan bien se estima ε, pero no equivale directamente a calidad perceptual, y los proxies de diversidad no descartan memorización de ejemplos del entrenamiento.

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

- Sohl-Dickstein et al. (2015), *Deep Unsupervised Learning using Nonequilibrium Thermodynamics*, ICML — formula la generación como inversión de un proceso de difusión.
- Ho, Jain & Abbeel (2020), *Denoising Diffusion Probabilistic Models*, NeurIPS — establece el objetivo de predicción de ε y el muestreo DDPM.
- Nichol & Dhariwal (2021), *Improved Denoising Diffusion Probabilistic Models*, ICML — varianzas aprendidas y cronogramas βₜ mejorados (coseno).
- Foster — *Generative Deep Learning* (2.ª ed., O'Reilly 2023) — presentación accesible de los modelos de difusión y su implementación.

### Cómo comprobar lo que dice esta guía

Ninguna cifra ni afirmación de esta página está escrita de memoria. Cada una se puede verificar en un archivo del repositorio:

| Lo que dice la guía | Dónde comprobarlo |
|---|---|
| Objetivo, línea base, métricas y arquitectura | `configs/advanced_tracks.yaml` |
| Fuente, licencia, procedencia y límites del dataset | `data/dataset.yaml` |
| Épocas, tamaño de lote, tasa de aprendizaje y recorte de `--quick` | `configs/baseline.yaml` y `configs/improved.yaml` |
| Nivel, prerrequisitos, resultados de aprendizaje y criterios | `lesson.yaml` |
| Opciones de los comandos y sus valores por defecto | `src/neural_labs/cli.py` |
| El orden de los pasos y los archivos que escribe cada ejecución | `src/neural_labs/advanced/training.py` |
| La teoría y su bibliografía | `theory.md` |
| La regla general del protocolo | `docs/experiment-protocol.md` |

Los datasets se descargan de su proveedor original y conservan su licencia; este repositorio no los redistribuye ni sustituye una descarga fallida por datos generados.

<!-- nav-bottom -->
## 🧭 Navegación del recorrido

| ⬅️ Laboratorio anterior | 🏠 Índice | Laboratorio siguiente ➡️ |
|---|:---:|---|
| [🖌️ WGAN-GP sobre Fashion-MNIST](../../advanced_labs/28_wgan_gp/README.md) | [Las 31 rutas](../../parts/README.md) | [🪞 Aprendizaje autosupervisado SimCLR](../../advanced_labs/30_self_supervised_simclr/README.md) |

**En este laboratorio:** **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🔬 [Parte 7 — Especializaciones avanzadas](../../parts/07-especializaciones-avanzadas.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/29_diffusion_ddpm/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
