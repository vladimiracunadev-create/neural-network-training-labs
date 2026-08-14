# WGAN-GP sobre Fashion-MNIST

<!-- nav-top -->
> 🧭 **Ruta 29 / 31** · 🔬 [Parte 7 — Especializaciones avanzadas](../../parts/07-especializaciones-avanzadas.md)
>
> [⬅️ 🎙️ Clasificación de audio con SpeechCommands](../../advanced_labs/27_audio_speechcommands/README.md) · [🏠 Índice de rutas](../../parts/README.md) · [🌫️ Difusión DDPM sobre Fashion-MNIST ➡️](../../advanced_labs/29_diffusion_ddpm/README.md)
>
> **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

## 🎯 Qué vas a hacer aquí

Estudiar estabilidad generativa y gradient penalty con imágenes reales.

Es la **ruta 29 de 31** del recorrido y pertenece a 🔬 la parte 7, *Especializaciones avanzadas*. Llegas desde **Clasificación de audio con SpeechCommands** y lo que hagas aquí lo da por supuesto **Difusión DDPM sobre Fashion-MNIST**.

Trabajarás con el dataset **`fashion_mnist`** (Torchvision / Zalando Research, licencia: MIT para código; consultar dataset), y tendrás que superar la línea base **DCGAN convolucional**, decidiendo con la métrica `wasserstein_estimate` medida sobre `validation`. Nivel avanzado.

**Qué recibe el modelo como entrada:** imágenes Fashion-MNIST normalizadas.

**Lo que conviene traer resuelto de las rutas anteriores:** GAN, CNN, optimización adversarial.

**Al terminar deberías ser capaz de:**

- Estudiar estabilidad generativa y gradient penalty con imágenes reales.
- Interpretar wasserstein_estimate, energy_distance_proxy
- Aplicar sellado de test y reproducibilidad

## 🧠 La teoría de este laboratorio

Esta sección es la explicación completa del tema. No hace falta abrir otro archivo para entender lo que viene después: aquí está la idea, la matemática que la sostiene y sus límites. (El mismo texto vive en `theory.md`, que es la fuente desde la que se genera esta guía, junto con la bibliografía del final.)

### La matemática, paso a paso

Una GAN enfrenta dos redes: un **generador** G que transforma ruido z ∼ p_z (típicamente z ∼ 𝒩(0, I)) en muestras G(z), y un discriminador/crítico que juzga qué tan reales parecen. La GAN clásica minimiza una divergencia de Jensen-Shannon entre la distribución real p_r y la generada p_g. El problema es que cuando ambas distribuciones tienen soportes casi disjuntos —lo habitual al inicio— la JS es constante y su gradiente se anula, provocando **entrenamiento inestable** y colapso de modos. La **WGAN** cambia el objetivo por la **distancia de Wasserstein-1** (o "earth mover"):

W(p_r, p_g) = inf_(γ ∈ Π(p_r, p_g)) 𝔼_((x,y)∼γ) ‖x − y‖,

que mide el "coste mínimo de transporte" para convertir una distribución en la otra. A diferencia de la JS, W varía suavemente aunque los soportes no se solapen, dando un gradiente útil en todo momento.

Calcular ese ínfimo es intratable, así que se usa la **dualidad de Kantorovich-Rubinstein**:

W(p_r, p_g) = sup_(‖f‖_L ≤ 1) [ 𝔼_(x∼p_r) f(x) − 𝔼_(x̃∼p_g) f(x̃) ],

donde el supremo se toma sobre todas las funciones **1-Lipschitz** f. Aquí f es el **crítico** (no un clasificador): a diferencia del discriminador clásico, no lleva sigmoide final ni produce una probabilidad, sino un valor escalar real. El generador se entrena para maximizar 𝔼 f(G(z)), es decir, para que sus muestras reciban puntuaciones altas del crítico. El nombre "crítico" en vez de "discriminador" subraya que estima una distancia, no clasifica real/falso.

La condición 1-Lipschitz (‖∇f‖ ≤ 1 en todo punto) es la clave y también la dificultad. La WGAN original la imponía recortando los pesos (*weight clipping*), lo que degrada la capacidad de la red y provoca gradientes que explotan o desaparecen. La mejora **WGAN-GP** (Gulrajani et al.) la sustituye por una **penalización de gradiente**: como una función 1-Lipschitz diferenciable tiene norma de gradiente ≤ 1, se penaliza que se aleje de 1 en puntos interpolados x̂ = ε·x + (1−ε)·x̃, con ε ∼ U[0,1] entre una muestra real x y una generada x̃. La pérdida del crítico queda

ℒ_crítico = 𝔼_(x̃∼p_g) f(x̃) − 𝔼_(x∼p_r) f(x) + λ · 𝔼_(x̂) [ (‖∇_x̂ f(x̂)‖₂ − 1)² ],

donde λ (habitualmente 10) pesa el término de penalización. Este regularizador impone la restricción de forma suave y local, estabilizando el entrenamiento y permitiendo arquitecturas más profundas. En la práctica se actualiza el crítico varias veces por cada paso del generador, para que su estimación de W sea buena antes de mover G. La línea base DCGAN convolucional (GAN clásica con sigmoide y pérdida JS) sirve de contraste directo para apreciar la ganancia en estabilidad y cobertura de modos.

### Qué conviene graficar

Muestras por época, interpolación latente, pérdidas y cobertura de clases mediante clasificador externo. La curva de pérdida del crítico aproxima la distancia de Wasserstein y, a diferencia de la GAN clásica, correlaciona con la calidad visual; la interpolación en z revela si el espacio latente es suave; el clasificador externo estima si se cubren todas las clases o hay colapso de modos.

### Qué se mide y con qué se decide

El laboratorio reporta `wasserstein_estimate`, `energy_distance_proxy`, `diversity`, `training_stability`. De todas ellas, la que **decide** qué modelo se conserva es `wasserstein_estimate`, y se mide siempre sobre `validation`: es la única forma de que `test` siga siendo una estimación honesta de lo que pasará con datos nuevos.

## 🖥️ Los comandos, explicados

Todo el laboratorio se maneja con una sola herramienta de terminal, `neural-labs`, que se instala junto con el paquete (`pip install -e ".[dev,notebooks]"`). Cada subcomando hace **una** cosa del protocolo, y por eso se pueden ejecutar por separado: preparar datos, auditar la partición, entrenar, repetir con varias semillas.

La forma general es siempre la misma:

```bash
neural-labs <subcomando> --track <identificador> [opciones]
```

| Opción | Valor por defecto | Valores | Qué hace y cuándo cambiarla |
|---|---|---|---|
| `--track` | `28_wgan_gp` | obligatorio | Qué especialización se entrena. Solo acepta los seis identificadores existentes. |
| `--quick` | desactivado | — | Reduce datos y épocas para comprobar que la ruta corre de extremo a extremo. |
| `--split-seed N` | `42` | entero | Semilla que decide **qué ejemplo cae en qué partición**. Se mantiene fija al comparar. |
| `--training-seed N` | `42` | entero | Semilla de la inicialización de pesos y del barajado. Es la que se varía para medir dispersión. |
| `--device` | `auto` | `auto` · `cpu` · `cuda` · `mps` | Dónde entrenar. `auto` elige GPU si la hay. |
| `--output-dir` | `runs-advanced` | ruta | Dónde se escribe el directorio de la ejecución. |

### Lo mismo desde Python

```python
from neural_labs.advanced.training import train_advanced

resultado = train_advanced(
    "28_wgan_gp",
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

**Qué ocurre.** Leer [`theory.md`](theory.md), que desarrolla Distancia Wasserstein, crítico sin sigmoide y restricción Lipschitz mediante gradient penalty. y cita las obras y papers de los que procede.

**Por qué.** Estas rutas usan arquitecturas donde un error de comprensión no se manifiesta como un fallo, sino como un número plausible pero equivocado.

**Cómo sabes que salió bien.** Puedes explicar qué mide `wasserstein_estimate` y por qué es la métrica de selección aquí.

### Paso 2 — Ejecutar la versión rápida

**Qué ocurre.** Descarga el dataset y los pesos preentrenados desde su proveedor, entrena una versión reducida y escribe la ejecución en `runs-advanced/`.

**Por qué.** Antes de gastar horas de cómputo conviene comprobar que la descarga, el entorno y la ruta completa funcionan de extremo a extremo.

```bash
neural-labs train-advanced --track 28_wgan_gp --quick
```

**Cómo sabes que salió bien.** Termina sin error y deja `metrics.json`, `history.json` y `best_model.pt` en el directorio de la ejecución.

### Paso 3 — Entrenar en serio y seleccionar con `validation`

**Qué ocurre.** Se entrena el modelo completo conservando el checkpoint con el mejor valor de `wasserstein_estimate` en validación, y se sella el experimento antes de evaluar `test`.

**Por qué.** Igual que en las rutas centrales: `validation` decide, `test` solo confirma, y el sello deja por escrito qué se había decidido antes de mirar.

```bash
neural-labs train-advanced --track 28_wgan_gp --split-seed 42 --training-seed 43
```

**Cómo sabes que salió bien.** Existe `experiment.lock.json` y `metrics.json` incluye tanto el valor de validación como el de test.

### Paso 4 — Repetir con otra semilla de entrenamiento

**Qué ocurre.** Se repite el entrenamiento con la misma partición y distinta semilla de entrenamiento.

**Por qué.** Estas arquitecturas —adversariales, contrastivas, de difusión— son especialmente sensibles a la inicialización: una sola ejecución no permite distinguir una mejora de una casualidad.

```bash
neural-labs train-advanced --track 28_wgan_gp --split-seed 42 --training-seed 44
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
- **Límite declarado de este dataset.** Las métricas generativas aproximadas no sustituyen evaluación humana ni validación del uso previsto.

### Riesgos al interpretar los resultados

Las métricas generativas aproximadas no sustituyen evaluación humana ni validación del uso previsto. El estimador de Wasserstein y los proxies de diversidad son indicadores, no garantías de fidelidad; muestras nítidas pueden coexistir con clases faltantes, y clases cubiertas con artefactos sutiles.

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

- Arjovsky, Chintala & Bottou (2017), *Wasserstein GAN*, ICML — reemplaza la divergencia JS por la distancia de Wasserstein y define el crítico Lipschitz.
- Gulrajani et al. (2017), *Improved Training of Wasserstein GANs*, NeurIPS — introduce la penalización de gradiente (WGAN-GP) en lugar del recorte de pesos.
- Foster — *Generative Deep Learning* (2.ª ed., O'Reilly 2023) — exposición práctica de GANs, WGAN y estabilización del entrenamiento.

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
| [🎙️ Clasificación de audio con SpeechCommands](../../advanced_labs/27_audio_speechcommands/README.md) | [Las 31 rutas](../../parts/README.md) | [🌫️ Difusión DDPM sobre Fashion-MNIST](../../advanced_labs/29_diffusion_ddpm/README.md) |

**En este laboratorio:** **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🔬 [Parte 7 — Especializaciones avanzadas](../../parts/07-especializaciones-avanzadas.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/28_wgan_gp/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
