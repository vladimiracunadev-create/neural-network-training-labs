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

### De qué trata

En la ruta 08 se entrenó una GAN clásica y se vio el problema de cerca: el entrenamiento oscila, a veces se derrumba, y la pérdida no dice nada sobre la calidad de las muestras. Puede bajar mientras las imágenes empeoran. Este laboratorio explica **por qué** ocurre eso y qué cambio matemático lo arregla.

La raíz del fallo está en qué se mide. La GAN original minimiza, implícitamente, una divergencia de Jensen-Shannon entre la distribución real y la generada. Esa divergencia tiene una propiedad fatal al comienzo del entrenamiento: cuando las dos distribuciones apenas se solapan —y no se solapan, porque las imágenes reales viven en una variedad de dimensión muy baja dentro del espacio de píxeles— la JS es prácticamente constante. Una función constante tiene gradiente cero. El generador no recibe señal sobre *hacia dónde* moverse; solo sabe que está mal, no en qué dirección corregir.

La distancia de Wasserstein resuelve exactamente eso. Mide el coste de transportar masa de una distribución a la otra, así que sigue variando de forma suave aunque los soportes sean disjuntos: siempre indica una dirección. El precio es que calcularla parece imposible —implica un ínfimo sobre todos los planes de transporte—, y aquí entra el segundo ingrediente: una dualidad clásica la convierte en un supremo sobre funciones 1-Lipschitz, algo que una red neuronal sí puede aproximar. Esa red deja de ser un clasificador y pasa a ser un **crítico**: no dice «real o falso», estima una distancia.

Queda un problema práctico: obligar a una red a ser 1-Lipschitz. La WGAN original lo hacía recortando los pesos a un intervalo, un remedio brusco que limita la capacidad del crítico. WGAN-GP lo sustituye por una penalización sobre la norma del gradiente, que impone la restricción donde importa y deja la red libre en lo demás. El laboratorio contrasta esta variante contra la DCGAN de la ruta 08 y, sobre todo, verifica algo que la GAN clásica no ofrece: que la pérdida del crítico **correlacione con la calidad visual**, es decir, que por fin haya un número al que valga la pena mirar.

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

### Por qué la JS se queda sin gradiente y la Wasserstein no

El argumento se ve con un ejemplo mínimo, el de Arjovsky. Sea p_r la distribución uniforme sobre el segmento {(0, y) : y ∈ [0,1]} y p_g la misma trasladada, sobre {(θ, y)}. Las dos son distribuciones sobre el plano cuyos soportes son segmentos paralelos separados por θ. Entonces:

JS(p_r, p_g) = log 2 si θ ≠ 0, y 0 si θ = 0;   mientras que   W(p_r, p_g) = |θ|.

La divergencia JS es una función escalonada: vale log 2 para *cualquier* separación, sea θ = 10 o θ = 0,001, y salta a 0 solo en el punto exacto de coincidencia. Su derivada respecto de θ es cero en todas partes donde está definida, así que un descenso de gradiente sobre θ no se mueve. La distancia de Wasserstein, en cambio, es |θ|: su derivada es ±1, y siempre apunta hacia la solución. Este ejemplo de juguete es el modelo exacto de lo que ocurre con imágenes, donde la variedad de datos reales tiene medida nula en el espacio de píxeles y el solape inicial con la distribución generada es efectivamente vacío.

De ahí se sigue una consecuencia contraintuitiva pero central: en una GAN clásica, **cuanto mejor es el discriminador, peor es el gradiente** que recibe el generador, porque un discriminador perfecto satura y aplana la pérdida. En WGAN ocurre lo contrario. Como el objetivo del crítico es aproximar un supremo, cuanto más se acerca a la función 1-Lipschitz óptima, mejor es la estimación de W y más fiel es la dirección que transmite. Por eso se entrena el crítico **n_critic veces** —típicamente cinco— por cada paso del generador: no es un truco de estabilidad, es una condición para que la cantidad que se está estimando signifique algo.

### La penalización de gradiente, término a término

La restricción ‖f‖_L ≤ 1 significa |f(x) − f(y)| ≤ ‖x − y‖ para todo par de puntos. Para una función diferenciable eso equivale a ‖∇f(x)‖ ≤ 1 en todo el dominio, y comprobarlo en todo el espacio es imposible. WGAN-GP hace dos aproximaciones deliberadas.

La primera es **dónde** se comprueba. En lugar de todo el espacio, se evalúa sobre la recta que une cada muestra real con una generada, x̂ = ε·x + (1 − ε)·x̃ con ε ∼ U[0,1]. La justificación es que el óptimo teórico tiene norma de gradiente exactamente 1 casi en todas partes a lo largo de esas líneas de transporte: es la región que determina el valor de W, y las demás no aportan.

La segunda es **penalizar la desviación de 1, no el exceso sobre 1**. El término es (‖∇_x̂ f(x̂)‖₂ − 1)², una penalización **bilateral**: castiga tanto que la norma suba de 1 como que baje. Podría parecer un error —la restricción solo exige ≤ 1— pero es intencional: el crítico óptimo satura la cota, así que empujar hacia 1 acelera la convergencia hacia esa solución en vez de permitir gradientes pequeños que darían señal débil al generador. El coeficiente λ = 10 es el valor que los autores encontraron robusto en arquitecturas y datasets muy distintos.

Un detalle de implementación con consecuencias: calcular esta penalización exige derivar respecto de la **entrada**, no de los pesos, y luego derivar ese resultado respecto de los pesos para actualizarlos. Es una derivada de segundo orden, y por eso el grafo de cómputo debe construirse con `create_graph=True`. Ahí se va buena parte del costo adicional de WGAN-GP frente a una GAN normal.

### Por qué el crítico no lleva normalización por lotes

La penalización se define **por muestra**: cada x̂ debe cumplir su restricción de gradiente independientemente. La normalización por lotes rompe justo eso, porque hace que la salida de cada ejemplo dependa de los demás ejemplos del lote —resta la media y divide por la desviación calculadas sobre el lote entero—. Con BatchNorm, ∇_x̂ f(x̂) deja de ser el gradiente de una función de x̂ sola, y la penalización pierde su sentido matemático. Por eso el crítico usa normalización de capa o de instancia, o ninguna, mientras que el generador sí puede usar BatchNorm sin problema.

También cambia el optimizador. Con Adam se recomiendan momentos bajos, β₁ = 0 o 0,5 y β₂ = 0,9, porque el objetivo del crítico se mueve constantemente —el generador cambia bajo sus pies— y un momento alto arrastra información de un objetivo que ya no existe.

### Qué mide realmente la curva de pérdida

La diferencia práctica más útil de WGAN-GP es que **la pérdida del crítico es interpretable**. Su parte no penalizada,

Ŵ = 𝔼_(x∼p_r) f(x) − 𝔼_(x̃∼p_g) f(x̃),

es una estimación de la distancia de Wasserstein (salvo una constante multiplicativa, ya que f solo es aproximadamente 1-Lipschitz). En una GAN clásica la pérdida del discriminador es un juego de suma casi nula que oscila alrededor de log 2 y no informa de nada; aquí Ŵ **decrece de forma monótona** cuando el entrenamiento va bien, y correlaciona con la calidad visual de las muestras. Eso es lo que permite, por primera vez en la familia GAN, usar la curva como criterio de selección de checkpoint en vez de mirar imágenes a ojo.

La advertencia se mantiene: Ŵ es una estimación con la constante de Lipschitz sin fijar, así que su **valor absoluto no es comparable** entre ejecuciones con distinta arquitectura o distinto λ. Lo interpretable es su tendencia dentro de una misma corrida.

### Qué conviene graficar

Muestras por época, interpolación latente, pérdidas y cobertura de clases mediante clasificador externo. La curva de pérdida del crítico aproxima la distancia de Wasserstein y, a diferencia de la GAN clásica, correlaciona con la calidad visual; la interpolación en z revela si el espacio latente es suave; el clasificador externo estima si se cubren todas las clases o hay colapso de modos.

### Qué se mide y con qué se decide

El laboratorio reporta `wasserstein_estimate`, `energy_distance_proxy`, `diversity`, `training_stability`. De todas ellas, la que **decide** qué modelo se conserva es `wasserstein_estimate`, y se mide siempre sobre `validation`: es la única forma de que `test` siga siendo una estimación honesta de lo que pasará con datos nuevos.

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
jupyter lab advanced_labs/28_wgan_gp/notebook.ipynb
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
- Fuente del dataset: https://github.com/zalandoresearch/fashion-mnist — **Fashion-MNIST** (Zalando Research (Zalando SE), MIT License); procedencia, versión y SHA-256 en el registro de fuentes, entrada `fashion-mnist` — esta clase la usa para entrenar un crítico Lipschitz con penalización de gradiente y comprobar que su pérdida correlaciona con la calidad visual.

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
| [🎙️ Clasificación de audio con SpeechCommands](../../advanced_labs/27_audio_speechcommands/README.md) | [Las 31 rutas](../../parts/README.md) | [🌫️ Difusión DDPM sobre Fashion-MNIST](../../advanced_labs/29_diffusion_ddpm/README.md) |

**En este laboratorio:** **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🔬 [Parte 7 — Especializaciones avanzadas](../../parts/07-especializaciones-avanzadas.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/28_wgan_gp/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
