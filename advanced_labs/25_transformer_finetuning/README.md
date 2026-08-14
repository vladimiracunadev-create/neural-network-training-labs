# Fine-tuning eficiente de transformer

<!-- nav-top -->
> 🧭 **Ruta 26 / 31** · 🔬 [Parte 7 — Especializaciones avanzadas](../../parts/07-especializaciones-avanzadas.md)
>
> [⬅️ 🏁 Proyecto final: churn de telecomunicaciones](../../labs/24_capstone_real_project/README.md) · [🏠 Índice de rutas](../../parts/README.md) · [🧷 Segmentación semántica con U-Net ➡️](../../advanced_labs/26_segmentation_unet/README.md)
>
> **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

## 🎯 Qué vas a hacer aquí

Comparar fine-tuning completo y LoRA sin tocar test durante selección.

Es la **ruta 26 de 31** del recorrido y pertenece a 🔬 la parte 7, *Especializaciones avanzadas*. Llegas desde **Proyecto final: churn de telecomunicaciones** y lo que hagas aquí lo da por supuesto **Segmentación semántica con U-Net**.

Trabajarás con el dataset **`ag_news`** (Hugging Face Datasets, licencia: Consultar ficha AG News), y tendrás que superar la línea base **TF-IDF + regresión logística**, decidiendo con la métrica `accuracy` medida sobre `validation`. Nivel avanzado.

**Qué recibe el modelo como entrada:** texto en inglés.

**Lo que conviene traer resuelto de las rutas anteriores:** PyTorch, NLP, Transformers.

**Al terminar deberías ser capaz de:**

- Comparar fine-tuning completo y LoRA sin tocar test durante selección.
- Interpretar accuracy, macro_f1
- Aplicar sellado de test y reproducibilidad

## 🧠 La teoría de este laboratorio

Esta sección es la explicación completa del tema. No hace falta abrir otro archivo para entender lo que viene después: aquí está la idea, la matemática que la sostiene y sus límites. (El mismo texto vive en `theory.md`, que es la fuente desde la que se genera esta guía, junto con la bibliografía del final.)

### La matemática, paso a paso

Un transformer procesa el texto tras convertirlo en tokens subword (WordPiece en el caso de DistilBERT). Cada token se representa por un vector de embedding al que se le suma una codificación posicional, formando una matriz X ∈ ℝ^(n×d), donde n es la longitud de secuencia y d la dimensión oculta. El corazón del modelo es la **autoatención**: a partir de X se proyectan consultas, claves y valores mediante matrices aprendidas, Q = X·W_Q, K = X·W_K, V = X·W_V, y se calcula

Attention(Q, K, V) = softmax( (Q·Kᵀ) / √d_k ) · V.

La división por √d_k evita que los productos escalares crezcan con la dimensión y saturen el softmax; la matriz softmax(Q·Kᵀ/√d_k) es la que se visualiza como "mapa de atención", pues cada fila indica cuánto pesa cada token del contexto al construir la representación de un token dado. Con varias cabezas (multi-head) el modelo atiende simultáneamente a distintos subespacios de relación.

El preentrenamiento (BERT) optimiza un objetivo de **modelado de lenguaje enmascarado**: se ocultan tokens aleatorios y la red minimiza la entropía cruzada al predecirlos, ℒ = −Σᵢ log p(xᵢ | x_contexto). Así el modelo aprende representaciones lingüísticas generales antes de ver la tarea final. DistilBERT es una versión **destilada**: un modelo "estudiante" más pequeño se entrena para imitar al "profesor" BERT, combinando la pérdida supervisada con un término de destilación sobre las distribuciones suaves del profesor (softmax con temperatura), conservando ~97% del rendimiento con ~40% menos parámetros.

En el **fine-tuning completo** se añade una capa de clasificación sobre el embedding del token especial [CLS] y se actualizan *todos* los pesos θ del modelo mediante descenso de gradiente sobre la entropía cruzada de la tarea, θ ← θ − α·∇_θ ℒ. Es potente pero costoso: hay que almacenar y actualizar decenas de millones de parámetros por tarea. La **adaptación eficiente LoRA** parte de la observación de que la actualización necesaria ΔW tiene rango efectivo bajo. En lugar de modificar la matriz preentrenada W₀ ∈ ℝ^(d×k), LoRA la congela y aprende una corrección factorizada de rango r pequeño:

W = W₀ + ΔW = W₀ + (α/r)·B·A,   con B ∈ ℝ^(d×r), A ∈ ℝ^(r×k), r ≪ min(d, k).

Solo se entrenan A y B (más el factor de escala α/r), reduciendo los parámetros entrenables en órdenes de magnitud sin añadir latencia en inferencia (las matrices pueden fusionarse en W). Esta es la esencia del *parameter-efficient transfer learning*, emparentada con los **adapters** de Houlsby et al., que insertan pequeños módulos entrenables entre capas congeladas. La línea base TF-IDF + regresión logística sirve de contraste: representa el texto por frecuencias de término ponderadas, sin capturar orden ni contexto, y ayuda a medir cuánto aporta realmente la atención preentrenada.

### Qué conviene graficar

Distribución de longitud, matriz de confusión, atención y comparación LoRA/full. Los mapas de atención revelan qué tokens influyen en la clasificación; la comparación LoRA vs. fine-tuning completo contrasta accuracy y macro_f1 frente al número de parámetros entrenables y la latencia, para juzgar el coste-beneficio de cada estrategia.

### Qué se mide y con qué se decide

El laboratorio reporta `accuracy`, `macro_f1`, `latency_ms`, `trainable_parameters`. De todas ellas, la que **decide** qué modelo se conserva es `accuracy`, y se mide siempre sobre `validation`: es la única forma de que `test` siga siendo una estimación honesta de lo que pasará con datos nuevos.

## 🖥️ Los comandos, explicados

Todo el laboratorio se maneja con una sola herramienta de terminal, `neural-labs`, que se instala junto con el paquete (`pip install -e ".[dev,notebooks]"`). Cada subcomando hace **una** cosa del protocolo, y por eso se pueden ejecutar por separado: preparar datos, auditar la partición, entrenar, repetir con varias semillas.

La forma general es siempre la misma:

```bash
neural-labs <subcomando> --track <identificador> [opciones]
```

| Opción | Valor por defecto | Valores | Qué hace y cuándo cambiarla |
|---|---|---|---|
| `--track` | `25_transformer_finetuning` | obligatorio | Qué especialización se entrena. Solo acepta los seis identificadores existentes. |
| `--quick` | desactivado | — | Reduce datos y épocas para comprobar que la ruta corre de extremo a extremo. |
| `--split-seed N` | `42` | entero | Semilla que decide **qué ejemplo cae en qué partición**. Se mantiene fija al comparar. |
| `--training-seed N` | `42` | entero | Semilla de la inicialización de pesos y del barajado. Es la que se varía para medir dispersión. |
| `--device` | `auto` | `auto` · `cpu` · `cuda` · `mps` | Dónde entrenar. `auto` elige GPU si la hay. |
| `--output-dir` | `runs-advanced` | ruta | Dónde se escribe el directorio de la ejecución. |
| `--lora` / `--no-lora` | `--no-lora` | — | Con LoRA se entrenan unas pocas matrices de bajo rango en vez de todos los pesos: el objetivo del laboratorio es comparar ambas. |

### Lo mismo desde Python

```python
from neural_labs.advanced.training import train_advanced

resultado = train_advanced(
    "25_transformer_finetuning",
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

**Qué ocurre.** Leer [`theory.md`](theory.md), que desarrolla Tokenización subword, atención preentrenada, fine-tuning completo y adaptación eficiente LoRA. y cita las obras y papers de los que procede.

**Por qué.** Estas rutas usan arquitecturas donde un error de comprensión no se manifiesta como un fallo, sino como un número plausible pero equivocado.

**Cómo sabes que salió bien.** Puedes explicar qué mide `accuracy` y por qué es la métrica de selección aquí.

### Paso 2 — Ejecutar la versión rápida

**Qué ocurre.** Descarga el dataset y los pesos preentrenados desde su proveedor, entrena una versión reducida y escribe la ejecución en `runs-advanced/`.

**Por qué.** Antes de gastar horas de cómputo conviene comprobar que la descarga, el entorno y la ruta completa funcionan de extremo a extremo.

```bash
neural-labs train-advanced --track 25_transformer_finetuning --quick --lora
```

**Cómo sabes que salió bien.** Termina sin error y deja `metrics.json`, `history.json` y `best_model.pt` en el directorio de la ejecución.

### Paso 3 — Entrenar en serio y seleccionar con `validation`

**Qué ocurre.** Se entrena el modelo completo conservando el checkpoint con el mejor valor de `accuracy` en validación, y se sella el experimento antes de evaluar `test`.

**Por qué.** Igual que en las rutas centrales: `validation` decide, `test` solo confirma, y el sello deja por escrito qué se había decidido antes de mirar.

```bash
neural-labs train-advanced --track 25_transformer_finetuning --split-seed 42 --training-seed 43 --lora
```

**Cómo sabes que salió bien.** Existe `experiment.lock.json` y `metrics.json` incluye tanto el valor de validación como el de test.

### Paso 4 — Repetir con otra semilla de entrenamiento

**Qué ocurre.** Se repite el entrenamiento con la misma partición y distinta semilla de entrenamiento.

**Por qué.** Estas arquitecturas —adversariales, contrastivas, de difusión— son especialmente sensibles a la inicialización: una sola ejecución no permite distinguir una mejora de una casualidad.

```bash
neural-labs train-advanced --track 25_transformer_finetuning --split-seed 42 --training-seed 44 --lora
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
- **Límite declarado de este dataset.** El corpus contiene titulares históricos y sesgos editoriales; no representa todo el lenguaje contemporáneo.

### Riesgos al interpretar los resultados

El corpus contiene titulares históricos y sesgos editoriales; no representa todo el lenguaje contemporáneo. Además, un mapa de atención alto no implica causalidad ni "explicación" fiable de la decisión: la atención es una entre varias señales internas del modelo y debe interpretarse con cautela.

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

- Devlin et al. (2019), *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*, NAACL — introduce el preentrenamiento bidireccional con modelado de lenguaje enmascarado.
- Sanh et al. (2019), *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter* — destilación que reduce tamaño y latencia conservando casi todo el rendimiento.
- Houlsby et al. (2019), *Parameter-Efficient Transfer Learning for NLP*, ICML — módulos adapter entrenables entre capas congeladas.
- Hu et al. (2022), *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR — adaptación de bajo rango que congela los pesos base y aprende una corrección B·A.
- Géron — *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* (3.ª ed., O'Reilly 2022), cap. 16 — tratamiento didáctico de atención y transformers para NLP.

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
| [🏁 Proyecto final: churn de telecomunicaciones](../../labs/24_capstone_real_project/README.md) | [Las 31 rutas](../../parts/README.md) | [🧷 Segmentación semántica con U-Net](../../advanced_labs/26_segmentation_unet/README.md) |

**En este laboratorio:** **📄 Guía** · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🔬 [Parte 7 — Especializaciones avanzadas](../../parts/07-especializaciones-avanzadas.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/25_transformer_finetuning/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
