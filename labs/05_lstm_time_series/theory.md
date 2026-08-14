# Teoría — LSTM para series temporales

<!-- nav-top -->
> 🧭 **Ruta 6 / 31** · [⬅️ 🔁 RNN para texto](../../labs/04_rnn_sequences/theory.md) · [🏠 Índice](../../README.md#laboratorios) · [🧬 Autoencoder para fraude ➡️](../../labs/06_autoencoder_anomaly/theory.md)
>
> [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

## Propósito

Pronosticar demanda horaria respetando el orden temporal.

## Idea central

Este laboratorio estudia **memoria recurrente para pronóstico temporal** usando `seoul_bike`, un dataset público real procedente de UCI.

Este laboratorio ataca directamente la limitación descubierta con las RNN simples: su incapacidad para retener información a lo largo de muchos pasos por el desvanecimiento del gradiente. La **LSTM** (Long Short-Term Memory) introduce un canal de memoria protegido —el estado de celda cₜ— y un sistema de **puertas** que deciden, de forma aprendida, qué información conservar, qué olvidar y qué exponer en cada instante. El resultado es una memoria capaz de sostener patrones a largo plazo (ciclos diarios y semanales de demanda) sin que el gradiente se disipe.

El problema es un pronóstico de series temporales genuino: predecir la demanda horaria de bicicletas compartidas en Seúl a partir de su historia reciente y variables climáticas, sobre 8.760 observaciones reales. A diferencia de la clasificación, aquí el **orden temporal es sagrado**: la partición no puede mezclar futuro con pasado, y las líneas base (persistencia, media móvil, Ridge) son duras de batir. La pregunta crítica —si el modelo supera a la persistencia en períodos de cambio— pone el foco donde un pronosticador realmente demuestra su valor.

## Fundamento matemático

La LSTM mantiene dos estados que viajan en el tiempo: el estado oculto hₜ (la salida en cada paso) y el **estado de celda** cₜ (la memoria a largo plazo). En cada instante, tres puertas —vectores con valores en (0, 1) producidos por sigmoides σ— regulan el flujo de información. Con la concatenación de la entrada xₜ y el estado previo hₜ₋₁:

Puerta de olvido:  fₜ = σ(W_f·[hₜ₋₁, xₜ] + b_f)

Puerta de entrada:  iₜ = σ(W_i·[hₜ₋₁, xₜ] + b_i)

Candidato de memoria:  c̃ₜ = tanh(W_c·[hₜ₋₁, xₜ] + b_c)

Puerta de salida:  oₜ = σ(W_o·[hₜ₋₁, xₜ] + b_o)

La actualización del estado de celda es el corazón del mecanismo y combina las puertas mediante el **producto elemento a elemento** ⊙:

cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ c̃ₜ

hₜ = oₜ ⊙ tanh(cₜ)

La lectura es intuitiva: la puerta de olvido fₜ decide qué fracción de la memoria vieja cₜ₋₁ se conserva (fₜ ≈ 1 recuerda, fₜ ≈ 0 borra); la puerta de entrada iₜ decide cuánto del nuevo candidato c̃ₜ se escribe; y la de salida oₜ decide qué parte de la memoria se expone como estado oculto. Cuando la red aprende fₜ ≈ 1, el estado de celda actúa como una **cinta transportadora** por la que la información —y el gradiente— fluye a través de muchos pasos casi sin atenuarse. Esa suma cₜ = fₜ ⊙ cₜ₋₁ + … es precisamente lo que evita el producto de jacobianos que desvanecía el gradiente en la RNN simple: la ruta aditiva mantiene ∂cₜ/∂cₜ₋₁ ≈ fₜ en lugar de un factor que se contrae exponencialmente.

Una alternativa más ligera es la **GRU** (Cho et al. 2014), que fusiona las puertas de olvido y entrada en una sola puerta de actualización y prescinde del estado de celda separado, con menos parámetros y rendimiento a menudo comparable. Para el pronóstico, la salida h_T (o la de cada paso) pasa por una capa densa que produce el valor real predicho, y el entrenamiento minimiza un error de regresión como el **MSE**, L = (1/N) Σᵢ (ŷᵢ − yᵢ)². La evaluación reporta MAE, RMSE, MAPE y R², siempre comparando contra las líneas base clásicas de series temporales.

La formulación debe conectarse con cuatro elementos: representación de entrada, función del modelo, función de pérdida y regla de actualización. El notebook muestra las dimensiones de los tensores y conserva la misma implementación que el script de terminal.

## Protocolo científico

- Ajustar transformaciones, vocabulario, normalización y selección de variables solo con `train`.
- Usar `validation` para arquitectura, hiperparámetros, checkpoint y umbrales.
- Evaluar `test` una vez, después de congelar las decisiones.
- Comparar contra **Persistencia, media móvil y Ridge**.
- Reportar variación entre semillas e intervalos de confianza; una métrica puntual no expresa toda la incertidumbre.

## Riesgos de interpretación

8.760 observaciones reales de arriendo de bicicletas y clima en Seúl.

El dataset refleja su proceso de recolección y no representa automáticamente otros períodos, países o poblaciones. Una asociación predictiva no demuestra causalidad.

## Pregunta crítica

> ¿El modelo supera persistencia en períodos de cambio?

## 🔗 Referencias

> Las referencias apuntan a las obras; no se reproduce su contenido, la redacción es original.

- Goodfellow, Bengio & Courville — *Deep Learning* (MIT Press 2016), cap. 10 — redes con puertas (LSTM/GRU) y dependencias de largo plazo.
- Hyndman & Athanasopoulos — *Forecasting: Principles and Practice* (3.ª ed., OTexts) — metodología de pronóstico, líneas base y evaluación de series temporales.
- Géron — *Hands-On Machine Learning* (3.ª ed., O'Reilly 2022), cap. 15 — procesamiento de secuencias y pronóstico con RNN/LSTM.
- Hochreiter & Schmidhuber (1997), *Long Short-Term Memory*, Neural Computation — celda LSTM original y solución al gradiente que se desvanece.
- Cho et al. (2014), *Learning Phrase Representations using RNN Encoder-Decoder (GRU)*, EMNLP — unidad recurrente con puertas simplificada.
- Fuente del dataset: https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand
- Consulte `docs/experiment-protocol.md`, `docs/reproducibility.md` y `docs/ethics-and-licenses.md`.

<!-- nav-bottom -->
## 🧭 Navegación del recorrido

| ⬅️ Laboratorio anterior | 🏠 Índice | Laboratorio siguiente ➡️ |
|---|:---:|---|
| [🔁 RNN para texto](../../labs/04_rnn_sequences/README.md) | [Las 31 rutas](../../README.md#laboratorios) | [🧬 Autoencoder para fraude](../../labs/06_autoencoder_anomaly/README.md) |

**En este laboratorio:** [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

[🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/05_lstm_time_series/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
