# Teoría — LSTM para series temporales

<!-- nav-top -->
> 🧭 **Ruta 6 / 31** · 🔵 [Parte 2 — Arquitecturas según la forma del dato](../../parts/02-arquitecturas.md)
>
> [⬅️ 🔁 RNN para texto](../../labs/04_rnn_sequences/theory.md) · [🏠 Índice de rutas](../../parts/README.md) · [🧬 Autoencoder para fraude ➡️](../../labs/06_autoencoder_anomaly/theory.md)
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

### Por qué las puertas resuelven el desvanecimiento

La ruta 04 dejó el diagnóstico: el gradiente de una RNN simple se multiplica por W_hᵀ·diag(σ′) en cada paso, y ese producto se apaga exponencialmente. La LSTM no lo mitiga, **cambia la operación**, y ahí está toda su ventaja.

Derivando la actualización del estado de celda cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ c̃ₜ respecto del estado anterior:

∂cₜ/∂cₜ₋₁ = fₜ    (elemento a elemento),

de modo que el gradiente que atraviesa k pasos por la vía de la celda se multiplica por Π fₜ, un **producto de escalares entre 0 y 1**, y no por un producto de matrices con activaciones saturantes. La diferencia es cualitativa: cuando la puerta de olvido se mantiene cerca de 1 —el modelo ha decidido conservar esa memoria—, el factor es cerca de 1 y el gradiente **atraviesa cientos de pasos casi intacto**. Ese camino se conoce como *carrusel de error constante*, y es lo que permite aprender dependencias largas.

Obsérvese la estructura: la actualización es **aditiva**, cₜ = (algo)·cₜ₋₁ + (algo), mientras que en la RNN simple era completamente multiplicativa, hₜ = tanh(W·hₜ₋₁ + …). Es la misma idea que reaparece en las conexiones residuales de la ruta 03 y en los atajos de la 07: dejar un camino donde la señal se suma en vez de transformarse es lo que mantiene vivo el gradiente. Y aún así la LSTM no es inmune —si las puertas de olvido se cierran, la memoria y su gradiente se pierden—: la diferencia es que ahora eso es una **decisión aprendida** y no una fatalidad de la arquitectura.

De ahí una recomendación práctica bien establecida: inicializar el sesgo de la puerta de olvido en un valor positivo (típicamente 1). Con b_f = 1, la sigmoide arranca en σ(1) ≈ 0,73, así que la red empieza **conservando** memoria por defecto y aprende luego a olvidar. Con b_f = 0 arranca en 0,5 y el gradiente ya se reduce a la mitad por paso desde la primera época, justo cuando aún no ha aprendido nada que valga la pena conservar.

El costo de las puertas es lineal en parámetros. Con entrada de dimensión d y estado oculto h, cada una de las cuatro transformaciones consume (d + h)·h pesos más h sesgos, de modo que

|θ|_LSTM = 4 · ( (d + h)·h + h ),

cuatro veces una RNN simple del mismo tamaño. La **GRU** fusiona la celda con el estado oculto y usa tres puertas en vez de cuatro, así que cuesta 3·((d + h)·h + h): en torno a un 25 % menos, con rendimiento comparable en muchas tareas. Comparar ambas con el mismo presupuesto de parámetros —y no con la misma h— es la forma honesta de decidir entre ellas.

### Qué hace que una serie temporal no sea un dataset normal

La particularidad de este laboratorio no está en la arquitectura sino en el protocolo, y es donde se cometen los errores más caros.

El dato original es una única serie continua, no un conjunto de ejemplos independientes. Para entrenar se construyen ejemplos con una **ventana deslizante**: cada entrada es el tramo (x_(t−L+1), …, x_t) de longitud L y el objetivo es el valor en t + H, donde H es el **horizonte** de pronóstico. Elegir L y H no es cosmético: L acota cuánto pasado puede ver el modelo —si la serie tiene estacionalidad diaria de 24 horas, una ventana de 12 no puede capturarla— y H define un problema distinto, porque pronosticar la hora siguiente y pronosticar dentro de una semana no son la misma tarea ni admiten la misma comparación.

La partición **no puede ser aleatoria**. Repartir ventanas al azar entre `train`, `validation` y `test` coloca en el entrenamiento momentos posteriores a los que hay que predecir en la evaluación: el modelo aprende del futuro. Es una fuga de datos que no produce ningún síntoma —de hecho produce métricas excelentes— y por eso es tan peligrosa. La partición correcta es **cronológica**: un corte temporal, todo lo anterior a entrenamiento y lo posterior a evaluación, respetando el orden.

Hay un detalle más fino que se escapa incluso partiendo por fecha. Como las ventanas se solapan, una ventana de `train` que termine justo antes del corte puede tener su objetivo **después** del corte, dentro del periodo de validación. La solución estándar es dejar un hueco (*embargo*) de al menos H pasos entre particiones. Sin él, el solape filtra exactamente la información que se quería aislar.

El escalado arrastra el mismo principio: la media y la desviación se calculan **solo con el tramo de entrenamiento**. Estandarizar con las estadísticas de la serie completa introduce en el preprocesamiento información sobre el nivel y la variabilidad del futuro, que es justo lo que el modelo debería tener que inferir.

Y la línea base debe ser honesta. En series temporales, el modelo **ingenuo** —predecir que el valor siguiente será igual al último observado, ŷ_(t+H) = y_t— es sorprendentemente difícil de batir, y su versión estacional —ŷ_(t+H) = y_(t+H−s) con s el periodo— aún más. Una red que no supere claramente a ese piso no ha aprendido dinámica: ha aprendido a copiar. Por eso el error se reporta con métricas escaladas frente a esa referencia, del tipo MASE = MAE_modelo / MAE_ingenuo, cuyo valor 1 marca exactamente el punto en que el modelo deja de aportar.

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
- Fuente del dataset: https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand — **Seoul Bike Sharing Demand** (UCI Machine Learning Repository, CC BY 4.0); procedencia, versión y SHA-256 en el registro de fuentes, entrada `uci-seoul-bike-sharing-demand` — esta clase la usa para pronosticar demanda horaria de bicicletas respetando el orden cronológico y sin filtrar el futuro al pasado.
- Consulte `docs/experiment-protocol.md`, `docs/reproducibility.md` y `docs/ethics-and-licenses.md`.

<!-- nav-bottom -->
## 🧭 Navegación del recorrido

| ⬅️ Laboratorio anterior | 🏠 Índice | Laboratorio siguiente ➡️ |
|---|:---:|---|
| [🔁 RNN para texto](../../labs/04_rnn_sequences/README.md) | [Las 31 rutas](../../parts/README.md) | [🧬 Autoencoder para fraude](../../labs/06_autoencoder_anomaly/README.md) |

**En este laboratorio:** [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🔵 [Parte 2 — Arquitecturas según la forma del dato](../../parts/02-arquitecturas.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/05_lstm_time_series/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
