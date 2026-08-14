# Teoría — Autoencoder para fraude

<!-- nav-top -->
> 🧭 **Ruta 7 / 31** · 🔵 [Parte 2 — Arquitecturas según la forma del dato](../../parts/02-arquitecturas.md)
>
> [⬅️ 📈 LSTM para series temporales](../../labs/05_lstm_time_series/theory.md) · [🏠 Índice de rutas](../../parts/README.md) · [🔭 Transformer para noticias ➡️](../../labs/07_transformer_attention/theory.md)
>
> [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

## Propósito

Detectar transacciones fraudulentas mediante error de reconstrucción.

## Idea central

Este laboratorio estudia **reconstrucción para detección de anomalías** usando `credit_card_fraud`, un dataset público real procedente de Kaggle / ULB.

La idea rectora es entrenar un modelo que solo aprenda a describir bien lo *normal*. Un **autoencoder** es una red con forma de reloj de arena: un codificador comprime la entrada x a una representación latente z de baja dimensión (el "cuello de botella"), y un decodificador intenta reconstruir x a partir de z. Si únicamente mostramos transacciones legítimas durante el entrenamiento, la red se especializa en la geometría de esa mayoría y aprende a copiar sus regularidades. Cuando más tarde le presentamos una transacción fraudulenta —que vive fuera de esa variedad aprendida— el decodificador falla y el **error de reconstrucción** se dispara. Ese error es, en la práctica, un detector de anomalías: no clasificamos "fraude vs. no fraude" directamente, sino que medimos cuánto se desvía cada caso del patrón normal.

El cuello de botella es lo que hace que esto funcione: al forzar z a tener menos dimensiones que x, la red no puede memorizar la identidad y debe descubrir los factores latentes que explican las transacciones comunes. Esto es especialmente valioso en fraude, donde los positivos son rarísimos (≈0,17 % del total) y un clasificador supervisado clásico tiende a ignorarlos; el enfoque no supervisado por reconstrucción esquiva ese desbalance porque nunca necesita ejemplos de fraude para aprender.

## Fundamento matemático

Formalmente, el codificador es una función f con parámetros θ que produce el código z = f(x; θ) ∈ ℝᵏ, y el decodificador g con parámetros φ produce la reconstrucción x̂ = g(z; φ) = g(f(x; θ); φ). Con k ≪ d (dimensión de x), la composición está obligada a ser una **proyección con pérdida** sobre una variedad de baja dimensión. El objetivo de entrenamiento minimiza el error cuadrático medio de reconstrucción sobre las transacciones normales:

    ℒ(θ, φ) = 𝔼ₓ ‖ x − g(f(x; θ); φ) ‖²  ≈  (1/N) Σᵢ ‖ xᵢ − x̂ᵢ ‖²

El gradiente ∇_{θ,φ} ℒ se propaga por retropropagación a través de decodificador y codificador, y los pesos se actualizan con descenso de gradiente estocástico o Adam: θ ← θ − η ∇_θ ℒ. La conexión con los cuatro elementos del laboratorio es: la **representación de entrada** es el vector x de características de la transacción (28 componentes PCA anonimizadas más `Time` y `Amount` normalizados); la **función del modelo** es la composición g∘f; la **función de pérdida** es el MSE de reconstrucción arriba; y la **regla de actualización** es el paso de gradiente. El notebook muestra las dimensiones de los tensores en cada capa y conserva la misma implementación que el script de terminal.

¿Por qué el MSE minimizado sobre datos normales sirve como puntaje de anomalía? Si asumimos que la reconstrucción está sujeta a un ruido gaussiano isótropo, minimizar ‖x − x̂‖² equivale a maximizar la log-verosimilitud de x bajo el modelo. Tras entrenar, el error r(x) = ‖x − g(f(x))‖² es bajo para lo que la red sabe reconstruir (lo normal) y alto para lo que nunca vio (el fraude). La regla de decisión es un simple umbral: se marca anomalía cuando r(x) > τ. El umbral τ **no se elige a ojo**: se calibra en `validation`, por ejemplo tomando un percentil alto (p. ej. el 99) de la distribución de errores sobre datos legítimos, o el punto que optimiza F1/coste esperado. Variar τ recorre la curva precision–recall completa, y por eso el laboratorio reporta ROC-AUC y PR-AUC en lugar de una sola métrica puntual.

### Qué relación tiene esto con PCA, y cómo se dimensiona el cuello de botella

Hay un resultado que fija exactamente qué está aprendiendo la red. Si el autoencoder fuera **lineal** —sin activaciones— y se entrenara con error cuadrático, su solución óptima abarcaría el mismo subespacio que las k primeras componentes principales de los datos. No es que el autoencoder «se parezca» a PCA: en el caso lineal es equivalente a él. La consecuencia metodológica es directa: **PCA es la referencia que hay que superar**, porque si la variedad donde viven las transacciones normales fuera un subespacio plano, la red no aportaría nada que un método de álgebra lineal cerrada no diera más rápido y de forma determinista. Las activaciones se justifican solo si esa variedad es curva.

Eso también explica por qué el cuello de botella es el hiperparámetro crítico. Si la dimensión latente k igualara la de la entrada, la red podría aprender la **función identidad**: reconstruiría todo a la perfección —incluido el fraude— y el error dejaría de discriminar. El poder de detección viene precisamente de que la red *no puede* reconstruirlo todo y se ve obligada a gastar su capacidad en lo que vio con más frecuencia. Con k demasiado pequeño, en cambio, ni siquiera lo normal se reconstruye bien: el error de fondo sube, la distribución de puntuaciones de ambas clases se solapa y la separación se pierde. La curva de PR-AUC frente a k tiene por eso forma de campana, y encontrar su máximo es trabajo de `validation`.

El error de reconstrucción es además **sensible a la escala** de las variables. Al sumar ‖x − x̂‖² sobre componentes con rangos distintos, una sola variable de magnitud grande puede aportar casi todo el error, y entonces lo que se está detectando no es «anomalía» sino «valor alto en esa columna». En este dataset las 28 componentes vienen ya de una PCA y están en escalas comparables, pero `Time` y `Amount` no: normalizarlas con las estadísticas de `train` es lo que hace que el error signifique algo.

### La contaminación del entrenamiento y el piso de las métricas

El método asume que el conjunto de entrenamiento contiene **solo** transacciones normales, y esa premisa casi nunca se cumple del todo: si un pequeño porcentaje de fraudes se cuela, la red aprende a reconstruirlos también y su error baja, justo el efecto contrario al deseado. La consecuencia es contraintuitiva y conviene anticiparla: **entrenar más tiempo puede empeorar la detección**, porque la capacidad sobrante acaba dedicándose a memorizar los pocos casos raros del entrenamiento. Es una razón adicional para seleccionar el checkpoint por la métrica de `validation` y no por la pérdida de reconstrucción, que seguirá bajando.

Sobre la interpretación de las métricas en un problema tan desbalanceado, conviene tener presente dónde está el piso de cada una. La **exactitud** es inservible: predecir «todo normal» supera el 99 % en este dataset. El **ROC-AUC** parece bueno con facilidad, porque su eje de falsos positivos se normaliza por la clase mayoritaria y multiplicar por diez las falsas alarmas apenas mueve la curva. El **PR-AUC**, en cambio, tiene como piso la **prevalencia** de la clase positiva —una fracción muy pequeña aquí—, así que cualquier valor apreciable representa una mejora real sobre el azar. Por eso, entre las dos cifras que el laboratorio reporta, la que hay que mirar primero es la segunda.

Y para decidir si el sistema sirve en operación, la cifra más honesta suele ser la **precisión a un presupuesto fijo**: de las N transacciones con mayor puntuación —las N que un equipo humano puede revisar en un turno—, cuántas eran fraude. Traduce el modelo a la restricción real, que no es un umbral abstracto sino cuántas revisiones caben en un día.

Una extensión conceptual importante es el **autoencoder variacional (VAE)**: en lugar de un código puntual z, el codificador produce una distribución q(z|x) = 𝒩(μ(x), σ²(x)) y se optimiza el ELBO, que suma el término de reconstrucción y una regularización KL, ℒ = 𝔼_q[‖x − x̂‖²] + β·D_KL(q(z|x) ‖ 𝒩(0, I)). El término KL empuja el espacio latente hacia una gaussiana estándar y da un puntaje de anomalía probabilístico más estable; entender el autoencoder determinista de este laboratorio es el paso previo natural hacia esa formulación.

## Protocolo científico

- Ajustar transformaciones, vocabulario, normalización y selección de variables solo con `train`.
- Usar `validation` para arquitectura, hiperparámetros, checkpoint y umbrales.
- Evaluar `test` una vez, después de congelar las decisiones.
- Comparar contra **Isolation Forest**.
- Reportar variación entre semillas e intervalos de confianza; una métrica puntual no expresa toda la incertidumbre.

## Riesgos de interpretación

284.807 transacciones reales; el laboratorio evita reequilibrar el conjunto de test.

El dataset refleja su proceso de recolección y no representa automáticamente otros períodos, países o poblaciones. Una asociación predictiva no demuestra causalidad.

## Pregunta crítica

> ¿Qué costo tiene priorizar recall frente a precision?

> Las referencias apuntan a las obras; no se reproduce su contenido, la redacción es original.

## 🔗 Referencias

- Goodfellow, Bengio & Courville — *Deep Learning* (MIT Press, 2016), cap. 14 — teoría de autoencoders, cuello de botella y autoencoders regularizados/de reducción de dimensión.
- Géron — *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* (3.ª ed., O'Reilly), cap. 17 — autoencoders y GANs en la práctica, detección de anomalías por reconstrucción.
- Hinton & Salakhutdinov (2006), *Reducing the Dimensionality of Data with Neural Networks*, Science — mostró que un autoencoder profundo aprende códigos compactos mejores que PCA.
- Kingma & Welling (2014), *Auto-Encoding Variational Bayes (VAE)*, ICLR — formulación variacional del autoencoder y base del puntaje de anomalía probabilístico.
- Fuente del dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Consulte `docs/experiment-protocol.md`, `docs/reproducibility.md` y `docs/ethics-and-licenses.md`.

<!-- nav-bottom -->
## 🧭 Navegación del recorrido

| ⬅️ Laboratorio anterior | 🏠 Índice | Laboratorio siguiente ➡️ |
|---|:---:|---|
| [📈 LSTM para series temporales](../../labs/05_lstm_time_series/README.md) | [Las 31 rutas](../../parts/README.md) | [🔭 Transformer para noticias](../../labs/07_transformer_attention/README.md) |

**En este laboratorio:** [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🔵 [Parte 2 — Arquitecturas según la forma del dato](../../parts/02-arquitecturas.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/06_autoencoder_anomaly/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
