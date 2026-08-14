# Teoría — Fusión de sensores

<!-- nav-top -->
> 🧭 **Ruta 13 / 31** · 🟣 [Parte 3 — Familias especializadas: generar, decidir, relacionar](../../parts/03-familias-especializadas.md)
>
> [⬅️ ♻️ Transfer learning con mascotas](../../labs/11_transfer_learning/theory.md) · [🏠 Índice de rutas](../../parts/README.md) · [🎛️ Búsqueda de hiperparámetros ➡️](../../labs/13_hyperparameter_search/theory.md)
>
> [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

## Propósito

Fusionar acelerómetro y giroscopio de smartphones para reconocer actividades.

## Idea central

Este laboratorio estudia **fusión de ramas de sensores** usando `uci_har`, un dataset público real procedente de UCI. La intuición es que cada sensor aporta una vista parcial y complementaria del mismo fenómeno físico: el acelerómetro captura la aceleración lineal (útil para distinguir estar de pie de caminar), mientras que el giroscopio mide la velocidad angular (útil para detectar giros y cambios de orientación). Ninguna modalidad basta por sí sola para separar todas las actividades, pero combinadas reducen la ambigüedad.

La **fusión tardía** (late fusion) que se practica aquí procesa cada modalidad con su propia rama (una red que aprende una representación específica del sensor) y luego concatena esas representaciones antes de la cabeza de clasificación. La alternativa de **fusión temprana** (early fusion) concatenaría las señales crudas desde el inicio. La fusión tardía suele ser más robusta cuando las modalidades tienen escalas, ruidos y estructuras temporales distintas, porque permite que cada rama normalice y abstraiga su señal antes de mezclarlas. El aprendizaje de estas representaciones intermedias es el mecanismo central del deep learning aplicado a datos heterogéneos.

El laboratorio también invita a comparar frente a líneas base de una sola modalidad y frente a un modelo lineal. Esto responde la pregunta de si la fusión realmente agrega valor o si una sola rama ya resuelve la tarea. Medir la ganancia marginal de cada sensor es tan importante como alcanzar buena exactitud global.

## Fundamento matemático

La entrada se divide en dos vistas de la misma ventana temporal: x_acc (canales del acelerómetro) y x_gyro (canales del giroscopio). Cada rama aplica una función parametrizada que produce un vector de características (embedding):

  h_acc = f_acc(x_acc; θ_acc),  h_gyro = f_gyro(x_gyro; θ_gyro)

La fusión concatena ambas representaciones y la cabeza produce los logits de las K actividades:

  f = [h_acc ; h_gyro],  z = head(f; θ_head),  ŷ = softmax(z)

donde la probabilidad de la clase k es ŷ_k = e^{z_k} / Σⱼ e^{z_j}. El entrenamiento minimiza la entropía cruzada sobre N ejemplos:

  ℒ(θ) = −(1/N) Σᵢ Σₖ y_{i,k} · log ŷ_{i,k}

con y_{i,k} la codificación one-hot de la etiqueta verdadera. La actualización de todos los parámetros θ = {θ_acc, θ_gyro, θ_head} sigue el descenso de gradiente estocástico:

  θ ← θ − η · ∇_θ ℒ

El gradiente ∇_θ ℒ se propaga por retropropagación a través de la cabeza y luego se **reparte** por las dos ramas. Aquí aparece la clave de la fusión: el error retrocede por ambos caminos simultáneamente, de modo que cada rama recibe una señal de aprendizaje condicionada por lo que la otra ya aporta. Por eso el modelo puede aprender a que el giroscopio se especialice en los patrones que el acelerómetro no discrimina bien. La formulación conecta cuatro elementos: representación de entrada (las dos vistas), función del modelo (ramas + cabeza), función de pérdida (entropía cruzada) y regla de actualización (SGD con ∇). El notebook muestra las dimensiones de los tensores y conserva la misma implementación que el script de terminal.

### Dónde se juntan las señales: tres arquitecturas, tres supuestos

«Fusionar» no es una operación única. Hay tres puntos del recorrido donde se pueden unir dos sensores, y cada uno codifica un supuesto distinto sobre el problema.

La **fusión temprana** concatena las señales crudas antes de cualquier procesamiento: x = [x_acc, x_gyro], y a partir de ahí hay un solo modelo. Es la más simple y la que más libertad da al modelo para descubrir relaciones cruzadas desde el primer instante, pero supone que ambas señales están **alineadas y en escalas comparables**, y hace imposible saber qué aportó cada una.

La **fusión tardía** entrena un modelo por sensor y combina sus predicciones —promedio, voto o una capa que aprende los pesos—: ŷ = w_acc·ŷ_acc + w_gyro·ŷ_gyro. Es robusta —si un sensor falla, el otro sigue prediciendo—, permite entrenar cada rama por separado y es trivialmente interpretable. Su límite es serio: al combinar solo al final, **no puede aprender interacciones** entre modalidades. Si distinguir «subir escaleras» de «caminar» requiere cruzar una firma del acelerómetro con una del giroscopio, la fusión tardía nunca verá esa combinación.

La **fusión intermedia** —la de este laboratorio— es el punto medio: cada sensor tiene su propia rama que aprende una representación adaptada a su naturaleza, y esas representaciones se concatenan antes de la cabeza común, h = [h_acc, h_gyro]. Conserva la especialización por sensor y sí permite que la cabeza aprenda interacciones. Es la razón de que sea la opción por defecto en reconocimiento de actividad.

Un detalle del gradiente que conviene tener presente: al concatenar y pasar por una capa densa, cada rama recibe gradiente a través de **su bloque de columnas** de la matriz de la cabeza. Si una modalidad es mucho más informativa, la cabeza tiende a apoyarse en ella, sus gradientes dominan y la otra rama aprende poco —fenómeno conocido como **modalidad perezosa**—. Se detecta comparando el desempeño de cada rama por separado con el del modelo fusionado: si la fusión no supera a la mejor rama sola, no hubo fusión real, hubo una rama trabajando y otra decorando.

### La ablación es el experimento, no un extra

La pregunta que da sentido al laboratorio no es «¿qué exactitud alcanza el modelo fusionado?» sino «¿cuánto aporta cada sensor?», y solo la **ablación** la responde. Se entrenan tres modelos con el mismo protocolo —solo acelerómetro, solo giroscopio, y ambos— y se comparan.

La lectura de los resultados es directa y admite tres desenlaces, todos informativos. Si la fusión supera claramente a las dos ramas individuales, las señales son **complementarias**: cada una aporta información que la otra no tiene. Si la fusión iguala a la mejor rama, son **redundantes** para esta tarea, y el segundo sensor es costo sin beneficio —una conclusión valiosa, porque cada sensor consume batería y ancho de banda—. Y si la fusión es **peor** que la mejor rama sola, el modelo está sobreajustando la dimensión añadida o una de las ramas está introduciendo ruido, lo que apunta a un problema de capacidad o de normalización.

Para que la comparación sea válida, las tres variantes deben compartir partición, semillas, presupuesto de épocas y criterio de parada. Cambiar el número de parámetros al quitar una rama es inevitable; lo que no puede cambiar es nada más.

### El detalle que hace que este dataset se filtre

Aquí hay una trampa específica de las señales de sensores, y es la razón por la que este laboratorio insiste en la auditoría de particiones.

Las ventanas de UCI HAR se construyen con **solapamiento** —cada ventana comparte la mitad de sus muestras con la siguiente—. Si esas ventanas se reparten al azar entre `train` y `test`, dos ventanas casi idénticas acaban una en cada lado, y el modelo evalúa sobre datos que prácticamente ha visto. La exactitud sube varios puntos sin que nada falle a la vista, y el resultado no se sostiene con datos nuevos.

Peor aún: aunque las ventanas no se solaparan, repartir al azar mezcla al **mismo sujeto** entre entrenamiento y evaluación. Cada persona camina, se sienta y sube escaleras con una firma característica, así que el modelo puede reconocer al sujeto y usar eso para predecir su actividad. Lo que se mide entonces no es «reconocer actividades» sino «reconocer a estas doce personas», y el modelo se derrumba con un usuario nuevo. La partición correcta es **por sujeto**: unos sujetos completos para entrenar, otros distintos para evaluar. Es exactamente el escenario que explora la ruta 15, y la razón de que ambos laboratorios usen el mismo dataset con particiones distintas.

Sobre el preprocesamiento, dos reglas que se derivan del mismo principio. La normalización se ajusta **solo con `train`**, y con las estadísticas de cada canal por separado: acelerómetro y giroscopio miden magnitudes físicas distintas —aceleración y velocidad angular— y en unidades distintas, así que estandarizarlos juntos deja a uno dominando la escala del otro. Y si los sensores tuvieran frecuencias de muestreo distintas, habría que remuestrearlos a una rejilla común antes de concatenar; asumir alineación sin comprobarla es una fuente silenciosa de degradación.

## Protocolo científico

- Ajustar transformaciones, vocabulario, normalización y selección de variables solo con `train`.
- Usar `validation` para arquitectura, hiperparámetros, checkpoint y umbrales.
- Evaluar `test` una vez, después de congelar las decisiones.
- Comparar contra **Acelerómetro solo, giroscopio solo y regresión logística**.
- Reportar variación entre semillas e intervalos de confianza; una métrica puntual no expresa toda la incertidumbre.

## Riesgos de interpretación

Señales inerciales reales de 30 participantes.

El dataset refleja su proceso de recolección y no representa automáticamente otros períodos, países o poblaciones. Una asociación predictiva no demuestra causalidad.

## Pregunta crítica

> ¿Qué modalidad explica cada actividad?

## 🔗 Referencias

> Las referencias apuntan a las obras; no se reproduce su contenido, la redacción es original.

- Goodfellow, Bengio & Courville — *Deep Learning* (MIT Press, 2016) — marco general sobre aprendizaje de representaciones y por qué las capas intermedias abstraen mejor los datos heterogéneos.
- Baltrušaitis, Ahuja & Morency (2019), *Multimodal Machine Learning: A Survey and Taxonomy*, IEEE TPAMI — taxonomía de estrategias de fusión (temprana, tardía, híbrida) y de los desafíos de alinear modalidades.
- Radford et al. (2021), *Learning Transferable Visual Models from Natural Language Supervision (CLIP)*, ICML — ejemplo influyente de aprender un espacio compartido entre modalidades distintas.
- Fuente del dataset: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
- Consulte `docs/experiment-protocol.md`, `docs/reproducibility.md` y `docs/ethics-and-licenses.md`.

<!-- nav-bottom -->
## 🧭 Navegación del recorrido

| ⬅️ Laboratorio anterior | 🏠 Índice | Laboratorio siguiente ➡️ |
|---|:---:|---|
| [♻️ Transfer learning con mascotas](../../labs/11_transfer_learning/README.md) | [Las 31 rutas](../../parts/README.md) | [🎛️ Búsqueda de hiperparámetros](../../labs/13_hyperparameter_search/README.md) |

**En este laboratorio:** [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🟣 [Parte 3 — Familias especializadas: generar, decidir, relacionar](../../parts/03-familias-especializadas.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/12_multimodal_fusion/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
