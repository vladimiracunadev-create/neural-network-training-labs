# Teoría — MLP multiclase

<!-- nav-top -->
> 🧭 **Ruta 3 / 31** · 🟢 [Parte 1 — Fundamentos: de la derivada a la primera red](../../parts/01-fundamentos.md)
>
> [⬅️ 🧩 Perceptrón con PyTorch](../../labs/01_pytorch_perceptron/theory.md) · [🏠 Índice de rutas](../../parts/README.md) · [🖼️ CNN para visión ➡️](../../labs/03_cnn_vision/theory.md)
>
> [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

## Propósito

Resolver clasificación no lineal con capas densas, activaciones y regularización.

## Idea central

Este laboratorio estudia **red multicapa para relaciones no lineales** usando `dry_bean`, un dataset público real procedente de UCI.

Un clasificador lineal solo puede trazar hiperplanos: falla cuando las clases se entrelazan de forma no lineal. El perceptrón multicapa (MLP) resuelve esto apilando capas de neuronas separadas por **funciones de activación no lineales**. La clave conceptual es que sin esas no linealidades, componer varias capas lineales sería inútil —el producto de matrices sigue siendo una matriz, es decir, otro modelo lineal—. La activación (aquí ReLU) es lo que permite que cada capa doble y pliegue el espacio de representación, de modo que clases inseparables en el espacio original se vuelvan separables en el espacio aprendido.

El problema —clasificar 13.611 granos en siete variedades a partir de 16 atributos de forma— es genuinamente multiclase y no lineal, ideal para observar cómo una capa oculta supera a la regresión logística multinomial. El laboratorio también introduce la regularización (dropout, weight decay) como respuesta al mayor riesgo de sobreajuste que trae la capacidad adicional.

## Fundamento matemático

Una red de una capa oculta calcula su predicción en dos etapas. Primero proyecta la entrada a un espacio oculto y aplica una no linealidad; luego proyecta ese espacio oculto a los logits de las clases:

h = ReLU(x·W₁ + b₁)    con    ReLU(a) = max(0, a)

logits = h·W₂ + b₂

La función **ReLU** (Rectified Linear Unit) es engañosamente simple: deja pasar los valores positivos y anula los negativos. Su derivada es 1 para a > 0 y 0 para a < 0, lo que la hace barata de calcular y, sobre todo, evita el problema del **desvanecimiento del gradiente** que sufren la sigmoide y la tanh, cuyas derivadas se aproximan a 0 en sus extremos y frenan el aprendizaje en redes profundas. El "codo" no lineal en a = 0 es lo que aporta la capacidad expresiva: cada neurona ReLU introduce un pliegue lineal por tramos, y su combinación aproxima superficies de decisión arbitrariamente complejas.

Este poder no es una intuición vaga sino un resultado formal: el **teorema de aproximación universal** (Cybenko 1989 para sigmoides; Hornik 1991 para activaciones generales) demuestra que una red con una sola capa oculta y suficientes neuronas puede aproximar cualquier función continua sobre un conjunto compacto con el error que se desee. El teorema garantiza la *existencia* de los pesos, no que el descenso de gradiente los encuentre fácilmente; en la práctica, apilar más capas suele ser más eficiente en parámetros que ensanchar una sola.

Para clasificación multiclase, los logits se convierten en una distribución de probabilidad con **softmax**, que normaliza exponenciales para que sumen 1:

softmax(z)ₖ = e^(zₖ) / Σⱼ e^(zⱼ)

y se entrena minimizando la **entropía cruzada categórica**, L = −(1/N) Σᵢ ln( p_{i, yᵢ} ), donde p_{i, yᵢ} es la probabilidad que el modelo asigna a la clase verdadera del ejemplo i. En PyTorch, `CrossEntropyLoss` fusiona softmax y log-verosimilitud de forma numéricamente estable, por lo que la última capa entrega logits crudos. Los gradientes fluyen hacia atrás por retropropagación: ∂L/∂W₂ se calcula directamente y, por la regla de la cadena a través de la ReLU, el error se propaga a la capa oculta (∂L/∂W₁), donde la máscara de la ReLU bloquea el gradiente en las neuronas que estaban inactivas.

La formulación debe conectarse con cuatro elementos: representación de entrada, función del modelo, función de pérdida y regla de actualización. El notebook muestra las dimensiones de los tensores y conserva la misma implementación que el script de terminal.

### Las ecuaciones de la retropropagación en esta red

Con dos capas, la retropropagación cabe en cuatro líneas y conviene tenerlas escritas, porque son el esqueleto de todo lo que viene después. Llamando a = x·W₁ + b₁ a la preactivación oculta, h = ReLU(a), z = h·W₂ + b₂ a los logits y p = softmax(z):

δ² = p − Y,   ∂L/∂W₂ = hᵀ·δ² / N,   ∂L/∂b₂ = Σ_filas δ² / N,

δ¹ = (δ²·W₂ᵀ) ⊙ 𝟙[a > 0],   ∂L/∂W₁ = xᵀ·δ¹ / N,   ∂L/∂b₁ = Σ_filas δ¹ / N,

donde ⊙ es el producto elemento a elemento y 𝟙[a > 0] la máscara de la ReLU. Vale la pena leer la segunda línea despacio: el error de la capa de salida viaja hacia atrás multiplicado por W₂ᵀ —la misma matriz del paso hacia adelante, transpuesta— y luego se **apaga** en las posiciones donde la neurona estaba inactiva. Una neurona que no participó en la predicción tampoco recibe corrección.

Que δ² = p − Y no es evidente, y es el mismo regalo que aparecía en la ruta 00. Derivando la entropía cruzada categórica respecto de los logits, el término del softmax ∂pₖ/∂z_j = pₖ·(δ_kj − p_j) se combina con ∂L/∂pₖ = −y_k/pₖ y todo se simplifica a p − y. Softmax con entropía cruzada, igual que sigmoide con entropía cruzada binaria, están emparejadas para que el gradiente sea el error puro.

Sobre el softmax hay una propiedad que se usa en toda implementación seria: es **invariante a desplazamientos**, softmax(z + c) = softmax(z) para cualquier constante c. Restar el máximo, softmax(z − max z), no cambia el resultado y garantiza que el mayor exponente sea e⁰ = 1, evitando el desbordamiento de e^z con logits grandes. Es lo que `CrossEntropyLoss` hace internamente, y la razón de que la última capa deba devolver logits crudos.

### Cuántos parámetros hay y cómo inicializarlos

Contar los parámetros es inmediato y conviene hacerlo antes de entrenar. Cada capa densa de m entradas y n salidas aporta m·n pesos más n sesgos, así que para una pila de anchuras (H₁, H₂, …) entre d características y C clases:

|θ| = (d·H₁ + H₁) + (H₁·H₂ + H₂) + … + (H_L·C + C).

El modelo tabular de este repositorio usa por defecto dos capas ocultas de 128 y 64 unidades. Con las 16 características de forma del dataset y sus 7 variedades, la cuenta es 16·128 + 128 + 128·64 + 64 + 64·7 + 7 = **10 887 parámetros**. Frente a los 13 611 granos del conjunto completo, la red tiene casi un parámetro por ejemplo: es exactamente la situación en la que memorizar es una estrategia disponible, y la que justifica el dropout y el weight decay que se estudian en la ruta 19.

La **inicialización** no es un detalle. Si todos los pesos se ponen a cero, todas las neuronas ocultas calculan lo mismo, reciben el mismo gradiente y siguen siendo idénticas para siempre: la red se comporta como si tuviera una sola neurona oculta. Es el problema de **simetría**, y por eso los pesos se inicializan al azar. Pero la escala de ese azar importa: si la varianza es alta, las preactivaciones crecen capa a capa y saturan; si es baja, se encogen y la señal se apaga.

La receta que usan las redes con ReLU es la de **He**: muestrear W de una normal de varianza 2/fan_in, donde fan_in es el número de entradas de la capa. El factor 2 compensa exactamente que la ReLU anula la mitad de las activaciones y por tanto reduce la varianza a la mitad. Para activaciones simétricas como tanh, la inicialización de **Glorot** usa 2/(fan_in + fan_out), que equilibra la propagación en ambos sentidos. Es la primera aparición de una idea que la ruta 18 desarrolla: gran parte del arte del entrenamiento consiste en mantener la varianza de las activaciones y de los gradientes dentro de un rango sano de extremo a extremo de la red.

De ahí sale también el fallo característico de la ReLU: si una neurona recibe una actualización que deja su preactivación negativa para **todos** los ejemplos, su gradiente es cero permanentemente y no vuelve a aprender nunca. Es la **ReLU muerta**, y su causa habitual es una tasa de aprendizaje demasiado alta. Las variantes Leaky ReLU y GELU, que se comparan en la ruta 17, existen precisamente para dejar pasar algo de gradiente en la zona negativa.

### Por qué la profundidad gana a la anchura

El teorema de aproximación universal garantiza que basta una capa oculta, pero no dice cuántas neuronas hacen falta, y ahí está la trampa: para muchas funciones ese número crece exponencialmente. Contar **regiones lineales** lo hace concreto. Una red ReLU es una función lineal a trozos, y el número de regiones en que divide el espacio de entrada mide su capacidad de plegar la frontera de decisión. Una red de una capa con H neuronas sobre d entradas genera del orden de O(H^d) regiones; una red profunda de L capas con H neuronas cada una alcanza del orden de Ω((H/d)^(d(L−1))·H^d), es decir, **exponencial en la profundidad** y solo polinómico en la anchura.

La lectura práctica es que añadir una capa multiplica la capacidad expresiva mucho más barato que ensanchar la existente, y es la justificación de que este laboratorio compare profundidad frente a anchura en vez de dar por buena la primera arquitectura que funcione. La contrapartida —redes profundas más difíciles de optimizar por gradientes que se desvanecen— es lo que motivará las conexiones residuales y la normalización en rutas posteriores.

## Protocolo científico

- Ajustar transformaciones, vocabulario, normalización y selección de variables solo con `train`.
- Usar `validation` para arquitectura, hiperparámetros, checkpoint y umbrales.
- Evaluar `test` una vez, después de congelar las decisiones.
- Comparar contra **Regresión logística multinomial y Random Forest**.
- Reportar variación entre semillas e intervalos de confianza; una métrica puntual no expresa toda la incertidumbre.

## Riesgos de interpretación

13.611 granos de siete variedades reales y 16 atributos de forma.

El dataset refleja su proceso de recolección y no representa automáticamente otros períodos, países o poblaciones. Una asociación predictiva no demuestra causalidad.

## Pregunta crítica

> ¿La complejidad adicional supera de forma estable a la línea base?

## 🔗 Referencias

> Las referencias apuntan a las obras; no se reproduce su contenido, la redacción es original.

- Goodfellow, Bengio & Courville — *Deep Learning* (MIT Press 2016), cap. 6 — redes hacia adelante, ReLU, softmax y retropropagación.
- Géron — *Hands-On Machine Learning* (3.ª ed., O'Reilly 2022), cap. 10 — diseño e implementación de MLP y regularización.
- Prince — *Understanding Deep Learning* (MIT Press 2024), cap. 3–4 — redes superficiales y profundas con activaciones lineales por tramos.
- He et al. (2015), *Delving Deep into Rectifiers*, ICCV — la inicialización de varianza 2/fan_in para redes con ReLU.
- Glorot & Bengio (2010), *Understanding the difficulty of training deep feedforward neural networks*, AISTATS — la inicialización que equilibra la varianza en ambos sentidos.
- Montúfar et al. (2014), *On the Number of Linear Regions of Deep Neural Networks*, NeurIPS — el conteo de regiones lineales que cuantifica la ventaja de la profundidad.
- Cybenko (1989), *Approximation by superpositions of a sigmoidal function*, Math. Control Signals Systems — teorema de aproximación universal para sigmoides.
- Hornik (1991), *Approximation capabilities of multilayer feedforward networks*, Neural Networks — generalización del teorema a activaciones arbitrarias.
- Fuente del dataset: https://archive.ics.uci.edu/dataset/602/dry+bean+dataset
- Consulte `docs/experiment-protocol.md`, `docs/reproducibility.md` y `docs/ethics-and-licenses.md`.

<!-- nav-bottom -->
## 🧭 Navegación del recorrido

| ⬅️ Laboratorio anterior | 🏠 Índice | Laboratorio siguiente ➡️ |
|---|:---:|---|
| [🧩 Perceptrón con PyTorch](../../labs/01_pytorch_perceptron/README.md) | [Las 31 rutas](../../parts/README.md) | [🖼️ CNN para visión](../../labs/03_cnn_vision/README.md) |

**En este laboratorio:** [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🟢 [Parte 1 — Fundamentos: de la derivada a la primera red](../../parts/01-fundamentos.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/02_mlp_nonlinear/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
