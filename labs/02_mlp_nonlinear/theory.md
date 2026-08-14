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
