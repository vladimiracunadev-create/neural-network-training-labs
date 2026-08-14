# Teoría — GNN sobre red de citas

<!-- nav-top -->
> 🧭 **Ruta 10 / 31** · 🟣 [Parte 3 — Familias especializadas: generar, decidir, relacionar](../../parts/03-familias-especializadas.md)
>
> [⬅️ 🎨 GAN generativa](../../labs/08_gan_generation/theory.md) · [🏠 Índice de rutas](../../parts/README.md) · [🕹️ DQN para inventario con demanda real ➡️](../../labs/10_dqn_reinforcement/theory.md)
>
> [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

## Propósito

Clasificar publicaciones científicas usando texto y enlaces de citas.

## Idea central

Este laboratorio estudia **propagación de mensajes sobre grafos** usando `cora`, un dataset público real procedente de PyTorch Geometric / Planetoid.

Cora es una red de citas: cada nodo es un artículo científico descrito por un vector de palabras (bolsa de términos), y cada arista es una cita entre dos artículos. La hipótesis que da sentido al laboratorio es la **homofilia**: los artículos que se citan tienden a tratar temas afines, de modo que la *estructura* del grafo aporta información que el texto por sí solo no captura. Una **red neuronal de grafos (GNN)** explota esa estructura haciendo que cada nodo actualice su representación combinando la suya con la de sus vecinos. Al apilar varias capas, la información se propaga a vecinos de vecinos, y cada nodo termina con un embedding que resume su vecindario local en el grafo.

El mecanismo general se llama **paso de mensajes** (message passing): en cada capa, cada nodo (1) recibe "mensajes" de sus vecinos, (2) los agrega con una función permutación-invariante (suma, media, máximo o atención) y (3) actualiza su estado con esa agregación. La línea base del laboratorio —un MLP que ignora las aristas— sirve justo para cuantificar cuánto aporta la estructura de citaciones frente a usar solo el texto de cada artículo.

## Fundamento matemático

La **red convolucional de grafos (GCN)** de Kipf & Welling define la actualización de una capa como:

    H^(l+1) = σ( D̃^{−1/2} Ã D̃^{−1/2} H^(l) W^(l) )

Desglosemos cada símbolo. H^(l) ∈ ℝ^{N×d_l} apila las representaciones de los N nodos en la capa l (H^(0) son las características de entrada). Ã = A + I es la matriz de adyacencia con **auto-lazos** añadidos, para que cada nodo se incluya a sí mismo en la agregación y no pierda su propia información. D̃ es la matriz diagonal de grados de Ã, con D̃ᵢᵢ = Σⱼ Ãᵢⱼ. W^(l) es la matriz de pesos aprendible que transforma las características, y σ es una no linealidad (ReLU). El término D̃^{−1/2} Ã D̃^{−1/2} es la **adyacencia normalizada simétricamente**: propaga las representaciones a los vecinos pero reescalando cada mensaje por 1/√(dᵢ·dⱼ), de modo que los nodos de grado muy alto (muy citados) no dominen la suma ni disparen la escala de las activaciones.

Intuitivamente, cada fila de esa multiplicación calcula, para el nodo i, un **promedio ponderado normalizado** de las características transformadas de i y de sus vecinos: hᵢ^(l+1) = σ( Σ_{j∈𝒩(i)∪{i}} (1/√(d̃ᵢ d̃ⱼ)) · hⱼ^(l) W^(l) ). Apilar L capas equivale a difundir información hasta L saltos de distancia; con L=2, cada artículo "ve" a los artículos que cita y a los que citan a esos. Un exceso de capas provoca **sobre-suavizado** (over-smoothing): las representaciones de todos los nodos convergen y se vuelven indistinguibles, por lo que en la práctica las GCN son poco profundas.

Conectando con los cuatro elementos: la **representación de entrada** es la matriz H^(0) de vectores de palabras por nodo más la estructura del grafo en A; la **función del modelo** es el apilamiento de capas GCN que termina en un softmax sobre las 7 clases temáticas; la **función de pérdida** es la entropía cruzada calculada *solo sobre los nodos de entrenamiento* enmascarados, ℒ = −Σ_{i∈train} Σ_c y_{ic} log ŷ_{ic}; y la **regla de actualización** es descenso de gradiente (Adam), θ ← θ − η ∇_θ ℒ. Es un problema **transductivo**: el grafo completo (con todos los nodos y aristas) participa en cada forward, pero el gradiente solo usa las etiquetas de la máscara de train. El notebook muestra las dimensiones de los tensores (N, d_l) en cada capa y conserva la misma implementación que el script de terminal.

El laboratorio compara variantes del paso de mensajes. **GraphSAGE** (Hamilton et al.) reemplaza la agregación por una que muestrea un subconjunto de vecinos y concatena el estado propio con el agregado, lo que la hace **inductiva** (generaliza a nodos nuevos no vistos). **GAT** (Veličković et al.) sustituye los pesos fijos de normalización por **coeficientes de atención aprendidos** α_{ij} = softmax_j( LeakyReLU(aᵀ[W hᵢ ‖ W hⱼ]) ), de modo que cada nodo decide cuánto pesar a cada vecino en lugar de usar solo el grado. Comparar GCN, GraphSAGE y GAT ilustra cómo cambia el resultado según cómo se agregan los mensajes.

### El esquema general: paso de mensajes

Las tres variantes son casos particulares de un mismo patrón, y verlo así evita aprenderlas como recetas sueltas. Toda capa de una GNN calcula, para cada nodo v:

h_v^(ℓ+1) = ACTUALIZAR( h_v^(ℓ), AGREGAR( { h_u^(ℓ) : u ∈ 𝒩(v) } ) ).

La GCN usa como agregador un promedio con pesos fijos por el grado; GraphSAGE muestrea vecinos y admite media, máximo o LSTM como agregador; GAT aprende los pesos con atención. Cambia el agregador, no el esquema.

Hay una restricción que ese agregador debe cumplir y que determina qué se puede usar: tiene que ser **invariante a permutaciones**. Los vecinos de un nodo son un conjunto, no una lista; si se numeran en otro orden, la representación no puede cambiar. Suma, media y máximo cumplen; concatenar en orden, no. Esa es la razón matemática de que las GNN se construyan con esas operaciones y no con una capa densa sobre los vecinos concatenados.

La elección tampoco es neutra en poder expresivo. La **media** pierde la información del grado: un nodo con dos vecinos idénticos y otro con veinte producen la misma representación. El **máximo** pierde multiplicidades: registra qué tipos de vecino hay, no cuántos. La **suma** conserva ambas cosas, y por eso es la única de las tres que alcanza el poder de distinción del test de isomorfismo de Weisfeiler-Lehman, la cota superior conocida para esta familia de arquitecturas. Si dos grafos no se distinguen con ese test, ninguna GNN de paso de mensajes los distinguirá.

### Qué hace la normalización simétrica, y por qué solo dos capas

La matriz Â = D̃^(−1/2)·Ã·D̃^(−1/2) parece una convención arbitraria y no lo es. Sin normalizar, multiplicar por A suma las representaciones de los vecinos, así que un nodo muy conectado acumula valores mucho mayores que uno periférico y las activaciones se descompensan con la profundidad. Normalizar por el grado a ambos lados hace que los autovalores de Â queden acotados en [−1, 1], y con los auto-lazos el mayor queda en 1: la propagación **no amplifica**, y por eso la red se puede apilar sin que las activaciones exploten.

Esa misma propiedad explica el límite. Aplicar Â repetidamente es un promediado iterado, y un promediado iterado sobre un grafo conexo converge a un punto fijo donde todos los nodos comparten la misma representación, proporcional al autovector dominante. Es el **sobre-suavizado**: con muchas capas, la señal que distingue a un nodo de otro se disuelve y la exactitud cae. De ahí un hecho que sorprende a quien viene de las CNN —donde más profundidad casi siempre ayuda—: las GNN de paso de mensajes suelen rendir mejor con **dos o tres capas**, y ese es el número que este laboratorio explora. El campo receptivo crece muy rápido de todos modos: dos capas ya cubren los vecinos a distancia dos, que en una red de citas puede ser una fracción notable del grafo.

### La fuga de datos en un grafo no es como en una tabla

Este es el punto donde el protocolo del repositorio se vuelve más delicado, y merece atención porque el error es invisible.

Cora se estudia en régimen **transductivo**: el grafo completo —todos los nodos y todas las aristas— está disponible durante el entrenamiento, y lo que se divide en `train`, `validation` y `test` son las **etiquetas**, no los nodos. Solo se calcula la pérdida sobre los nodos etiquetados como entrenamiento. Que un nodo de test participe en el paso de mensajes no es una fuga: es la definición del problema, y así se compara con la literatura.

Lo que sí es una fuga es usar sus **etiquetas** de cualquier forma —directa o indirectamente— antes de la evaluación final. Y hay dos vías sutiles por las que se cuela. La primera es la selección: parar el entrenamiento o elegir arquitectura mirando la exactitud de test es, aquí igual que en cualquier otro laboratorio, contaminar la estimación. La segunda es más específica de los grafos: cualquier característica derivada del grafo que incorpore etiquetas de otros nodos —por ejemplo, «proporción de vecinos de la clase X»— transporta las etiquetas de test al conjunto de entrada por la puerta de atrás.

En el régimen **inductivo**, que es el que GraphSAGE hace posible, las reglas cambian: los nodos de evaluación no existen durante el entrenamiento y deben eliminarse del grafo junto con sus aristas. Es más exigente y más parecido al uso real —clasificar una publicación nueva que acaba de aparecer—, y ambos regímenes no son comparables entre sí. Declarar cuál se está usando forma parte del reporte, porque una exactitud transductiva y una inductiva no miden lo mismo.

Por último, una comparación que este laboratorio pide y que conviene entender: un **MLP que ignore las aristas**, alimentado solo con los atributos de texto de cada publicación. Si la GNN no lo supera con claridad, la estructura de citas no estaba aportando información y toda la maquinaria de paso de mensajes es complejidad sin retorno. Es el equivalente en grafos de la línea base honesta.

## Protocolo científico

- Ajustar transformaciones, vocabulario, normalización y selección de variables solo con `train`.
- Usar `validation` para arquitectura, hiperparámetros, checkpoint y umbrales.
- Evaluar `test` una vez, después de congelar las decisiones.
- Comparar contra **MLP sin aristas**.
- Reportar variación entre semillas e intervalos de confianza; una métrica puntual no expresa toda la incertidumbre.

## Riesgos de interpretación

Usa las máscaras públicas fijas de train, validación y test.

El dataset refleja su proceso de recolección y no representa automáticamente otros períodos, países o poblaciones. Una asociación predictiva no demuestra causalidad.

## Pregunta crítica

> ¿Cuánto aporta la estructura de citaciones?

> Las referencias apuntan a las obras; no se reproduce su contenido, la redacción es original.

## 🔗 Referencias

- Hamilton — *Graph Representation Learning* (Morgan & Claypool, 2020) — texto de referencia sobre embeddings de grafos, paso de mensajes y GNN.
- Kipf & Welling (2017), *Semi-Supervised Classification with Graph Convolutional Networks*, ICLR — la GCN y la normalización simétrica de la adyacencia usada en este laboratorio.
- Hamilton, Ying & Leskovec (2017), *Inductive Representation Learning on Large Graphs (GraphSAGE)*, NeurIPS — agregación por muestreo de vecinos y aprendizaje inductivo.
- Veličković et al. (2018), *Graph Attention Networks*, ICLR — atención sobre vecinos para ponderar mensajes de forma aprendida.
- Fuente del dataset: https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.datasets.Planetoid.html
- Consulte `docs/experiment-protocol.md`, `docs/reproducibility.md` y `docs/ethics-and-licenses.md`.

<!-- nav-bottom -->
## 🧭 Navegación del recorrido

| ⬅️ Laboratorio anterior | 🏠 Índice | Laboratorio siguiente ➡️ |
|---|:---:|---|
| [🎨 GAN generativa](../../labs/08_gan_generation/README.md) | [Las 31 rutas](../../parts/README.md) | [🕹️ DQN para inventario con demanda real](../../labs/10_dqn_reinforcement/README.md) |

**En este laboratorio:** [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🟣 [Parte 3 — Familias especializadas: generar, decidir, relacionar](../../parts/03-familias-especializadas.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/09_gnn_graphs/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
