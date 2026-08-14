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
