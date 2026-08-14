# Teoría — DQN para inventario con demanda real

<!-- nav-top -->
> 🧭 **Ruta 11 / 31** · 🟣 [Parte 3 — Familias especializadas: generar, decidir, relacionar](../../parts/03-familias-especializadas.md)
>
> [⬅️ 🕸️ GNN sobre red de citas](../../labs/09_gnn_graphs/theory.md) · [🏠 Índice de rutas](../../parts/README.md) · [♻️ Transfer learning con mascotas ➡️](../../labs/11_transfer_learning/theory.md)
>
> [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

## Propósito

Aprender una política de reposición usando una secuencia de demanda observada en transacciones reales.

## Idea central

Este laboratorio estudia **valor de acciones con demanda histórica** usando `online_retail`, un dataset público real procedente de UCI.

El problema se plantea como **aprendizaje por refuerzo**: un agente observa el estado del inventario, elige cuánto reponer, y recibe una recompensa que penaliza tanto quedarse sin stock (ventas perdidas) como mantener inventario en exceso (coste de almacenamiento). No hay etiquetas de "acción correcta"; el agente debe descubrir una **política** —una regla que mapea estados a acciones— probando y observando consecuencias a lo largo del tiempo. La dificultad propia del refuerzo es que las decisiones tienen efectos diferidos: reponer poco hoy puede ahorrar coste ahora pero causar un quiebre de stock costoso mañana. El agente debe optimizar la recompensa *acumulada*, no la inmediata.

La pieza clave es aprender el **valor** de cada acción en cada estado: cuánta recompensa futura total cabe esperar si tomo esta acción y luego actúo bien. Con esa función de valor Q(s, a), la política óptima es trivial: en cada estado elegir la acción de mayor Q. **DQN** (Deep Q-Network) aproxima Q con una red neuronal, lo que permite manejar estados continuos (inventario, demanda reciente, posición temporal) sin tabular todos los casos. Lo distintivo de este laboratorio es que la demanda de cada paso no la genera un simulador arbitrario: proviene del **historial real** de transacciones de Online Retail, de modo que la política se enfrenta a la variabilidad genuina de la demanda.

## Fundamento matemático

El valor Q óptimo satisface la **ecuación de Bellman de optimalidad**, que expresa el valor de un par (s, a) como la recompensa inmediata más el mejor valor posible del estado siguiente, descontado:

    Q*(s, a) = 𝔼[ r + γ · max_{a′} Q*(s′, a′) | s, a ]

Aquí r es la recompensa recibida al ejecutar a en s, s′ es el estado siguiente, y γ ∈ [0, 1) es el **factor de descuento**, que fija cuánto pesan las recompensas futuras frente a las inmediatas (γ cercano a 1 → agente previsor). DQN entrena una red Q(s, a; θ) para satisfacer esta ecuación minimizando el **error de diferencia temporal (TD)** contra un objetivo (target):

    y = r + γ · max_{a′} Q_target(s′, a′; θ⁻)        ℒ(θ) = 𝔼_{(s,a,r,s′)∼𝒟}[ ( y − Q(s, a; θ) )² ]

Dos ingredientes hacen esto estable. Primero, la **repetición de experiencias** (replay buffer 𝒟): las transiciones (s, a, r, s′) se guardan y se muestrean en minibatches aleatorios, rompiendo la correlación temporal entre muestras consecutivas. Segundo, la **red objetivo** con parámetros θ⁻: una copia rezagada de θ que se actualiza cada cierto tiempo; usarla para calcular y evita que el objetivo persiga a la propia red en cada paso, lo que provocaría oscilaciones. La demanda de cada paso proviene del historial real, no de un generador. Conectando con los cuatro elementos: la **representación de entrada** es el vector de estado s (inventario, demanda reciente, tiempo); la **función del modelo** es la red Q que produce un valor por cada acción discreta de reposición; la **función de pérdida** es el error TD cuadrático de arriba; y la **regla de actualización** es descenso de gradiente, θ ← θ − η ∇_θ ℒ. El notebook muestra las dimensiones de los tensores y conserva la misma implementación que el script de terminal.

Este laboratorio incorpora dos mejoras estándar sobre el DQN original. **Double DQN** corrige la sobreestimación del valor: el operador max en el target tiende a elegir acciones cuyo Q está inflado por ruido, así que se **desacopla** la selección de la evaluación —la red en línea elige la acción y la red objetivo la valora: y = r + γ · Q_target(s′, argmax_{a′} Q(s′, a′; θ); θ⁻). **Dueling DQN** reorganiza la arquitectura separando el valor del estado V(s) de la **ventaja** A(s, a) de cada acción, y las recombina como Q(s, a) = V(s) + ( A(s, a) − (1/|𝒜|) Σ_{a′} A(s, a′) ). La resta de la ventaja media es un truco de identificabilidad que estabiliza el aprendizaje; la intuición es que en muchos estados el valor depende poco de la acción concreta, y estimar V(s) por separado hace el aprendizaje más eficiente.

Por último, el agente equilibra **exploración y explotación** típicamente con una política ε-greedy: con probabilidad ε toma una acción aleatoria (explora) y con probabilidad 1−ε toma argmax_a Q(s, a) (explota), reduciendo ε a lo largo del entrenamiento. Sin exploración suficiente, el agente podría fijar prematuramente una política de reposición subóptima.

## Protocolo científico

- Ajustar transformaciones, vocabulario, normalización y selección de variables solo con `train`.
- Usar `validation` para arquitectura, hiperparámetros, checkpoint y umbrales.
- Evaluar `test` una vez, después de congelar las decisiones.
- Comparar contra **Política de reposición periódica basada en demanda media histórica**.
- Reportar variación entre semillas e intervalos de confianza; una métrica puntual no expresa toda la incertidumbre.

## Riesgos de interpretación

La dinámica de inventario es un entorno educativo, pero la demanda diaria se construye exclusivamente desde transacciones reales de Online Retail.

El dataset refleja su proceso de recolección y no representa automáticamente otros períodos, países o poblaciones. Una asociación predictiva no demuestra causalidad.

## Pregunta crítica

> ¿La política es robusta a cambios en costo y demanda?

> Las referencias apuntan a las obras; no se reproduce su contenido, la redacción es original.

## 🔗 Referencias

- Sutton & Barto — *Reinforcement Learning: An Introduction* (2.ª ed., MIT Press) — texto canónico: procesos de decisión de Markov, ecuación de Bellman, Q-learning y equilibrio exploración–explotación.
- Mnih et al. (2015), *Human-level control through deep reinforcement learning (DQN)*, Nature — DQN con replay buffer y red objetivo, base del laboratorio.
- van Hasselt, Guez & Silver (2016), *Deep Reinforcement Learning with Double Q-learning*, AAAI — corrección de la sobreestimación desacoplando selección y evaluación.
- Wang et al. (2016), *Dueling Network Architectures for Deep Reinforcement Learning*, ICML — separación de valor de estado y ventaja de acción.
- Fuente del dataset: https://archive.ics.uci.edu/dataset/352/online+retail
- Consulte `docs/experiment-protocol.md`, `docs/reproducibility.md` y `docs/ethics-and-licenses.md`.

<!-- nav-bottom -->
## 🧭 Navegación del recorrido

| ⬅️ Laboratorio anterior | 🏠 Índice | Laboratorio siguiente ➡️ |
|---|:---:|---|
| [🕸️ GNN sobre red de citas](../../labs/09_gnn_graphs/README.md) | [Las 31 rutas](../../parts/README.md) | [♻️ Transfer learning con mascotas](../../labs/11_transfer_learning/README.md) |

**En este laboratorio:** [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🟣 [Parte 3 — Familias especializadas: generar, decidir, relacionar](../../parts/03-familias-especializadas.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/10_dqn_reinforcement/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
