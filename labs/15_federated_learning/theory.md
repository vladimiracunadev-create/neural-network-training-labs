# Teoría — Aprendizaje federado por participante

<!-- nav-top -->
> 🧭 **Ruta 16 / 31** · 🟠 [Parte 4 — Entrenar mejor, más barato y sin centralizar datos](../../parts/04-entrenamiento-eficiente.md)
>
> [⬅️ ⚗️ Destilación de conocimiento](../../labs/14_knowledge_distillation/theory.md) · [🏠 Índice de rutas](../../parts/README.md) · [∂ Backpropagation manual ➡️](../../labs/16_backpropagation_manual/theory.md)
>
> [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

## Propósito

Aplicar FedAvg usando participantes reales como clientes naturales.

## Idea central

Este laboratorio estudia **agregación federada de clientes reales** usando `uci_har_subjects`, un dataset público real procedente de UCI. El aprendizaje federado responde a una tensión práctica: los datos viven distribuidos entre muchos dispositivos o personas (aquí, cada participante del estudio de actividad humana), y por razones de privacidad, ancho de banda o regulación no se pueden centralizar en un servidor. La pregunta es cómo entrenar un modelo global sin mover los datos crudos fuera de su origen.

La respuesta que implementa el laboratorio es **FedAvg** (Federated Averaging). En cada ronda, el servidor envía el modelo actual a un conjunto de clientes; cada cliente entrena localmente unas cuantas épocas con *sus propios* datos y devuelve solo los pesos resultantes (no los datos). El servidor promedia esos pesos, ponderando por la cantidad de datos de cada cliente, y obtiene el nuevo modelo global. Se repite el ciclo. Lo que viaja por la red son parámetros, no ejemplos, lo que reduce la exposición de información sensible.

Una decisión metodológica importante es usar el **identificador real de cada sujeto como cliente natural**, en lugar de trocear los datos al azar. Esto preserva la heterogeneidad genuina: cada persona camina, se sienta y sube escaleras de forma ligeramente distinta, por lo que las distribuciones locales son **no-IID** (no idénticamente distribuidas). Esa heterogeneidad es precisamente lo que hace difícil el aprendizaje federado, y estudiarla con clientes reales es más honesto que fabricar particiones artificiales. La pregunta crítica —qué clientes quedan perjudicados por la agregación— apunta a que un promedio global puede favorecer a la mayoría y degradar a los participantes atípicos.

## Fundamento matemático

Hay K clientes; el cliente k posee n_k ejemplos y el total es n = Σₖ n_k. Cada cliente define una pérdida local promedio sobre sus datos, F_k(w). El objetivo global es la pérdida ponderada por tamaño de dataset:

  F(w) = Σₖ (n_k / n) · F_k(w)

FedAvg optimiza F(w) sin acceder a los datos crudos. En la ronda t, partiendo del modelo global w_t:

1. El servidor envía w_t a los clientes seleccionados.
2. Cada cliente hace E épocas de descenso de gradiente local, w_k ← w_k − η · ∇ F_k(w_k), partiendo de w_k = w_t, y obtiene w_k^{(t+1)}.
3. El servidor agrega por promedio ponderado:

  w_{t+1} = Σₖ (n_k / n) · w_k^{(t+1)}

La ponderación n_k/n hace que un cliente con más datos influya proporcionalmente más en el modelo global, lo que equivale a tratar por igual a cada *ejemplo* aunque estén repartidos entre clientes. Un caso límite ilumina la fórmula: si cada cliente diera un solo paso de gradiente completo (E = 1, batch = todos sus datos), el promedio de sus actualizaciones locales coincide exactamente con un paso de gradiente sobre F(w) centralizada. Con E > 1, los clientes se alejan localmente antes de promediar; ese **desvío del cliente** (client drift) es mayor cuanto más no-IID son los datos, y explica por qué FedAvg puede converger más lento o de forma menos estable que el entrenamiento centralizado.

Por eso la línea base natural es el entrenamiento centralizado, y una métrica clave es la *dispersión* de la exactitud entre clientes (client_accuracy_std): no basta con una buena media global si algunos participantes quedan sistemáticamente mal servidos. La formulación conecta cuatro elementos: representación de entrada, función del modelo, función de pérdida local F_k y regla de actualización (SGD local + agregación con Σ). El notebook muestra las dimensiones de los tensores y conserva la misma implementación que el script de terminal.

### Por qué los datos no-IID rompen el promedio

FedAvg parece inofensivo: cada cliente entrena localmente y el servidor promedia los pesos. Funciona bien cuando los clientes tienen datos parecidos, y se degrada exactamente en la medida en que no lo son. Conviene ver por qué.

Si todos los clientes tuvieran datos extraídos de la misma distribución (caso **IID**), el gradiente local sería un estimador insesgado del gradiente global y promediar actualizaciones sería casi equivalente a entrenar de forma centralizada. Pero en este dataset cada cliente es una **persona**, y las personas no son intercambiables: una realiza sobre todo actividades sedentarias, otra camina mucho, y sus etiquetas están desbalanceadas de formas distintas. Cada cliente optimiza entonces un objetivo local F_k(θ) cuyo mínimo está en un lugar distinto del mínimo global F(θ) = Σ_k (n_k/n)·F_k(θ).

La consecuencia se llama **desvío del cliente**: con E épocas locales, cada modelo se aleja hacia *su* óptimo antes de que el servidor promedie. Cuanto mayor es E, más lejos llegan y más se contradicen las direcciones al juntarlas; el modelo promediado puede quedar peor que cualquiera de los locales, y en el caso extremo el entrenamiento oscila sin converger. Ahí está el compromiso central de FedAvg, y es contraintuitivo: **más cómputo local no es mejor**. Aumentar E ahorra rondas de comunicación —que es el recurso caro— pero incrementa el desvío. El número de épocas locales es, por tanto, un hiperparámetro experimental, no un detalle de implementación.

El promedio, además, es **ponderado por el tamaño del cliente**:

θ_(t+1) = Σ_k (n_k / n) · θ_k^(t),

lo que reproduce el objetivo global correcto pero da voz proporcional al número de ejemplos. Un participante con muchos datos domina el modelo resultante, y uno con pocos apenas influye: es una decisión de diseño con consecuencias de equidad, no solo de optimización.

### Qué hay que medir además del promedio

La métrica global de un modelo federado esconde justo lo que el enfoque debería vigilar. Un modelo puede alcanzar un buen promedio y funcionar muy mal para un subconjunto de participantes —típicamente aquellos cuya distribución se aleja más de la mayoritaria—, y ese fallo es invisible en la cifra agregada.

Por eso este laboratorio reporta el desempeño **por participante** y no solo el global. Las cifras que conviene mirar son la dispersión entre clientes, el peor cliente y la brecha entre el mejor y el peor: son la traducción operativa de «¿a quién le funciona esto?». Una mejora del promedio conseguida a costa de empeorar al peor cliente rara vez es aceptable en un despliegue real.

La comparación honesta necesita además dos referencias. Arriba, el **modelo centralizado** entrenado con todos los datos juntos: marca el techo, y la diferencia con él es el precio de no centralizar. Abajo, los **modelos puramente locales**, uno por cliente entrenado solo con sus datos: si el federado no los supera, el participante no gana nada colaborando, y la propuesta se cae. Entre esos dos límites es donde el resultado significa algo.

### Lo que la federación protege y lo que no

Es importante ser preciso en esto, porque es la motivación del enfoque y también su malentendido más común. Que los datos crudos no salgan del dispositivo **no equivale a privacidad**. Lo que se transmite —actualizaciones de pesos o gradientes— es una función de los datos y filtra información sobre ellos: se han demostrado ataques de inferencia de pertenencia, que determinan si un ejemplo concreto estuvo en el entrenamiento, y ataques de reconstrucción que recuperan aproximaciones de las entradas a partir de los gradientes.

Las defensas existen y tienen un costo explícito. La **privacidad diferencial** añade ruido calibrado a las actualizaciones y ofrece una garantía formal con un presupuesto ε, a cambio de exactitud. La **agregación segura** impide que el servidor vea las actualizaciones individuales y solo le permite obtener la suma, a cambio de protocolo criptográfico y coordinación. Ninguna es gratis, y este laboratorio no las implementa: se limita a mostrar el mecanismo de FedAvg, y por eso su alcance debe declararse tal cual —una demostración del algoritmo, no un sistema con garantías de privacidad—.

El otro costo que conviene contabilizar es la **comunicación**. Cada ronda transmite el modelo completo en ambos sentidos, así que el tráfico total es del orden de 2 · |θ| · K · R bytes para K clientes y R rondas. Con modelos grandes es el cuello de botella dominante, muy por encima del cómputo, y es lo que motiva las técnicas de compresión y cuantización de actualizaciones. Reportar el número de rondas hasta alcanzar cierta exactitud es, en federado, tan relevante como reportar la exactitud misma.

## Protocolo científico

- Ajustar transformaciones, vocabulario, normalización y selección de variables solo con `train`.
- Usar `validation` para arquitectura, hiperparámetros, checkpoint y umbrales.
- Evaluar `test` una vez, después de congelar las decisiones.
- Comparar contra **Entrenamiento centralizado**.
- Reportar variación entre semillas e intervalos de confianza; una métrica puntual no expresa toda la incertidumbre.

## Riesgos de interpretación

No crea clientes espaciales artificiales; conserva identificadores reales de sujetos.

El dataset refleja su proceso de recolección y no representa automáticamente otros períodos, países o poblaciones. Una asociación predictiva no demuestra causalidad.

## Pregunta crítica

> ¿Qué clientes quedan perjudicados por la agregación?

## 🔗 Referencias

> Las referencias apuntan a las obras; no se reproduce su contenido, la redacción es original.

- Kairouz et al. (2021), *Advances and Open Problems in Federated Learning*, Foundations and Trends in Machine Learning — monografía de referencia sobre el marco federado, datos no-IID, privacidad y problemas abiertos.
- McMahan et al. (2017), *Communication-Efficient Learning of Deep Networks from Decentralized Data (FedAvg)*, AISTATS — artículo que introduce el algoritmo FedAvg implementado en este laboratorio.
- Fuente del dataset: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones — **Human Activity Recognition Using Smartphones** (UCI Machine Learning Repository, CC BY 4.0); procedencia, versión y SHA-256 en el registro de fuentes, entrada `uci-human-activity-recognition-smartphones` — esta clase la usa para aplicar FedAvg tomando a los participantes reales del estudio como clientes naturales, sin inventar particiones.
- Consulte `docs/experiment-protocol.md`, `docs/reproducibility.md` y `docs/ethics-and-licenses.md`.

<!-- nav-bottom -->
## 🧭 Navegación del recorrido

| ⬅️ Laboratorio anterior | 🏠 Índice | Laboratorio siguiente ➡️ |
|---|:---:|---|
| [⚗️ Destilación de conocimiento](../../labs/14_knowledge_distillation/README.md) | [Las 31 rutas](../../parts/README.md) | [∂ Backpropagation manual](../../labs/16_backpropagation_manual/README.md) |

**En este laboratorio:** [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🟠 [Parte 4 — Entrenar mejor, más barato y sin centralizar datos](../../parts/04-entrenamiento-eficiente.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/15_federated_learning/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
