# Teoría — Aprendizaje federado por participante

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
- Fuente del dataset: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
- Consulte `docs/experiment-protocol.md`, `docs/reproducibility.md` y `docs/ethics-and-licenses.md`.
