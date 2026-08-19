# Teoría — Exportación e inferencia

<!-- nav-top -->
> 🧭 **Ruta 24 / 31** · ⚫ [Parte 6 — Confiar en el modelo y sacarlo del cuaderno](../../parts/06-confianza-y-despliegue.md)
>
> [⬅️ 🎯 Incertidumbre y calibración](../../labs/22_uncertainty_calibration/theory.md) · [🏠 Índice de rutas](../../parts/README.md) · [🏁 Proyecto final: churn de telecomunicaciones ➡️](../../labs/24_capstone_real_project/theory.md)
>
> [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

## Propósito

Exportar ONNX, validar paridad y medir latencia por lotes.

## Idea central

Este laboratorio estudia **exportación y perfil de inferencia** usando `cifar10`, un dataset público real procedente de Torchvision / University of Toronto.

Un modelo entrenado en PyTorch vive dentro de un intérprete de Python y un grafo dinámico; llevarlo a producción exige *desacoplarlo* de ese entorno y empaquetarlo en un formato portátil que un motor de inferencia optimizado pueda ejecutar en servidores, móviles o dispositivos edge. **ONNX** (Open Neural Network Exchange) es ese formato intermedio: representa la red como un grafo estático de operadores estándar, independiente del framework de origen. Exportar consiste en trazar el modelo entrenado y volcarlo a ONNX; a partir de ahí un runtime como ONNX Runtime lo carga y lo ejecuta con optimizaciones de bajo nivel.

La pregunta central del laboratorio es de **compromisos**: al pasar de PyTorch eager al modelo exportado —y opcionalmente cuantizado— ganamos velocidad y reducimos tamaño, pero debemos verificar que no rompemos la corrección. Por eso el flujo es exportar, validar *paridad numérica* (que ONNX y PyTorch producen la misma salida) y luego perfilar latencia, throughput y tamaño sobre `cifar10`, comparando siempre contra la línea base de PyTorch eager.

## Fundamento matemático

Paridad numérica y costo de inferencia.

La **paridad numérica** es la condición de que el modelo exportado calcule esencialmente la misma función que el original. No se exige igualdad bit a bit —el reordenamiento de operaciones y las diferencias de kernels producen redondeos distintos en aritmética de punto flotante— sino que la diferencia esté acotada por una tolerancia. Se comprueba pasando las mismas entradas x por ambos grafos y midiendo, por ejemplo, la desviación máxima absoluta: máxₓ ‖f_PyTorch(x) − f_ONNX(x)‖_∞ < ε, con ε del orden de 10⁻⁵ para float32. Esta prueba es imprescindible porque un export puede "compilar" correctamente y aun así alterar la semántica (por dimensiones dinámicas mal trazadas, operadores no soportados o un modo train/eval equivocado): sin verificar paridad, un modelo desplegado podría dar predicciones sutilmente distintas a las validadas.

El **costo de inferencia** se descompone en tres magnitudes que suelen estar en tensión. La **latencia** es el tiempo de una sola pasada hacia adelante (relevante cuando importa responder rápido a cada petición); se reporta con estadísticos robustos como la mediana y percentiles (p50, p95) porque su distribución tiene colas. El **throughput** es el número de muestras procesadas por segundo, que crece al agrupar entradas en lotes: un lote de tamaño B amortiza los costos fijos por invocación y aprovecha el paralelismo del hardware, de modo que throughput ≈ B / latencia_lote, aunque a costa de mayor latencia por muestra individual. El **tamaño del modelo** (en MB) condiciona la memoria y el ancho de banda, decisivos en edge.

La **cuantización** es la palanca principal para reducir ambos, tamaño y latencia. Consiste en representar pesos y activaciones con enteros de baja precisión (típicamente INT8) en lugar de float32, mediante una función afín de escala s y punto cero z: r ≈ s·(q − z), donde r es el valor real y q su representación entera. Al usar 8 bits en vez de 32, la memoria se reduce hasta ≈4× y las operaciones enteras son más rápidas y eficientes energéticamente en el hardware adecuado. El precio es una pérdida de precisión numérica que puede degradar la exactitud; por eso s y z se calibran cuidadosamente y la cuantización se trata como otro compromiso a *medir*, no a asumir. La lectura global: exportación e inferencia optimizada solo son válidas si la paridad se verifica primero y el impacto en exactitud, latencia y tamaño se cuantifica de forma reproducible.

### Qué es un grafo exportado y por qué puede diferir

Exportar no es guardar los pesos: es capturar **la función** que el modelo calcula, en un formato que otro motor pueda ejecutar sin Python. El exportador recorre el modelo con una entrada de ejemplo, registra las operaciones que se ejecutan y las escribe como un grafo de operadores estandarizados junto con sus pesos.

De ahí se sigue la limitación fundamental del método: se captura **el camino que esa entrada recorrió**. Si el modelo tiene control de flujo dependiente de los datos —un `if` sobre un valor del tensor, un bucle cuya longitud depende de la entrada—, la rama no recorrida no queda en el grafo, y el modelo exportado calculará algo distinto para entradas que la habrían tomado. El exportador basado en `torch.export` detecta buena parte de estos casos, pero la regla práctica sigue vigente: un modelo pensado para exportarse se escribe sin lógica dependiente de los valores.

Hay dos diferencias más que explican por qué la paridad numérica nunca es exacta. La primera es que el motor de destino **fusiona operaciones** —convolución + normalización + ReLU en un solo núcleo— y reordena cálculos para ir más rápido; como la suma en punto flotante no es asociativa, (a + b) + c y a + (b + c) difieren en los últimos bits. La segunda es que las **dimensiones dinámicas** deben declararse al exportar: si no se marca el eje del lote como dinámico, el grafo queda fijado al tamaño de la entrada de ejemplo y fallará con cualquier otro.

Por eso la verificación no se hace con igualdad exacta sino con tolerancias, comparando la salida del modelo original y la del exportado sobre un conjunto de entradas:

máx |y_torch − y_onnx| ≤ atol + rtol · |y_torch|,

con valores típicos atol = 10⁻⁵ y rtol = 10⁻³ en float32. Y la comprobación debe hacerse sobre **varias** entradas, incluidos casos extremos y distintos tamaños de lote: verificar con una sola entrada no dice nada sobre las ramas que esa entrada no recorrió.

### Qué hace la cuantización y qué cuesta

La cuantización representa pesos y activaciones con enteros de 8 bits en lugar de flotantes de 32. El mapeo es afín y se define con dos números por tensor:

r ≈ S · (q − Z),   con   S = (r_max − r_min) / (q_max − q_min),

donde r es el valor real, q el entero, S la **escala** y Z el **punto cero**. El beneficio inmediato es un factor 4 de reducción de tamaño; el beneficio mayor, que la aritmética entera es más rápida y consume mucha menos energía en el hardware que la soporta —y que el movimiento de datos, que suele dominar la latencia, se reduce en la misma proporción—.

Las dos variantes se distinguen por cuándo se calculan esas constantes. En la **cuantización dinámica** —la que usa este laboratorio— los pesos se cuantizan una vez al exportar y las activaciones se cuantizan al vuelo en cada inferencia, midiendo su rango en el momento. No requiere datos ni reentrenamiento, y por eso es la opción por defecto. En la **cuantización estática** el rango de las activaciones se estima previamente pasando un conjunto de calibración representativo, lo que elimina el costo de medir en tiempo de ejecución y suele dar más velocidad, a cambio de necesitar datos.

La granularidad importa: una escala por tensor es simple pero pierde precisión si los canales tienen rangos muy distintos; una escala **por canal** en las capas convolucionales conserva bastante más exactitud por un costo mínimo. Y cuando la degradación es inaceptable, queda el **entrenamiento consciente de cuantización**, que simula el redondeo durante el entrenamiento para que la red aprenda a tolerarlo.

La pérdida de exactitud debe medirse, no suponerse, y en el mismo conjunto de evaluación que el modelo original. Reportar «se redujo el tamaño 4×» sin la métrica al lado es reportar la mitad del resultado.

### Cómo se mide la latencia sin engañarse

La medición de tiempos es donde se cometen los errores más fáciles de detectar y más frecuentes.

**Calentamiento.** Las primeras inferencias incluyen la inicialización de núcleos, la reserva de memoria y, en GPU, la compilación de kernels: pueden ser un orden de magnitud más lentas. Se descartan las primeras repeticiones antes de medir.

**Repetición y estadística.** Una sola medición captura ruido del sistema operativo. Se repite decenas de veces y se reporta la **mediana** y un percentil alto —p95 o p99— además de la media, porque en un servicio real lo que determina la experiencia es la cola de la distribución, no el promedio.

**Sincronización.** En GPU las operaciones son asíncronas: medir el tiempo sin sincronizar mide el encolado, no la ejecución, y produce cifras absurdamente buenas.

**Latencia y throughput no son lo mismo.** La latencia es el tiempo de una petición; el throughput, las peticiones por segundo. Aumentar el tamaño de lote mejora el segundo y **empeora** el primero, porque hay que esperar a llenar el lote. Cuál optimizar depende del caso de uso —interactivo o por lotes— y la respuesta no es la misma.

**Condiciones declaradas.** Hardware, número de hilos, versión del motor y tamaño de lote forman parte del resultado: una latencia sin esos datos no es comparable con ninguna otra.

Por último, el **contrato de inferencia** que el laboratorio genera es lo que hace utilizable el artefacto. Un modelo exportado sin su preprocesamiento es una función a la que nadie sabe qué darle: las mismas medias y desviaciones de normalización, el mismo orden y nombre de las variables, el mismo tamaño de imagen, el mismo mapeo de índices a clases. La causa más común de que un modelo funcione en el cuaderno y falle en producción no es el modelo: es un preprocesamiento reimplementado de forma ligeramente distinta al otro lado.

## Protocolo científico

- Ajustar transformaciones, vocabulario, normalización y selección de variables solo con `train`.
- Usar `validation` para arquitectura, hiperparámetros, checkpoint y umbrales.
- Evaluar `test` una vez, después de congelar las decisiones.
- Comparar contra **PyTorch eager**.
- Reportar variación entre semillas e intervalos de confianza; una métrica puntual no expresa toda la incertidumbre.

## Riesgos de interpretación

Incluye predicción, exportación y benchmark reproducible.

El dataset refleja su proceso de recolección y no representa automáticamente otros períodos, países o poblaciones. Una asociación predictiva no demuestra causalidad.

## Pregunta crítica

> ¿Qué compromisos existen entre tamaño, latencia y precisión?

## 🔗 Referencias

> Las referencias apuntan a las obras; no se reproduce su contenido, la redacción es original.

- Huyen — *Designing Machine Learning Systems* (O'Reilly, 2022), capítulos de despliegue y optimización de modelos — compromisos de latencia, throughput y compresión en producción.
- Jacob et al. (2018), *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference*, CVPR — esquema de cuantización INT8 con aritmética entera y su calibración.
- Documentación oficial de ONNX — especificación del formato de intercambio y conjunto de operadores: https://onnx.ai/
- Documentación oficial de PyTorch (`torch.onnx` / `torch.export`) — exportación de modelos y validación de paridad: https://pytorch.org/docs/stable/onnx.html
- Fuente del dataset: https://www.cs.toronto.edu/~kriz/cifar.html — **CIFAR-10** (University of Toronto, La fuente no declara una licencia); procedencia, versión y SHA-256 en el registro de fuentes, entrada `cifar-10` — esta clase la usa para exportar a ONNX, validar la paridad numérica y medir latencia por lotes sobre fotografías reales.
- Consulte `docs/experiment-protocol.md`, `docs/reproducibility.md` y `docs/ethics-and-licenses.md`.

<!-- nav-bottom -->
## 🧭 Navegación del recorrido

| ⬅️ Laboratorio anterior | 🏠 Índice | Laboratorio siguiente ➡️ |
|---|:---:|---|
| [🎯 Incertidumbre y calibración](../../labs/22_uncertainty_calibration/README.md) | [Las 31 rutas](../../parts/README.md) | [🏁 Proyecto final: churn de telecomunicaciones](../../labs/24_capstone_real_project/README.md) |

**En este laboratorio:** [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

⚫ [Parte 6 — Confiar en el modelo y sacarlo del cuaderno](../../parts/06-confianza-y-despliegue.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/23_model_export_and_inference/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
