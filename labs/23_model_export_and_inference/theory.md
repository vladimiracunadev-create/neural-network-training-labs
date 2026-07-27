# Teoría — Exportación e inferencia

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
- Fuente del dataset: https://www.cs.toronto.edu/~kriz/cifar.html
- Consulte `docs/experiment-protocol.md`, `docs/reproducibility.md` y `docs/ethics-and-licenses.md`.
