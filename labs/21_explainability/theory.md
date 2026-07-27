# Teoría — Explicabilidad

## Propósito

Explicar predicciones con Integrated Gradients y permutación.

## Idea central

Este laboratorio estudia **atribución de características** usando `adult_census`, un dataset público real procedente de UCI.

Un modelo puede acertar mucho y, aun así, ser una caja negra: predice si el ingreso anual supera cierto umbral, pero no dice *por qué*. La explicabilidad busca responder esa pregunta asignando a cada característica de entrada una **atribución**: cuánto empujó ese atributo la predicción hacia una clase u otra. En un dataset con variables demográficas y laborales como `adult_census`, esa atribución no es un lujo académico —es un requisito de auditoría, porque una explicación revela si el modelo se apoya en señales legítimas o en correlaciones espurias con atributos sensibles.

El laboratorio contrasta métodos locales y globales. **Integrated Gradients** explica una predicción concreta (por qué *este* individuo fue clasificado así) integrando el gradiente del modelo a lo largo de un camino desde una entrada de referencia. La **importancia por permutación** mide, en cambio, cuánto se degrada el rendimiento global cuando se destruye la información de una variable barajándola. Comparamos siempre contra una regresión logística interpretable, cuyo peso por variable ofrece un patrón de referencia de qué debería "importar".

## Fundamento matemático

Atribución por integración del gradiente entre baseline y entrada.

**Integrated Gradients** parte de una idea de teoría de juegos y cálculo: para explicar la salida f(x) de un modelo, comparamos x con una entrada de referencia (*baseline*) x′ que representa "ausencia de información" (por ejemplo, el vector de medias o de ceros). La contribución de la característica i es la integral del gradiente parcial ∂f/∂xᵢ a lo largo del segmento recto que va de x′ a x:

IGᵢ(x) = (xᵢ − x′ᵢ) · ∫₀¹ ∂f(x′ + α·(x − x′)) / ∂xᵢ · dα

Intuitivamente recorremos el camino de x′ hasta x en pasos infinitesimales, acumulamos cuánto responde la salida a la variable i en cada punto del trayecto, y lo multiplicamos por cuánto cambió esa variable, (xᵢ − x′ᵢ). Usar la integral del gradiente —y no el gradiente en un solo punto— evita el problema de la *saturación*: cuando f está en una zona plana, el gradiente local es casi cero aunque la variable sea decisiva, y la integral a lo largo del camino sí captura su efecto. En la práctica la integral se aproxima por una suma de Riemann con m pasos: IGᵢ ≈ (xᵢ − x′ᵢ)·(1/m)·Σₖ ∂f(x′ + (k/m)(x − x′))/∂xᵢ. El método cumple dos propiedades deseables: *sensibilidad* (una variable que cambia la predicción recibe atribución no nula) y *completitud* (la suma de las atribuciones iguala la diferencia f(x) − f(x′)).

La **importancia por permutación** es un método agnóstico al modelo y de alcance global. Se mide una métrica de referencia s (por ejemplo, exactitud o AUC) sobre datos de validación; luego, para la característica j, se baraja aleatoriamente su columna —rompiendo su relación con la etiqueta pero conservando su distribución marginal— y se vuelve a medir s_j^perm. La importancia es la caída media Iⱼ = s − 𝔼[s_j^perm], normalmente promediada sobre varias permutaciones para estimar su variabilidad. La lógica es contrafactual: si al destruir la información de j el rendimiento no cae, el modelo no la estaba usando; si se desploma, esa variable era load-bearing.

Ambos enfoques tienen límites que el laboratorio hace explícitos. Las atribuciones dependen de decisiones —el baseline elegido en Integrated Gradients, la correlación entre variables en la permutación— y pueden ser **inestables**: pequeñas perturbaciones de la entrada, o variables muy correlacionadas entre sí, pueden repartir el crédito de formas distintas sin que la predicción cambie. Una explicación describe cómo *actúa* el modelo, no un mecanismo causal del mundo; confundir atribución con causalidad es el error de interpretación central que hay que evitar.

## Protocolo científico

- Ajustar transformaciones, vocabulario, normalización y selección de variables solo con `train`.
- Usar `validation` para arquitectura, hiperparámetros, checkpoint y umbrales.
- Evaluar `test` una vez, después de congelar las decisiones.
- Comparar contra **Regresión logística interpretable**.
- Reportar variación entre semillas e intervalos de confianza; una métrica puntual no expresa toda la incertidumbre.

## Riesgos de interpretación

Incluye advertencias éticas sobre variables demográficas y sesgo.

El dataset refleja su proceso de recolección y no representa automáticamente otros períodos, países o poblaciones. Una asociación predictiva no demuestra causalidad.

## Pregunta crítica

> ¿La explicación es estable ante pequeñas perturbaciones?

## 🔗 Referencias

> Las referencias apuntan a las obras; no se reproduce su contenido, la redacción es original.

- Molnar — *Interpretable Machine Learning* (libro online, molnar.github.io/interpretable-ml-book) — referencia integral sobre métodos de interpretabilidad, incluidos importancia por permutación, LIME y SHAP.
- Ribeiro, Singh y Guestrin (2016), *"Why Should I Trust You?": Explaining the Predictions of Any Classifier (LIME)*, KDD — explicaciones locales mediante modelos sustitutos interpretables.
- Lundberg y Lee (2017), *A Unified Approach to Interpreting Model Predictions (SHAP)*, NeurIPS — marco unificado de atribución basado en valores de Shapley.
- Selvaraju et al. (2017), *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*, ICCV — mapas de atribución basados en gradientes para redes convolucionales.
- Fuente del dataset: https://archive.ics.uci.edu/dataset/2/adult
- Consulte `docs/experiment-protocol.md`, `docs/reproducibility.md` y `docs/ethics-and-licenses.md`.
