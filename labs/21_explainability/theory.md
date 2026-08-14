# Teoría — Explicabilidad

<!-- nav-top -->
> 🧭 **Ruta 22 / 31** · ⚫ [Parte 6 — Confiar en el modelo y sacarlo del cuaderno](../../parts/06-confianza-y-despliegue.md)
>
> [⬅️ 🔄 Aumento de datos](../../labs/20_data_augmentation/theory.md) · [🏠 Índice de rutas](../../parts/README.md) · [🎯 Incertidumbre y calibración ➡️](../../labs/22_uncertainty_calibration/theory.md)
>
> [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

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

### Los axiomas que hacen de Integrated Gradients algo más que un gradiente

El gradiente por sí solo ya parece una explicación —dice cuánto cambia la salida si se mueve una entrada— y falla por dos motivos concretos. Primero, es **local**: describe la pendiente en un punto, no el efecto de haber llegado hasta ahí. Segundo, sufre **saturación**: si una característica ya llevó la predicción a un extremo, su gradiente ahí es casi cero, y una atribución que la declare irrelevante contradice el hecho de que fue determinante.

Integrated Gradients corrige ambas cosas integrando el gradiente a lo largo del camino desde una **referencia** x′ hasta la entrada x:

IG_i(x) = (x_i − x′_i) · ∫₀¹ ∂F( x′ + α·(x − x′) ) / ∂x_i · dα,

que en la práctica se aproxima con una suma de Riemann de m pasos (m entre 50 y 300 suele bastar). Su valor es que satisface dos propiedades demostrables. La **completitud**: la suma de todas las atribuciones es exactamente F(x) − F(x′), de modo que el presupuesto explicativo cuadra y no queda efecto sin repartir —es la comprobación numérica que conviene hacer siempre, porque una suma que no cuadra delata que m es demasiado pequeño—. Y la **sensibilidad**: si dos entradas difieren en una sola característica y producen predicciones distintas, esa característica recibe atribución no nula, algo que el gradiente puro no garantiza.

La elección de la **referencia** x′ no es un detalle técnico: define la pregunta que se está respondiendo. Las atribuciones explican la diferencia respecto de esa referencia, así que con x′ = 0 se responde «¿por qué esta predicción y no la de una entrada nula?», y con x′ = la media del conjunto de entrenamiento, «¿por qué esta y no la de un caso típico?». Son preguntas distintas y producen atribuciones distintas. Declarar cuál se usó es parte del reporte; omitirlo hace la explicación ininterpretable.

### La importancia por permutación mide otra cosa

Conviene no confundir las dos técnicas del laboratorio, porque responden preguntas diferentes y a menudo se citan como si fueran intercambiables.

Integrated Gradients es **local**: explica una predicción concreta. La importancia por permutación es **global**: baraja los valores de una característica en todo el conjunto y mide cuánto se degrada la métrica,

imp_j = métrica(D) − métrica(D con la columna j permutada),

lo que responde «¿cuánto depende el modelo de esta variable en promedio?». Una variable puede ser globalmente poco importante y decisiva para un caso particular; ambas cosas son ciertas a la vez.

La permutación tiene además un fallo conocido que hay que declarar: con variables **correlacionadas**, permutar una crea combinaciones imposibles —una edad de 20 años con 30 de experiencia laboral—, el modelo evalúa fuera de la distribución en la que fue entrenado y la degradación resultante mezcla dos efectos, la pérdida de información y la extrapolación. El resultado tiende a **repartir** la importancia entre variables correlacionadas, subestimando a cada una. Por eso conviene mirar antes la matriz de correlación y, si hay grupos, permutarlos en bloque.

Y como se calcula sobre un conjunto y una métrica concretos, hay una regla que se incumple con frecuencia: la importancia debe calcularse sobre datos **no usados para entrenar**. Sobre `train` mide de qué se apoyó el modelo para memorizar, que no es lo mismo que de qué depende su capacidad de generalizar.

### Qué no demuestra una explicación

Es la parte más importante de esta ruta y la que suele omitirse.

Una atribución alta significa que **el modelo** usó esa característica, no que la característica **cause** el fenómeno. Si el modelo aprendió a apoyarse en un proxy —un código postal que correlaciona con ingresos, una marca de agua que correlaciona con la clase— la explicación señalará fielmente ese proxy, y confundirlo con una causa lleva a decisiones erróneas sobre el mundo. La explicación audita el modelo; la causalidad requiere intervención, no observación.

Además, las atribuciones son **inestables**: métodos distintos aplicados al mismo modelo y la misma entrada producen rankings distintos, y pequeñas perturbaciones de la entrada pueden alterarlos notablemente. De ahí dos exigencias prácticas: contrastar al menos dos métodos —que es la razón de que este laboratorio use dos— y desconfiar de cualquier conclusión que dependa de las diferencias finas del orden.

Por último, una explicación **no valida** un modelo. Un modelo con métricas malas y explicaciones plausibles sigue siendo malo; uno con métricas buenas y explicaciones incómodas puede estar revelando un sesgo real de los datos, que es información valiosa y no un fallo del método. En un dominio como el de este dataset —predicción de ingresos a partir de datos censales— la explicación sirve sobre todo para detectar apoyos en variables sensibles o en sus proxies, y esa detección obliga a una decisión humana, no a un ajuste de hiperparámetros.

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

<!-- nav-bottom -->
## 🧭 Navegación del recorrido

| ⬅️ Laboratorio anterior | 🏠 Índice | Laboratorio siguiente ➡️ |
|---|:---:|---|
| [🔄 Aumento de datos](../../labs/20_data_augmentation/README.md) | [Las 31 rutas](../../parts/README.md) | [🎯 Incertidumbre y calibración](../../labs/22_uncertainty_calibration/README.md) |

**En este laboratorio:** [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

⚫ [Parte 6 — Confiar en el modelo y sacarlo del cuaderno](../../parts/06-confianza-y-despliegue.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/21_explainability/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
