# Teoría — Proyecto final: churn de telecomunicaciones

<!-- nav-top -->
> 🧭 **Ruta 25 / 31** · ⚫ [Parte 6 — Confiar en el modelo y sacarlo del cuaderno](../../parts/06-confianza-y-despliegue.md)
>
> [⬅️ 📦 Exportación e inferencia](../../labs/23_model_export_and_inference/theory.md) · [🏠 Índice de rutas](../../parts/README.md) · [🔧 Fine-tuning eficiente de transformer ➡️](../../advanced_labs/25_transformer_finetuning/theory.md)
>
> [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md)
<!-- /nav-top -->

## Propósito

Resolver de extremo a extremo un problema real de abandono de clientes con documentación, evaluación y despliegue.

## Idea central

Este laboratorio estudia **proyecto integral de churn** usando `iranian_churn`, un dataset público real procedente de UCI.

El *churn* —abandono de clientes— es un problema de negocio antes que un problema de aprendizaje: una operadora quiere anticipar qué clientes dejarán el servicio para intervenir con retención. Traducirlo a un modelo obliga a recorrer todo el ciclo de vida de un proyecto de ML: entender los datos y su procedencia, definir la métrica que importa, construir líneas base honestas, entrenar y calibrar, elegir un umbral de decisión ligado al *costo* de los errores, y documentar el resultado para que sea auditable y desplegable. Este capstone integra todo lo aprendido en los laboratorios anteriores sobre un caso real: 3.150 clientes de una empresa iraní de telecomunicaciones seguidos durante 12 meses.

Lo distintivo de un proyecto end-to-end es que la exactitud bruta rara vez es la meta. El churn es un problema **desbalanceado** (los que se van son minoría) y con **costos asimétricos**: no cuesta lo mismo dejar escapar a un cliente que se iba (falso negativo, se pierde su valor) que ofrecer una promoción a alguien que se quedaba igual (falso positivo, gasto innecesario). Por eso el laboratorio insiste en métricas sensibles al desbalance, en calibración de probabilidades y en la selección de umbral como decisión de negocio, comparando siempre contra líneas base sólidas: regresión logística y gradient boosting.

## Fundamento matemático

Clasificación, calibración, selección de umbral y costo de errores.

El modelo produce una probabilidad de abandono p = P(churn | x) para cada cliente, pero la *decisión* de actuar requiere un **umbral** τ: se interviene si p ≥ τ. La elección de τ no es un detalle técnico, sino donde entra la economía del problema. Cada resultado tiene un costo: un verdadero positivo detectado permite una acción de retención; un falso negativo (τ demasiado alto) deja escapar clientes; un falso positivo (τ demasiado bajo) malgasta recursos. Si asignamos costos c_FN y c_FP a cada tipo de error, el umbral óptimo minimiza el costo esperado y, bajo el análisis clásico, satisface una relación de la forma τ\* = c_FP / (c_FP + c_FN): cuanto más caro es dejar escapar a un cliente (c_FN grande), más bajo conviene poner el umbral para capturar a más candidatos. Este umbral se elige en **validación**, nunca en test.

Como el umbral depende de la probabilidad, esa probabilidad debe ser **confiable**, y aquí reaparece la calibración del laboratorio 22: un modelo sobreconfiado desplaza el punto de operación y distorsiona el análisis de costos. Por eso, antes de fijar τ, conviene recalibrar (p. ej. con temperature scaling o Platt scaling) para que p ≈ frecuencia real de churn. Para evaluar el modelo con independencia del umbral se usan métricas basadas en el *ranking*: el **ROC-AUC** mide la probabilidad de ordenar correctamente un par (cliente que abandona, cliente que se queda), mientras que el **PR-AUC** (precisión–recall) es más informativo en datos desbalanceados porque se centra en la clase positiva minoritaria y no se deja "inflar" por la abundancia de negativos. La **balanced accuracy** —media de la sensibilidad y la especificidad— corrige el sesgo de la exactitud simple cuando las clases están desequilibradas.

La cadena de razonamiento del capstone es, entonces: entrenar un clasificador → recalibrar sus probabilidades → medir su capacidad de ordenamiento con AUC/PR-AUC de forma independiente del umbral → traducir esa capacidad en una política de decisión eligiendo τ según los costos del negocio en validación → y solo entonces reportar el desempeño una única vez en test. La contribución matemática no está en un algoritmo nuevo, sino en *encadenar correctamente* clasificación, calibración, coste y umbral para que el número final sea una decisión responsable y no una métrica aislada.

## Protocolo científico

- Ajustar transformaciones, vocabulario, normalización y selección de variables solo con `train`.
- Usar `validation` para arquitectura, hiperparámetros, checkpoint y umbrales.
- Evaluar `test` una vez, después de congelar las decisiones.
- Comparar contra **Regresión logística y Gradient Boosting**.
- Reportar variación entre semillas e intervalos de confianza; una métrica puntual no expresa toda la incertidumbre.

## Riesgos de interpretación

3.150 clientes recolectados aleatoriamente de la base de una empresa iraní de telecomunicaciones durante 12 meses.

El dataset refleja su proceso de recolección y no representa automáticamente otros períodos, países o poblaciones. Una asociación predictiva no demuestra causalidad.

## Pregunta crítica

> ¿Cómo convertir resultados en una decisión responsable?

## 🔗 Referencias

> Las referencias apuntan a las obras; no se reproduce su contenido, la redacción es original.

- Géron — *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* (3.ª ed., O'Reilly, 2022), cap. 2 — recorrido completo de un proyecto de ML de punta a punta, del marco del problema al despliegue.
- Huyen — *Designing Machine Learning Systems* (O'Reilly, 2022) — diseño de sistemas de ML en producción: métricas de negocio, monitorización y despliegue responsable.
- Kuhn y Johnson — *Applied Predictive Modeling* (Springer, 2013) — modelado predictivo aplicado: preprocesamiento, evaluación con clases desbalanceadas y selección de umbral.
- Fuente del dataset: https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset
- Consulte `docs/experiment-protocol.md`, `docs/reproducibility.md` y `docs/ethics-and-licenses.md`.

<!-- nav-bottom -->
## 🧭 Navegación del recorrido

| ⬅️ Laboratorio anterior | 🏠 Índice | Laboratorio siguiente ➡️ |
|---|:---:|---|
| [📦 Exportación e inferencia](../../labs/23_model_export_and_inference/README.md) | [Las 31 rutas](../../parts/README.md) | [🔧 Fine-tuning eficiente de transformer](../../advanced_labs/25_transformer_finetuning/README.md) |

**En este laboratorio:** [📄 Guía](README.md) · **🧠 Teoría** · [🔬 Experimentos](experiments.md) · [📝 Evaluación](assessment.md) · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

⚫ [Parte 6 — Confiar en el modelo y sacarlo del cuaderno](../../parts/06-confianza-y-despliegue.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/24_capstone_real_project/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
