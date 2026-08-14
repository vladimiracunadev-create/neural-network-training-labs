# Evaluación — Segmentación semántica con U-Net

<!-- nav-top -->
> 🧭 **Ruta 27 / 31** · 🔬 [Parte 7 — Especializaciones avanzadas](../../parts/07-especializaciones-avanzadas.md)
>
> [⬅️ 🔧 Fine-tuning eficiente de transformer](../../advanced_labs/25_transformer_finetuning/assessment.md) · [🏠 Índice de rutas](../../parts/README.md) · [🎙️ Clasificación de audio con SpeechCommands ➡️](../../advanced_labs/27_audio_speechcommands/assessment.md)
>
> [📄 Guía](README.md) · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · **📝 Evaluación**
<!-- /nav-top -->

## Cómo se evalúa este laboratorio

No se evalúa el número final. Un modelo con una métrica alta obtenida mirando `test`, o sin compararse con nada, vale menos que uno modesto cuyo resultado se puede auditar. Lo que se califica es el **proceso**: si las particiones están limpias, si la decisión se tomó donde debía, si la conclusión distingue lo que se midió de lo que se supone.

## Evidencias que debes entregar

| Evidencia | Dónde vive | Por qué se pide |
|---|---|---|
| Dataset preparado y auditado | salida de `neural-labs audit` | Es la única prueba de que el resultado no está contaminado por una fuga entre particiones. |
| Notebook ejecutado sin celdas omitidas | `notebook.ipynb` | Una celda saltada suele ser justo la que rompía el argumento. |
| Comparación contra Máscara de clase mayoritaria | `metrics.json` y `baseline_metrics.json` | Sin línea base, una métrica no dice si el modelo aporta algo. |
| Al menos tres semillas de entrenamiento, o la justificación de por qué no | salida de `benchmark` | Una sola corrida no permite separar mejora de azar. |
| Análisis de errores y limitaciones | tu reporte | Saber *dónde* falla vale más que saber cuánto acierta. |
| Model card actualizada | `model_card.md` | Es lo que permite a otra persona saber cuándo **no** debería usar tu modelo. |

## Las preguntas, y qué se busca en tu respuesta

No se corrige la longitud de la respuesta, sino si demuestra comprensión. Debajo de cada pregunta está lo que una buena respuesta debería contener.

**1. Explica con tus palabras: Arquitectura encoder-decoder, conexiones skip, pérdida por píxel e intersección sobre unión.**

*Qué se busca:* Una buena respuesta conecta cuatro cosas —cómo se representa la entrada, qué calcula el modelo, qué mide la función de pérdida y cómo se actualizan los pesos— en vez de repetir la definición del libro. Si puedes explicarlo sin la fórmula delante, lo entendiste.

**2. ¿Qué información del dataset solo puede usarse durante el entrenamiento?**

*Qué se busca:* Se espera que nombres casos concretos: las estadísticas de normalización, el vocabulario, la selección de variables, los umbrales. Todo eso se ajusta **solo** con `train`; calcularlo sobre el conjunto completo es una fuga silenciosa que infla el resultado sin dar ningún aviso.

**3. ¿Por qué Máscara de clase mayoritaria es una comparación razonable para este problema?**

*Qué se busca:* Una buena respuesta explica qué captura la línea base y qué no, y por qué superarla —o no superarla— es informativo aquí. Si la línea base ya resuelve el problema, la conclusión correcta es que la red no estaba justificada.

**4. ¿Qué te dice `mean_iou` que no te dirían las otras métricas?**

*Qué se busca:* Cada métrica pondera distinto los errores. Se espera que expliques por qué esa es la que decide aquí y en qué situación sería una mala elección.

**5. ¿Qué cambiarías antes de usar este modelo fuera del laboratorio?**

*Qué se busca:* Aquí se evalúa el criterio, no la técnica: licencias y condiciones de uso de `oxford_iiit_pet_segmentation`, representatividad de la población, calibración de las probabilidades, vigilancia de la deriva, desempeño por subgrupo y supervisión humana. Un «funcionaría bien» sin condiciones se corrige como respuesta incompleta.

## La rúbrica, explicada

| Criterio | Insuficiente | Adecuado | Excelente | Peso |
|---|---|---|---|---:|
| Integridad de los datos | mezcla particiones o no puede demostrar que no lo hizo | las tres particiones están separadas y auditadas | además documenta hashes, política de partición y justifica la estrategia elegida | 20 % |
| Implementación | no ejecuta o produce resultados irreproducibles | entrena y evalúa siguiendo el protocolo | código claro y reutilizable, con las decisiones explicadas | 20 % |
| Diseño experimental | un solo resultado aislado | comparación controlada contra la línea base | varias semillas, dispersión reportada y variables controladas explícitas | 20 % |
| Análisis | repite las métricas sin interpretarlas | interpreta los errores y su distribución | identifica sesgos, límites y costo, y distingue evidencia de suposición | 25 % |
| Comunicación | incompleta o sin contexto | reporte entendible y model card presente | conclusiones verificables por un tercero a partir de los artefactos | 15 % |

La diferencia entre *adecuado* y *excelente* casi nunca está en la métrica: está en si el trabajo permite que otra persona llegue a la misma conclusión con los artefactos que dejaste. Un resultado peor, bien medido y bien explicado, se califica por encima de uno mejor que no se puede auditar.

**La aprobación exige al menos 70 % y cero errores críticos de fuga de datos.** La fuga es eliminatoria por sí sola porque invalida todas las demás cifras del trabajo, por buenas que parezcan.

## Autoevaluación antes de entregar

- [ ] Puedo explicar el laboratorio a alguien que no lo hizo, sin leer el código.
- [ ] Sé qué decisión tomé en cada paso y con qué evidencia la tomé.
- [ ] Miré `test` una sola vez, después de que existiera `experiment.lock.json`.
- [ ] Mi conclusión dice magnitud, incertidumbre, costo, errores y condiciones.
- [ ] Puedo señalar al menos una limitación real de mi resultado.

<!-- nav-bottom -->
## 🧭 Navegación del recorrido

| ⬅️ Laboratorio anterior | 🏠 Índice | Laboratorio siguiente ➡️ |
|---|:---:|---|
| [🔧 Fine-tuning eficiente de transformer](../../advanced_labs/25_transformer_finetuning/README.md) | [Las 31 rutas](../../parts/README.md) | [🎙️ Clasificación de audio con SpeechCommands](../../advanced_labs/27_audio_speechcommands/README.md) |

**En este laboratorio:** [📄 Guía](README.md) · [🧠 Teoría](theory.md) · [🔬 Experimentos](experiments.md) · **📝 Evaluación** · [📓 Recorrido](notebook.ipynb) · [✏️ Estudiante](notebook_student.ipynb) · [✅ Solución](notebook_solution.ipynb)

🔬 [Parte 7 — Especializaciones avanzadas](../../parts/07-especializaciones-avanzadas.md) · [🏠 Portada del repositorio](../../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/26_segmentation_unet/index.html) · [🖥️ Página HTML local](index.html)
<!-- /nav-bottom -->
