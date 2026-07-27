# Búsqueda de hiperparámetros

<!-- nav-top -->
> 🧭 [⬅️ Anterior](../../labs/12_multimodal_fusion/README.md) · [🏠 Índice](../../README.md#laboratorios) · [Siguiente ➡️](../../labs/14_knowledge_distillation/README.md)
<!-- /nav-top -->

## Objetivo

Optimizar profundidad, ancho, dropout y learning rate sin tocar test.

## Dataset real

- **Dataset:** `adult_census`
- **Fuente:** UCI
- **Referencia:** https://archive.ics.uci.edu/dataset/2/adult
- **Licencia/condiciones:** CC BY 4.0
- **Uso:** los datos se descargan desde la fuente; no hay ejemplos sintéticos ni archivos inventados.

48.842 registros reales del censo de 1994.

## Fundamento matemático

Selección por validación; test se usa solo tras elegir la mejor prueba.

## Protocolo experimental

1. Descargar y verificar la procedencia.
2. Conservar o crear una partición reproducible.
3. Ajustar transformaciones únicamente con `train`.
4. Seleccionar modelo e hiperparámetros usando `validation`.
5. Evaluar `test` una sola vez tras congelar la decisión.
6. Comparar con la línea base: **Regresión logística**.
7. Guardar configuración, entorno, métricas, predicciones, gráficos y modelo.

## Ejecución

```bash
python labs/13_hyperparameter_search/train.py --quick
python labs/13_hyperparameter_search/train.py --config improved
```

Preparar únicamente el dataset:

```bash
python -m neural_labs.cli dataset --lab 13_hyperparameter_search
```

Inferencia y exportación:

```bash
neural-labs predict --lab 13_hyperparameter_search --run latest --input sample.json
neural-labs export --lab 13_hyperparameter_search --run latest --format onnx --verify
```

## Métricas

accuracy, balanced_accuracy, f1, roc_auc, pr_auc.

## Archivos

- `notebook.ipynb`: recorrido completo y ejecutable.
- `notebook_student.ipynb`: actividades evaluables sin soluciones.
- `notebook_solution.ipynb`: resolución docente y pruebas de referencia.
- `train.py`: interfaz de terminal que usa el mismo código del cuaderno.
- `configs/baseline.yaml`: configuración base.
- `configs/improved.yaml`: configuración ampliada.
- `data/dataset.yaml`: procedencia, licencia y política de partición.

## Ejercicios

- Cambiar una decisión experimental y justificarla.
- Analizar errores por clase o segmento.
- Comparar costo, precisión y latencia.
- Documentar sesgos, limitaciones y usos no recomendados.


## Material formativo v3

- [`theory.md`](theory.md): fundamento, protocolo y riesgos de interpretación.
- [`experiments.md`](experiments.md): hipótesis, variables controladas y tabla multi-semilla.
- [`assessment.md`](assessment.md): preguntas y rúbrica de evaluación.
- [`lesson.yaml`](lesson.yaml): resultados de aprendizaje, prerrequisitos y entregables.

## Comandos profesionales

```bash
neural-labs quality --lab 13_hyperparameter_search --quick
neural-labs benchmark --lab 13_hyperparameter_search --quick --split-seed 42 --training-seeds 41 42 43
neural-labs leaderboard
```

## Sellado del experimento

La partición se controla con `split_seed`; la inicialización y el entrenamiento con `training_seed`. El conjunto `test` se abre solamente después de seleccionar el checkpoint mediante validación y escribir `experiment.lock.json`.

<!-- nav-bottom -->
## 🧭 Navegación del curso

| ⬅️ Anterior | Siguiente ➡️ |
|---|---|
| [🔀 Fusión de sensores](../../labs/12_multimodal_fusion/README.md) | [⚗️ Destilación de conocimiento](../../labs/14_knowledge_distillation/README.md) |

[🏠 Portada del repositorio](../../README.md) · [🌐 Ver en el sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/13_hyperparameter_search/index.html)
<!-- /nav-bottom -->
