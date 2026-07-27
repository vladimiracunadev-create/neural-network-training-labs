# Aprendizaje federado por participante

<!-- nav-top -->
> 🧭 [⬅️ Anterior](../../labs/14_knowledge_distillation/README.md) · [🏠 Índice](../../README.md#laboratorios) · [Siguiente ➡️](../../labs/16_backpropagation_manual/README.md)
<!-- /nav-top -->

## Objetivo

Aplicar FedAvg usando participantes reales como clientes naturales.

## Dataset real

- **Dataset:** `uci_har_subjects`
- **Fuente:** UCI
- **Referencia:** https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
- **Licencia/condiciones:** CC BY 4.0
- **Uso:** los datos se descargan desde la fuente; no hay ejemplos sintéticos ni archivos inventados.

No crea clientes espaciales artificiales; conserva identificadores reales de sujetos.

## Fundamento matemático

w_{t+1}=Σ_k(n_k/n)w_k.

## Protocolo experimental

1. Descargar y verificar la procedencia.
2. Conservar o crear una partición reproducible.
3. Ajustar transformaciones únicamente con `train`.
4. Seleccionar modelo e hiperparámetros usando `validation`.
5. Evaluar `test` una sola vez tras congelar la decisión.
6. Comparar con la línea base: **Entrenamiento centralizado**.
7. Guardar configuración, entorno, métricas, predicciones, gráficos y modelo.

## Ejecución

```bash
python labs/15_federated_learning/train.py --quick
python labs/15_federated_learning/train.py --config improved
```

Preparar únicamente el dataset:

```bash
python -m neural_labs.cli dataset --lab 15_federated_learning
```

Inferencia y exportación:

```bash
neural-labs predict --lab 15_federated_learning --run latest --input sample.json
neural-labs export --lab 15_federated_learning --run latest --format onnx --verify
```

## Métricas

accuracy, macro_f1, client_accuracy_std.

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
neural-labs quality --lab 15_federated_learning --quick
neural-labs benchmark --lab 15_federated_learning --quick --split-seed 42 --training-seeds 41 42 43
neural-labs leaderboard
```

## Sellado del experimento

La partición se controla con `split_seed`; la inicialización y el entrenamiento con `training_seed`. El conjunto `test` se abre solamente después de seleccionar el checkpoint mediante validación y escribir `experiment.lock.json`.

<!-- nav-bottom -->
## 🧭 Navegación del curso

| ⬅️ Anterior | Siguiente ➡️ |
|---|---|
| [⚗️ Destilación de conocimiento](../../labs/14_knowledge_distillation/README.md) | [∂ Backpropagation manual](../../labs/16_backpropagation_manual/README.md) |

[🏠 Portada del repositorio](../../README.md) · [🌐 Ver en el sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/15_federated_learning/index.html)
<!-- /nav-bottom -->
