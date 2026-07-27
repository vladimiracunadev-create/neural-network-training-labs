# Transfer learning con mascotas

<!-- nav-top -->
> 🧭 [⬅️ Anterior](../../labs/10_dqn_reinforcement/README.md) · [🏠 Índice](../../README.md#laboratorios) · [Siguiente ➡️](../../labs/12_multimodal_fusion/README.md)
<!-- /nav-top -->

## Objetivo

Comparar extracción de características, fine-tuning y entrenamiento desde cero.

## Dataset real

- **Dataset:** `oxford_iiit_pet`
- **Fuente:** Torchvision / Oxford
- **Referencia:** https://www.robots.ox.ac.uk/~vgg/data/pets/
- **Licencia/condiciones:** Uso académico según fuente
- **Uso:** los datos se descargan desde la fuente; no hay ejemplos sintéticos ni archivos inventados.

7.349 imágenes reales de 37 razas de perros y gatos.

## Fundamento matemático

Inicializar con pesos ImageNet y ajustar capas seleccionadas.

## Protocolo experimental

1. Descargar y verificar la procedencia.
2. Conservar o crear una partición reproducible.
3. Ajustar transformaciones únicamente con `train`.
4. Seleccionar modelo e hiperparámetros usando `validation`.
5. Evaluar `test` una sola vez tras congelar la decisión.
6. Comparar con la línea base: **CNN pequeña entrenada desde cero**.
7. Guardar configuración, entorno, métricas, predicciones, gráficos y modelo.

## Ejecución

```bash
python labs/11_transfer_learning/train.py --quick
python labs/11_transfer_learning/train.py --config improved
```

Preparar únicamente el dataset:

```bash
python -m neural_labs.cli dataset --lab 11_transfer_learning
```

Inferencia y exportación:

```bash
neural-labs predict --lab 11_transfer_learning --run latest --input sample.json
neural-labs export --lab 11_transfer_learning --run latest --format onnx --verify
```

## Métricas

accuracy, balanced_accuracy, macro_f1.

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
neural-labs quality --lab 11_transfer_learning --quick
neural-labs benchmark --lab 11_transfer_learning --quick --split-seed 42 --training-seeds 41 42 43
neural-labs leaderboard
```

## Sellado del experimento

La partición se controla con `split_seed`; la inicialización y el entrenamiento con `training_seed`. El conjunto `test` se abre solamente después de seleccionar el checkpoint mediante validación y escribir `experiment.lock.json`.

<!-- nav-bottom -->
## 🧭 Navegación del curso

| ⬅️ Anterior | Siguiente ➡️ |
|---|---|
| [🕹️ DQN para inventario con demanda real](../../labs/10_dqn_reinforcement/README.md) | [🔀 Fusión de sensores](../../labs/12_multimodal_fusion/README.md) |

[🏠 Portada del repositorio](../../README.md) · [🌐 Ver en el sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/11_transfer_learning/index.html)
<!-- /nav-bottom -->
