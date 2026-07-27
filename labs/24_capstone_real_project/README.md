# Proyecto final: churn de telecomunicaciones

## Objetivo

Resolver de extremo a extremo un problema real de abandono de clientes con documentación, evaluación y despliegue.

## Dataset real

- **Dataset:** `iranian_churn`
- **Fuente:** UCI
- **Referencia:** https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset
- **Licencia/condiciones:** CC BY 4.0
- **Uso:** los datos se descargan desde la fuente; no hay ejemplos sintéticos ni archivos inventados.

3.150 clientes recolectados aleatoriamente de la base de una empresa iraní de telecomunicaciones durante 12 meses.

## Fundamento matemático

Clasificación, calibración, selección de umbral y costo de errores.

## Protocolo experimental

1. Descargar y verificar la procedencia.
2. Conservar o crear una partición reproducible.
3. Ajustar transformaciones únicamente con `train`.
4. Seleccionar modelo e hiperparámetros usando `validation`.
5. Evaluar `test` una sola vez tras congelar la decisión.
6. Comparar con la línea base: **Regresión logística y Gradient Boosting**.
7. Guardar configuración, entorno, métricas, predicciones, gráficos y modelo.

## Ejecución

```bash
python labs/24_capstone_real_project/train.py --quick
python labs/24_capstone_real_project/train.py --config improved
```

Preparar únicamente el dataset:

```bash
python -m neural_labs.cli dataset --lab 24_capstone_real_project
```

Inferencia y exportación:

```bash
neural-labs predict --lab 24_capstone_real_project --run latest --input sample.json
neural-labs export --lab 24_capstone_real_project --run latest --format onnx --verify
```

## Métricas

accuracy, balanced_accuracy, precision, recall, f1, roc_auc, pr_auc.

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
neural-labs quality --lab 24_capstone_real_project --quick
neural-labs benchmark --lab 24_capstone_real_project --quick --split-seed 42 --training-seeds 41 42 43
neural-labs leaderboard
```

## Sellado del experimento

La partición se controla con `split_seed`; la inicialización y el entrenamiento con `training_seed`. El conjunto `test` se abre solamente después de seleccionar el checkpoint mediante validación y escribir `experiment.lock.json`.
