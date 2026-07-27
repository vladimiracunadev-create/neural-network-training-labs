# MLP multiclase

## Objetivo

Resolver clasificación no lineal con capas densas, activaciones y regularización.

## Dataset real

- **Dataset:** `dry_bean`
- **Fuente:** UCI
- **Referencia:** https://archive.ics.uci.edu/dataset/602/dry+bean+dataset
- **Licencia/condiciones:** CC BY 4.0
- **Uso:** los datos se descargan desde la fuente; no hay ejemplos sintéticos ni archivos inventados.

13.611 granos de siete variedades reales y 16 atributos de forma.

## Fundamento matemático

h=ReLU(xW1+b1); logits=hW2+b2.

## Protocolo experimental

1. Descargar y verificar la procedencia.
2. Conservar o crear una partición reproducible.
3. Ajustar transformaciones únicamente con `train`.
4. Seleccionar modelo e hiperparámetros usando `validation`.
5. Evaluar `test` una sola vez tras congelar la decisión.
6. Comparar con la línea base: **Regresión logística multinomial y Random Forest**.
7. Guardar configuración, entorno, métricas, predicciones, gráficos y modelo.

## Ejecución

```bash
python labs/02_mlp_nonlinear/train.py --quick
python labs/02_mlp_nonlinear/train.py --config improved
```

Preparar únicamente el dataset:

```bash
python -m neural_labs.cli dataset --lab 02_mlp_nonlinear
```

Inferencia y exportación:

```bash
neural-labs predict --lab 02_mlp_nonlinear --run latest --input sample.json
neural-labs export --lab 02_mlp_nonlinear --run latest --format onnx --verify
```

## Métricas

accuracy, balanced_accuracy, macro_precision, macro_recall, macro_f1.

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
neural-labs quality --lab 02_mlp_nonlinear --quick
neural-labs benchmark --lab 02_mlp_nonlinear --quick --split-seed 42 --training-seeds 41 42 43
neural-labs leaderboard
```

## Sellado del experimento

La partición se controla con `split_seed`; la inicialización y el entrenamiento con `training_seed`. El conjunto `test` se abre solamente después de seleccionar el checkpoint mediante validación y escribir `experiment.lock.json`.
