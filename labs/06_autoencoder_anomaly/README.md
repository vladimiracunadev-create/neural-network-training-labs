# Autoencoder para fraude

<!-- nav-top -->
> 🧭 [⬅️ Anterior](../../labs/05_lstm_time_series/README.md) · [🏠 Índice](../../README.md#laboratorios) · [Siguiente ➡️](../../labs/07_transformer_attention/README.md)
<!-- /nav-top -->

## Objetivo

Detectar transacciones fraudulentas mediante error de reconstrucción.

## Dataset real

- **Dataset:** `credit_card_fraud`
- **Fuente:** Kaggle / ULB
- **Referencia:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- **Licencia/condiciones:** Uso sujeto a términos de Kaggle y autor
- **Uso:** los datos se descargan desde la fuente; no hay ejemplos sintéticos ni archivos inventados.

284.807 transacciones reales; el laboratorio evita reequilibrar el conjunto de test.

## Fundamento matemático

Minimizar ||x-decoder(encoder(x))||² sobre transacciones normales.

## Protocolo experimental

1. Descargar y verificar la procedencia.
2. Conservar o crear una partición reproducible.
3. Ajustar transformaciones únicamente con `train`.
4. Seleccionar modelo e hiperparámetros usando `validation`.
5. Evaluar `test` una sola vez tras congelar la decisión.
6. Comparar con la línea base: **Isolation Forest**.
7. Guardar configuración, entorno, métricas, predicciones, gráficos y modelo.

## Ejecución

```bash
python labs/06_autoencoder_anomaly/train.py --quick
python labs/06_autoencoder_anomaly/train.py --config improved
```

Preparar únicamente el dataset:

```bash
python -m neural_labs.cli dataset --lab 06_autoencoder_anomaly
```

Inferencia y exportación:

```bash
neural-labs predict --lab 06_autoencoder_anomaly --run latest --input sample.json
neural-labs export --lab 06_autoencoder_anomaly --run latest --format onnx --verify
```

## Métricas

precision, recall, f1, roc_auc, pr_auc.

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
neural-labs quality --lab 06_autoencoder_anomaly --quick
neural-labs benchmark --lab 06_autoencoder_anomaly --quick --split-seed 42 --training-seeds 41 42 43
neural-labs leaderboard
```

## Sellado del experimento

La partición se controla con `split_seed`; la inicialización y el entrenamiento con `training_seed`. El conjunto `test` se abre solamente después de seleccionar el checkpoint mediante validación y escribir `experiment.lock.json`.

<!-- nav-bottom -->
## 🧭 Navegación del curso

| ⬅️ Anterior | Siguiente ➡️ |
|---|---|
| [📈 LSTM para series temporales](../../labs/05_lstm_time_series/README.md) | [🔭 Transformer para noticias](../../labs/07_transformer_attention/README.md) |

[🏠 Portada del repositorio](../../README.md) · [🌐 Ver en el sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/06_autoencoder_anomaly/index.html)
<!-- /nav-bottom -->
