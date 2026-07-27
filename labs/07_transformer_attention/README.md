# Transformer para noticias

<!-- nav-top -->
> 🧭 [⬅️ Anterior](../../labs/06_autoencoder_anomaly/README.md) · [🏠 Índice](../../README.md#laboratorios) · [Siguiente ➡️](../../labs/08_gan_generation/README.md)
<!-- /nav-top -->

## Objetivo

Aplicar atención multi-cabeza a clasificación de noticias reales.

## Dataset real

- **Dataset:** `ag_news`
- **Fuente:** Hugging Face
- **Referencia:** https://huggingface.co/datasets/fancyzhx/ag_news
- **Licencia/condiciones:** Consultar dataset card
- **Uso:** los datos se descargan desde la fuente; no hay ejemplos sintéticos ni archivos inventados.

Noticias reales en cuatro categorías con particiones públicas.

## Fundamento matemático

Attention(Q,K,V)=softmax(QKᵀ/√d)V.

## Protocolo experimental

1. Descargar y verificar la procedencia.
2. Conservar o crear una partición reproducible.
3. Ajustar transformaciones únicamente con `train`.
4. Seleccionar modelo e hiperparámetros usando `validation`.
5. Evaluar `test` una sola vez tras congelar la decisión.
6. Comparar con la línea base: **TF-IDF + regresión logística**.
7. Guardar configuración, entorno, métricas, predicciones, gráficos y modelo.

## Ejecución

```bash
python labs/07_transformer_attention/train.py --quick
python labs/07_transformer_attention/train.py --config improved
```

Preparar únicamente el dataset:

```bash
python -m neural_labs.cli dataset --lab 07_transformer_attention
```

Inferencia y exportación:

```bash
neural-labs predict --lab 07_transformer_attention --run latest --input sample.json
neural-labs export --lab 07_transformer_attention --run latest --format onnx --verify
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
neural-labs quality --lab 07_transformer_attention --quick
neural-labs benchmark --lab 07_transformer_attention --quick --split-seed 42 --training-seeds 41 42 43
neural-labs leaderboard
```

## Sellado del experimento

La partición se controla con `split_seed`; la inicialización y el entrenamiento con `training_seed`. El conjunto `test` se abre solamente después de seleccionar el checkpoint mediante validación y escribir `experiment.lock.json`.

<!-- nav-bottom -->
## 🧭 Navegación del curso

| ⬅️ Anterior | Siguiente ➡️ |
|---|---|
| [🧬 Autoencoder para fraude](../../labs/06_autoencoder_anomaly/README.md) | [🎨 GAN generativa](../../labs/08_gan_generation/README.md) |

[🏠 Portada del repositorio](../../README.md) · [🌐 Ver en el sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/07_transformer_attention/index.html)
<!-- /nav-bottom -->
