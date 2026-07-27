# GAN generativa

## Objetivo

Generar prendas a partir de imágenes reales de Fashion-MNIST.

## Dataset real

- **Dataset:** `fashion_mnist`
- **Fuente:** Torchvision / Zalando Research
- **Referencia:** https://github.com/zalandoresearch/fashion-mnist
- **Licencia/condiciones:** MIT
- **Uso:** los datos se descargan desde la fuente; no hay ejemplos sintéticos ni archivos inventados.

No usa anillos ni puntos inventados; entrena con prendas reales etiquetadas.

## Fundamento matemático

min_G max_D E[log D(x)] + E[log(1-D(G(z)))].

## Protocolo experimental

1. Descargar y verificar la procedencia.
2. Conservar o crear una partición reproducible.
3. Ajustar transformaciones únicamente con `train`.
4. Seleccionar modelo e hiperparámetros usando `validation`.
5. Evaluar `test` una sola vez tras congelar la decisión.
6. Comparar con la línea base: **PCA generativa y distribución real de referencia**.
7. Guardar configuración, entorno, métricas, predicciones, gráficos y modelo.

## Ejecución

```bash
python labs/08_gan_generation/train.py --quick
python labs/08_gan_generation/train.py --config improved
```

Preparar únicamente el dataset:

```bash
python -m neural_labs.cli dataset --lab 08_gan_generation
```

Inferencia y exportación:

```bash
neural-labs predict --lab 08_gan_generation --run latest --input sample.json
neural-labs export --lab 08_gan_generation --run latest --format onnx --verify
```

## Métricas

generator_loss, discriminator_loss, mmd_rbf, diversity, nearest_real_distance, moment_distance.

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
neural-labs quality --lab 08_gan_generation --quick
neural-labs benchmark --lab 08_gan_generation --quick --split-seed 42 --training-seeds 41 42 43
neural-labs leaderboard
```

## Sellado del experimento

La partición se controla con `split_seed`; la inicialización y el entrenamiento con `training_seed`. El conjunto `test` se abre solamente después de seleccionar el checkpoint mediante validación y escribir `experiment.lock.json`.
