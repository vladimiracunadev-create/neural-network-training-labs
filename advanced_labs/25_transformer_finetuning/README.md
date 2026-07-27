# Fine-tuning eficiente de transformer

## Objetivo

Comparar fine-tuning completo y LoRA sin tocar test durante selección.

## Dataset público real

- **Dataset:** `ag_news`
- **Fuente:** Hugging Face Datasets
- **Licencia/condiciones:** Consultar ficha AG News
- **Entrada:** texto en inglés
- **Datos sintéticos:** no se usan.

Los datos se descargan desde el proveedor oficial mediante los adaptadores del repositorio. Los archivos grandes no se incluyen en Git.

## Modelo y fundamento

- **Modelo:** `distilbert-base-uncased`
- **Teoría:** Tokenización subword, atención preentrenada, fine-tuning completo y adaptación eficiente LoRA.
- **Línea base:** TF-IDF + regresión logística

## Protocolo

1. Crear particiones con `split_seed` o conservar los splits oficiales.
2. Entrenar y seleccionar exclusivamente con `train` y `validation`.
3. Guardar `best_model.pt` y escribir `experiment.lock.json`.
4. Abrir `test` una sola vez después del congelamiento.
5. Registrar métricas, configuración, procedencia y limitaciones.

## Ejecución

```bash
neural-labs train-advanced --track 25_transformer_finetuning --quick
neural-labs train-advanced --track 25_transformer_finetuning --split-seed 42 --training-seed 43
```

## Métricas

accuracy, macro_f1, latency_ms, trainable_parameters.

## Cuadernos

- `notebook.ipynb`: recorrido completo.
- `notebook_student.ipynb`: actividades sin resolver.
- `notebook_solution.ipynb`: referencia docente.

## Limitación principal

El corpus contiene titulares históricos y sesgos editoriales; no representa todo el lenguaje contemporáneo.
