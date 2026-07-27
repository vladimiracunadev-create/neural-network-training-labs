# Aprendizaje autosupervisado SimCLR

## Objetivo

Preentrenar representaciones con dos vistas reales y evaluar mediante linear probe.

## Dataset público real

- **Dataset:** `cifar10`
- **Fuente:** Torchvision / University of Toronto
- **Licencia/condiciones:** Consultar términos CIFAR-10
- **Entrada:** imágenes CIFAR-10
- **Datos sintéticos:** no se usan.

Los datos se descargan desde el proveedor oficial mediante los adaptadores del repositorio. Los archivos grandes no se incluyen en Git.

## Modelo y fundamento

- **Modelo:** `resnet18-simclr`
- **Teoría:** Dos vistas, similitud coseno, pérdida NT-Xent y evaluación linear probe.
- **Línea base:** ResNet18 aleatoria + linear probe

## Protocolo

1. Crear particiones con `split_seed` o conservar los splits oficiales.
2. Entrenar y seleccionar exclusivamente con `train` y `validation`.
3. Guardar `best_model.pt` y escribir `experiment.lock.json`.
4. Abrir `test` una sola vez después del congelamiento.
5. Registrar métricas, configuración, procedencia y limitaciones.

## Ejecución

```bash
neural-labs train-advanced --track 30_self_supervised_simclr --quick
neural-labs train-advanced --track 30_self_supervised_simclr --split-seed 42 --training-seed 43
```

## Métricas

nt_xent, linear_probe_accuracy, knn_accuracy, embedding_uniformity.

## Cuadernos

- `notebook.ipynb`: recorrido completo.
- `notebook_student.ipynb`: actividades sin resolver.
- `notebook_solution.ipynb`: referencia docente.

## Limitación principal

La elección de aumentos define invariancias y puede borrar información relevante para tareas posteriores.
