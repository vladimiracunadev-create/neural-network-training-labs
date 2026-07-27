# Segmentación semántica con U-Net

<!-- nav-top -->
> 🧭 [⬅️ Anterior](../../advanced_labs/25_transformer_finetuning/README.md) · [🏠 Índice](../../README.md#laboratorios) · [Siguiente ➡️](../../advanced_labs/27_audio_speechcommands/README.md)
<!-- /nav-top -->

## Objetivo

Segmentar mascota, fondo y contorno con IoU por clase.

## Dataset público real

- **Dataset:** `oxford_iiit_pet_segmentation`
- **Fuente:** Torchvision / University of Oxford
- **Licencia/condiciones:** Consultar términos Oxford-IIIT Pet
- **Entrada:** imagen RGB y máscara trimap
- **Datos sintéticos:** no se usan.

Los datos se descargan desde el proveedor oficial mediante los adaptadores del repositorio. Los archivos grandes no se incluyen en Git.

## Modelo y fundamento

- **Modelo:** `unet-small`
- **Teoría:** Arquitectura encoder-decoder, conexiones skip, pérdida por píxel e intersección sobre unión.
- **Línea base:** Máscara de clase mayoritaria

## Protocolo

1. Crear particiones con `split_seed` o conservar los splits oficiales.
2. Entrenar y seleccionar exclusivamente con `train` y `validation`.
3. Guardar `best_model.pt` y escribir `experiment.lock.json`.
4. Abrir `test` una sola vez después del congelamiento.
5. Registrar métricas, configuración, procedencia y limitaciones.

## Ejecución

```bash
neural-labs train-advanced --track 26_segmentation_unet --quick
neural-labs train-advanced --track 26_segmentation_unet --split-seed 42 --training-seed 43
```

## Métricas

mean_iou, iou_per_class, pixel_accuracy, dice.

## Cuadernos

- `notebook.ipynb`: recorrido completo.
- `notebook_student.ipynb`: actividades sin resolver.
- `notebook_solution.ipynb`: referencia docente.

## Limitación principal

Las imágenes se concentran en mascotas y fondos cotidianos; no generaliza a segmentación médica o industrial.

<!-- nav-bottom -->
## 🧭 Navegación del curso

| ⬅️ Anterior | Siguiente ➡️ |
|---|---|
| [🔧 Fine-tuning eficiente de transformer](../../advanced_labs/25_transformer_finetuning/README.md) | [🎙️ Clasificación de audio con SpeechCommands](../../advanced_labs/27_audio_speechcommands/README.md) |

[🏠 Portada del repositorio](../../README.md) · [🌐 Ver en el sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/26_segmentation_unet/index.html)
<!-- /nav-bottom -->
