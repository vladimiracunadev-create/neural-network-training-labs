# Difusión DDPM sobre Fashion-MNIST

<!-- nav-top -->
> 🧭 [⬅️ Anterior](../../advanced_labs/28_wgan_gp/README.md) · [🏠 Índice](../../README.md#laboratorios) · [Siguiente ➡️](../../advanced_labs/30_self_supervised_simclr/README.md)
<!-- /nav-top -->

## Objetivo

Aprender predicción de ruido y muestreo iterativo sobre imágenes reales.

## Dataset público real

- **Dataset:** `fashion_mnist`
- **Fuente:** Torchvision / Zalando Research
- **Licencia/condiciones:** MIT para código; consultar dataset
- **Entrada:** imágenes Fashion-MNIST normalizadas
- **Datos sintéticos:** no se usan.

Los datos se descargan desde el proveedor oficial mediante los adaptadores del repositorio. Los archivos grandes no se incluyen en Git.

## Modelo y fundamento

- **Modelo:** `tiny-ddpm`
- **Teoría:** Proceso directo de ruido, predicción de epsilon, cronograma beta y muestreo inverso.
- **Línea base:** Autoencoder generativo simple

## Protocolo

1. Crear particiones con `split_seed` o conservar los splits oficiales.
2. Entrenar y seleccionar exclusivamente con `train` y `validation`.
3. Guardar `best_model.pt` y escribir `experiment.lock.json`.
4. Abrir `test` una sola vez después del congelamiento.
5. Registrar métricas, configuración, procedencia y limitaciones.

## Ejecución

```bash
neural-labs train-advanced --track 29_diffusion_ddpm --quick
neural-labs train-advanced --track 29_diffusion_ddpm --split-seed 42 --training-seed 43
```

## Métricas

noise_mse, sample_diversity, sampling_latency, reconstruction_proxy.

## Cuadernos

- `notebook.ipynb`: recorrido completo.
- `notebook_student.ipynb`: actividades sin resolver.
- `notebook_solution.ipynb`: referencia docente.

## Limitación principal

El modelo pequeño sirve para estudio; no debe extrapolarse a generación fotográfica de alta resolución.

<!-- nav-bottom -->
## 🧭 Navegación del curso

| ⬅️ Anterior | Siguiente ➡️ |
|---|---|
| [🖌️ WGAN-GP sobre Fashion-MNIST](../../advanced_labs/28_wgan_gp/README.md) | [🪞 Aprendizaje autosupervisado SimCLR](../../advanced_labs/30_self_supervised_simclr/README.md) |

[🏠 Portada del repositorio](../../README.md) · [🌐 Ver en el sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/29_diffusion_ddpm/index.html)
<!-- /nav-bottom -->
