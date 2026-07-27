# WGAN-GP sobre Fashion-MNIST

<!-- nav-top -->
> 🧭 [⬅️ Anterior](../../advanced_labs/27_audio_speechcommands/README.md) · [🏠 Índice](../../README.md#laboratorios) · [Siguiente ➡️](../../advanced_labs/29_diffusion_ddpm/README.md)
<!-- /nav-top -->

## Objetivo

Estudiar estabilidad generativa y gradient penalty con imágenes reales.

## Dataset público real

- **Dataset:** `fashion_mnist`
- **Fuente:** Torchvision / Zalando Research
- **Licencia/condiciones:** MIT para código; consultar dataset
- **Entrada:** imágenes Fashion-MNIST normalizadas
- **Datos sintéticos:** no se usan.

Los datos se descargan desde el proveedor oficial mediante los adaptadores del repositorio. Los archivos grandes no se incluyen en Git.

## Modelo y fundamento

- **Modelo:** `wgan-gp`
- **Teoría:** Distancia Wasserstein, crítico sin sigmoide y restricción Lipschitz mediante gradient penalty.
- **Línea base:** DCGAN convolucional

## Protocolo

1. Crear particiones con `split_seed` o conservar los splits oficiales.
2. Entrenar y seleccionar exclusivamente con `train` y `validation`.
3. Guardar `best_model.pt` y escribir `experiment.lock.json`.
4. Abrir `test` una sola vez después del congelamiento.
5. Registrar métricas, configuración, procedencia y limitaciones.

## Ejecución

```bash
neural-labs train-advanced --track 28_wgan_gp --quick
neural-labs train-advanced --track 28_wgan_gp --split-seed 42 --training-seed 43
```

## Métricas

wasserstein_estimate, energy_distance_proxy, diversity, training_stability.

## Cuadernos

- `notebook.ipynb`: recorrido completo.
- `notebook_student.ipynb`: actividades sin resolver.
- `notebook_solution.ipynb`: referencia docente.

## Limitación principal

Las métricas generativas aproximadas no sustituyen evaluación humana ni validación del uso previsto.

<!-- nav-bottom -->
## 🧭 Navegación del curso

| ⬅️ Anterior | Siguiente ➡️ |
|---|---|
| [🎙️ Clasificación de audio con SpeechCommands](../../advanced_labs/27_audio_speechcommands/README.md) | [🌫️ Difusión DDPM sobre Fashion-MNIST](../../advanced_labs/29_diffusion_ddpm/README.md) |

[🏠 Portada del repositorio](../../README.md) · [🌐 Ver en el sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/labs/28_wgan_gp/index.html)
<!-- /nav-bottom -->
