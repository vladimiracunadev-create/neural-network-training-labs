# Clasificación de audio con SpeechCommands

## Objetivo

Clasificar comandos hablados desde waveform y log-mel spectrograms.

## Dataset público real

- **Dataset:** `speechcommands_v0.02`
- **Fuente:** Torchaudio / Google Speech Commands
- **Licencia/condiciones:** Creative Commons BY 4.0
- **Entrada:** audio mono de un segundo a 16 kHz
- **Datos sintéticos:** no se usan.

Los datos se descargan desde el proveedor oficial mediante los adaptadores del repositorio. Los archivos grandes no se incluyen en Git.

## Modelo y fundamento

- **Modelo:** `audio-cnn`
- **Teoría:** Waveform, espectrograma log-mel, convolución 2D y robustez ante ruido.
- **Línea base:** MFCC + regresión logística

## Protocolo

1. Crear particiones con `split_seed` o conservar los splits oficiales.
2. Entrenar y seleccionar exclusivamente con `train` y `validation`.
3. Guardar `best_model.pt` y escribir `experiment.lock.json`.
4. Abrir `test` una sola vez después del congelamiento.
5. Registrar métricas, configuración, procedencia y limitaciones.

## Ejecución

```bash
neural-labs train-advanced --track 27_audio_speechcommands --quick
neural-labs train-advanced --track 27_audio_speechcommands --split-seed 42 --training-seed 43
```

## Métricas

accuracy, macro_f1, confusion_matrix, noise_robustness.

## Cuadernos

- `notebook.ipynb`: recorrido completo.
- `notebook_student.ipynb`: actividades sin resolver.
- `notebook_solution.ipynb`: referencia docente.

## Limitación principal

Acentos, micrófonos y ambientes no están representados uniformemente.
