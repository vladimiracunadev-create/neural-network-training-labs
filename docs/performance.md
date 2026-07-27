# Rendimiento y hardware

## Dispositivos

- `cpu`: máxima compatibilidad.
- `cuda`: GPU NVIDIA; permite precisión mixta.
- `mps`: aceleración en Apple Silicon cuando la operación está soportada.

```bash
neural-labs train --lab 03_cnn_vision --device cuda --amp
neural-labs train --lab 03_cnn_vision --device mps
```

## Perfil automático

Para modelos compatibles se registran:

- latencia media;
- latencia p95;
- throughput en muestras por segundo;
- tamaño de lote del perfil;
- memoria CUDA máxima cuando está disponible.

Las mediciones son locales y dependen de hardware, carga del sistema, tamaño del lote y calentamiento. No compare cifras de máquinas distintas como si fueran equivalentes.

## Determinismo

El modo determinista mejora la repetibilidad, pero puede reducir el rendimiento y no garantiza resultados idénticos entre versiones o plataformas. Puede desactivarse conscientemente con `--no-deterministic` para mediciones de velocidad, documentando la decisión.

## Compilación

`--compile` activa compilación experimental para modelos estándar. Debe validarse la equivalencia numérica y el beneficio real; la primera ejecución puede ser más lenta debido al costo de compilación.
