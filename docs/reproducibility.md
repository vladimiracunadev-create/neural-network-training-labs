# Reproducibilidad

## Entorno

`requirements/core-tested.txt` registra las versiones del núcleo verificadas durante la construcción. `pyproject.toml` define rangos compatibles para instalación normal. Una ejecución conserva las versiones efectivas en `environment.json`.

## Semillas

`seed_everything` configura Python, NumPy y PyTorch. Cuando la plataforma lo admite, activa algoritmos deterministas y desactiva el benchmark no determinista de cuDNN.

## Datos

Los identificadores de cada partición se convierten en hashes SHA-256. El manifiesto permite comprobar que dos ejecuciones usaron la misma división aunque los datos no estén en Git.

## Hardware

CPU, CUDA y MPS pueden producir pequeñas diferencias numéricas. El dispositivo y las versiones se registran. Para comparaciones estrictas, use la misma plataforma y el mismo entorno.

## Ejecuciones repetidas

```bash
for seed in 11 22 33 44 55; do
  neural-labs train --lab 02_mlp_nonlinear --seed "$seed"
done
```

No sobrescriba una ejecución previa. Cada llamada crea una carpeta con marca UTC.

## Dependencias opcionales

Instale solamente los grupos necesarios o use `.[full]`. Para regenerar un bloqueo completo en una plataforma concreta se recomienda resolver `requirements/full.in` con una herramienta de locking y conservar el archivo resultante junto al informe del experimento.
