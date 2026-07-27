# Informe de construcción y validación — versión 1.0.0

Fecha: 24 de julio de 2026.

## Alcance consolidado

- 25 laboratorios centrales y 6 especializaciones avanzadas.
- 31 rutas de aprendizaje y 93 notebooks Jupyter.
- 21 configuraciones de datasets, equivalentes a 20 familias públicas reales.
- 0 datasets sintéticos o fallbacks inventados.
- 31 documentos teóricos, 31 planes experimentales y 31 rúbricas.
- 23 factorías de modelos registradas por dominio.
- API, inferencia por lotes, registro de modelos, monitoreo, panel, exportación y entrenamiento distribuido.

## Validaciones locales

```text
python scripts/validate_repository.py
Resultado: 25 laboratorios centrales, 6 avanzados, 93 notebooks y 0 advertencias.

python scripts/validate_nbgrader.py
Resultado: 31 pares estudiante/solución válidos.

pytest -m "not network and not slow" --cov=neural_labs
Resultado: 68 pruebas aprobadas, 2 externas separadas y 84,62 % de cobertura.

neural-labs validate --warnings-as-errors
Resultado: 0 errores y 0 advertencias.

pip install -e . --no-deps --no-build-isolation
Resultado: instalación editable y comando neural-labs operativos.
```

También se comprobaron localmente los forwards de U-Net, CNN de audio y SimCLR, la generación de checkpoints portables y el flujo de promoción con puertas de calidad.

## Fuentes externas

Los entrenamientos avanzados descargan AG News, Oxford-IIIT Pet, SpeechCommands, Fashion-MNIST y CIFAR-10 desde sus adaptadores oficiales. Las pruebas de descarga permanecen separadas para que una caída de red no sea ocultada con datos artificiales. El workflow `advanced-smoke.yml` ejecuta semanalmente una matriz con las seis especializaciones.

## Dependencias reproducibles

La instalación editable y el wheel local se validan sin descargar dependencias. No se incluyó un `uv.lock` inventado: el índice disponible durante el empaquetado respondió HTTP 503. El workflow de release ejecuta `uv lock`, `uv lock --check` y la batería antes de publicar. `requirements/core-tested.txt` conserva las versiones del núcleo usadas en esta construcción.

## Alcance de la cifra de cobertura

La cobertura mínima de 80 % se aplica al núcleo verificable de producción. Los entrenadores que descargan modelos o datasets grandes y los backends opcionales de audio/transformers se prueban mediante contratos locales y workflows externos, no se contabilizan como si hubieran sido ejecutados sin sus dependencias y fuentes.
