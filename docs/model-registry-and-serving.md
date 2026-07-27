# Registro y serving

## Registro local

`LocalModelRegistry` conserva nombre, versión, ejecución, métricas y alias. Registrar exige un `experiment.lock.json` válido.

Los alias recomendados son:

- `challenger`: candidato todavía no promovido.
- `champion`: modelo servido por defecto.
- `archived`: versión conservada sin tráfico.

## MLflow

El backend opcional registra el modelo PyTorch, adjunta artefactos y asigna alias mediante MLflow Model Registry.

## API

La API FastAPI carga el alias configurado y expone salud, contrato, predicción y métricas. El contrato define shape, clases, preprocesador, vocabulario y formatos aceptados.

No todos los laboratorios comparten el mismo tipo de serving. GAN, grafos y ciertos modelos especializados requieren endpoints específicos y se marcan como no servibles por la API genérica.
