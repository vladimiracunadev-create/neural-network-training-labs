# MLOps y seguimiento de experimentos

La capa MLOps es opcional. El núcleo funciona sin servidor externo y registra cada evento en `tracking.jsonl`.

## Seguimiento local

```bash
neural-labs train --lab 02_mlp_nonlinear --quick --tracker json
```

El registro contiene inicio, parámetros, métricas, artefactos y estado final. Esta salida es portable y puede procesarse sin una base de datos.

## MLflow

```bash
pip install -e ".[mlops]"
docker compose -f deploy/mlflow/compose.yaml up -d
export MLFLOW_TRACKING_URI=http://localhost:5000
neural-labs train --lab 02_mlp_nonlinear --tracker mlflow
```

El tracker compuesto conserva simultáneamente el archivo local y envía parámetros, métricas y artefactos a MLflow.

## Convención

- Un experimento MLflow corresponde a un laboratorio.
- Un run corresponde a una combinación de configuración, semilla, dataset y entorno.
- Los checkpoints no deben promoverse solo por una métrica; revise model card, intervalos, subgrupos y costo.
- No envíe tokens, credenciales ni datos sensibles como parámetros o etiquetas.
