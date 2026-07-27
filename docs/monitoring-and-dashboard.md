# Monitoreo y panel

La API registra por defecto resúmenes estadísticos de las entradas y las predicciones en JSONL. No conserva las características crudas salvo que una organización modifique explícitamente esa política.

```bash
neural-labs serve
neural-labs monitor
neural-labs dashboard
```

`GET /drift` compara las estadísticas recientes con `monitoring/reference_stats.json` cuando existe. El panel Streamlit permite revisar ejecuciones, métricas y el registro local de modelos.

Este monitor es una referencia educativa. Un despliegue real debe añadir ventanas temporales, alertas, control de acceso, retención, métricas por subgrupo, detección de cambios de esquema y revisión humana.
