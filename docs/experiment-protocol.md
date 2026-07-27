# Protocolo experimental

## Semillas independientes

`split_seed` controla exclusivamente la creación de `train`, `validation` y `test`. `training_seed` controla pesos iniciales, orden de batches, dropout y exploración. Un benchmark serio mantiene fija la primera y repite la segunda.

## Apertura del test

1. Descargar y verificar fuente.
2. Crear particiones.
3. Ajustar preprocesamiento solo con train.
4. Calcular línea base en validation.
5. Entrenar y seleccionar por validation.
6. Escribir `experiment.lock.json`.
7. Evaluar test una vez.
8. Guardar predicciones y métricas.

El lock registra laboratorio, semillas, configuración, métrica de selección, checkpoint previsto, hash del dataset y momento de congelación.

## Validación cruzada

La validación cruzada integrada concatena train y validation. Nunca incorpora test. Para series temporales se usan ventanas walk-forward, no folds aleatorios.

## Métricas

- Clasificación: accuracy, balanced accuracy, precision, recall, F1, ROC-AUC y PR-AUC cuando corresponda.
- Regresión: MAE, RMSE, MAPE y R².
- Generación: diversidad, MMD, distancia a ejemplos reales y diagnóstico visual.
- Refuerzo: retorno, tasa de stockout, costo de inventario, service level y dispersión entre semillas.
- Despliegue: latencia media, P95, throughput, tamaño y diferencia numérica.
