# Dataset Card — credit_card_fraud

## Identificación

- **Fuente:** Kaggle / ULB
- **Identificador:** `mlg-ulb/creditcardfraud`
- **Referencia oficial:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- **Licencia/condiciones:** Uso sujeto a términos de Kaggle y autor
- **Estado de revisión de licencia:** `review_required`
- **Usado por:** `06_autoencoder_anomaly`

## Descripción

284.807 transacciones reales; el laboratorio evita reequilibrar el conjunto de test.

## Política del repositorio

- El dataset no se incluye en Git.
- Se descarga desde la fuente declarada.
- No existe fallback sintético.
- Transformadores y vocabularios se ajustan únicamente con `train`.
- `validation` selecciona decisiones y `test` se reserva para evaluación final.
- Cada ejecución conserva manifest, IDs y fingerprints cuando el adaptador los expone.

## Revisión antes de reutilizar

- Confirmar que la licencia y términos siguen vigentes.
- Confirmar que la fuente y versión no cambiaron.
- Revisar privacidad, representatividad, sesgos y redistribución.
- Documentar cualquier filtrado o transformación adicional.

## Limitaciones

284.807 transacciones reales; el laboratorio evita reequilibrar el conjunto de test.

Esta ficha resume el uso dentro del repositorio y no reemplaza la documentación oficial.
