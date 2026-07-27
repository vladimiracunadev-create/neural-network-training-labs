# Dataset Card — adult_census

## Identificación

- **Fuente:** UCI
- **Identificador:** `2`
- **Referencia oficial:** https://archive.ics.uci.edu/dataset/2/adult
- **Licencia/condiciones:** CC BY 4.0
- **Estado de revisión de licencia:** `declared`
- **Usado por:** `13_hyperparameter_search`, `21_explainability`

## Descripción

48.842 registros reales del censo de 1994.

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

48.842 registros reales del censo de 1994.

Esta ficha resume el uso dentro del repositorio y no reemplaza la documentación oficial.
