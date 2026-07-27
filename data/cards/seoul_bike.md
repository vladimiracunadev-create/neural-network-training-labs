# Dataset Card — seoul_bike

## Identificación

- **Fuente:** UCI
- **Identificador:** `560`
- **Referencia oficial:** https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand
- **Licencia/condiciones:** CC BY 4.0
- **Estado de revisión de licencia:** `declared`
- **Usado por:** `05_lstm_time_series`

## Descripción

8.760 observaciones reales de arriendo de bicicletas y clima en Seúl.

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

8.760 observaciones reales de arriendo de bicicletas y clima en Seúl.

Esta ficha resume el uso dentro del repositorio y no reemplaza la documentación oficial.
