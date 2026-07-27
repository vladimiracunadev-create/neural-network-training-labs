# Dataset Card — wine_quality

## Identificación

- **Fuente:** UCI
- **Identificador:** `186`
- **Referencia oficial:** https://archive.ics.uci.edu/dataset/186/wine+quality
- **Licencia/condiciones:** CC BY 4.0
- **Estado de revisión de licencia:** `declared`
- **Usado por:** `17_activations_and_losses`

## Descripción

Muestras reales de vinho verde con análisis fisicoquímico y evaluación sensorial.

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

Muestras reales de vinho verde con análisis fisicoquímico y evaluación sensorial.

Esta ficha resume el uso dentro del repositorio y no reemplaza la documentación oficial.
