# Dataset Card — iris

## Identificación

- **Fuente:** UCI
- **Identificador:** `53`
- **Referencia oficial:** https://archive.ics.uci.edu/dataset/53/iris
- **Licencia/condiciones:** CC BY 4.0
- **Estado de revisión de licencia:** `declared`
- **Usado por:** `16_backpropagation_manual`

## Descripción

150 mediciones botánicas reales de tres especies de Iris.

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

150 mediciones botánicas reales de tres especies de Iris.

Esta ficha resume el uso dentro del repositorio y no reemplaza la documentación oficial.
