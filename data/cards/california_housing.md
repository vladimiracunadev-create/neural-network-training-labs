# Dataset Card — california_housing

## Identificación

- **Fuente:** scikit-learn / StatLib
- **Identificador:** `fetch_california_housing`
- **Referencia oficial:** https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html
- **Licencia/condiciones:** Consultar fuente StatLib
- **Estado de revisión de licencia:** `review_required`
- **Usado por:** `18_optimizers_and_schedulers`

## Descripción

Datos reales del censo de California de 1990.

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

Datos reales del censo de California de 1990.

Esta ficha resume el uso dentro del repositorio y no reemplaza la documentación oficial.
