# Dataset Card — dry_bean

## Identificación

- **Fuente:** UCI
- **Identificador:** `602`
- **Referencia oficial:** https://archive.ics.uci.edu/dataset/602/dry+bean+dataset
- **Licencia/condiciones:** CC BY 4.0
- **Estado de revisión de licencia:** `declared`
- **Usado por:** `02_mlp_nonlinear`

## Descripción

13.611 granos de siete variedades reales y 16 atributos de forma.

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

13.611 granos de siete variedades reales y 16 atributos de forma.

Esta ficha resume el uso dentro del repositorio y no reemplaza la documentación oficial.
