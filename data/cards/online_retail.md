# Dataset Card — online_retail

## Identificación

- **Fuente:** UCI
- **Identificador:** `352`
- **Referencia oficial:** https://archive.ics.uci.edu/dataset/352/online+retail
- **Licencia/condiciones:** CC BY 4.0
- **Estado de revisión de licencia:** `declared`
- **Usado por:** `10_dqn_reinforcement`

## Descripción

La dinámica de inventario es un entorno educativo, pero la demanda diaria se construye exclusivamente desde transacciones reales de Online Retail.

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

La dinámica de inventario es un entorno educativo, pero la demanda diaria se construye exclusivamente desde transacciones reales de Online Retail.

Esta ficha resume el uso dentro del repositorio y no reemplaza la documentación oficial.
