# Dataset Card — banknote_authentication

## Identificación

- **Fuente:** UCI
- **Identificador:** `267`
- **Referencia oficial:** https://archive.ics.uci.edu/dataset/267/banknote+authentication
- **Licencia/condiciones:** Consultar ficha UCI
- **Estado de revisión de licencia:** `review_required`
- **Usado por:** `01_pytorch_perceptron`

## Descripción

Características extraídas de imágenes reales de billetes.

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

Características extraídas de imágenes reales de billetes.

Esta ficha resume el uso dentro del repositorio y no reemplaza la documentación oficial.
