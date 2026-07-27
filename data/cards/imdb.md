# Dataset Card — imdb

## Identificación

- **Fuente:** Hugging Face / Stanford
- **Identificador:** `stanfordnlp/imdb`
- **Referencia oficial:** https://huggingface.co/datasets/stanfordnlp/imdb
- **Licencia/condiciones:** Consultar dataset card
- **Estado de revisión de licencia:** `review_required`
- **Usado por:** `04_rnn_sequences`

## Descripción

Reseñas cinematográficas reales con partición oficial.

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

Reseñas cinematográficas reales con partición oficial.

Esta ficha resume el uso dentro del repositorio y no reemplaza la documentación oficial.
