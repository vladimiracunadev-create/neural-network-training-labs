# Dataset Card — ag_news

## Identificación

- **Fuente:** Hugging Face
- **Identificador:** `fancyzhx/ag_news`
- **Referencia oficial:** https://huggingface.co/datasets/fancyzhx/ag_news
- **Licencia/condiciones:** Consultar dataset card
- **Estado de revisión de licencia:** `review_required`
- **Usado por:** `07_transformer_attention`

## Descripción

Noticias reales en cuatro categorías con particiones públicas.

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

Noticias reales en cuatro categorías con particiones públicas.

Esta ficha resume el uso dentro del repositorio y no reemplaza la documentación oficial.
