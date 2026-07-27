# Dataset Card — oxford_iiit_pet

## Identificación

- **Fuente:** Torchvision / Oxford
- **Identificador:** `OxfordIIITPet`
- **Referencia oficial:** https://www.robots.ox.ac.uk/~vgg/data/pets/
- **Licencia/condiciones:** Uso académico según fuente
- **Estado de revisión de licencia:** `review_required`
- **Usado por:** `11_transfer_learning`

## Descripción

7.349 imágenes reales de 37 razas de perros y gatos.

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

7.349 imágenes reales de 37 razas de perros y gatos.

Esta ficha resume el uso dentro del repositorio y no reemplaza la documentación oficial.
