# Dataset Card — cora

## Identificación

- **Fuente:** PyTorch Geometric / Planetoid
- **Identificador:** `Cora`
- **Referencia oficial:** https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.datasets.Planetoid.html
- **Licencia/condiciones:** Consultar dataset original
- **Estado de revisión de licencia:** `review_required`
- **Usado por:** `09_gnn_graphs`

## Descripción

Usa las máscaras públicas fijas de train, validación y test.

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

Usa las máscaras públicas fijas de train, validación y test.

Esta ficha resume el uso dentro del repositorio y no reemplaza la documentación oficial.
