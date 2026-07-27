# Dataset Card — cifar10

## Identificación

- **Fuente:** Torchvision / University of Toronto
- **Identificador:** `CIFAR10`
- **Referencia oficial:** https://www.cs.toronto.edu/~kriz/cifar.html
- **Licencia/condiciones:** Consultar términos CIFAR-10
- **Estado de revisión de licencia:** `review_required`
- **Usado por:** `03_cnn_vision`, `14_knowledge_distillation`, `20_data_augmentation`, `23_model_export_and_inference`

## Descripción

CIFAR-10 contiene 60.000 imágenes reales de 32×32.

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

CIFAR-10 contiene 60.000 imágenes reales de 32×32.

Esta ficha resume el uso dentro del repositorio y no reemplaza la documentación oficial.
