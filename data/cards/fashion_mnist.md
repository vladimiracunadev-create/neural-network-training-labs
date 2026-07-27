# Dataset Card — fashion_mnist

## Identificación

- **Fuente:** Torchvision / Zalando Research
- **Identificador:** `FashionMNIST`
- **Referencia oficial:** https://github.com/zalandoresearch/fashion-mnist
- **Licencia/condiciones:** MIT
- **Estado de revisión de licencia:** `declared`
- **Usado por:** `08_gan_generation`, `19_regularization_dropout_batchnorm`

## Descripción

No usa anillos ni puntos inventados; entrena con prendas reales etiquetadas.

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

No usa anillos ni puntos inventados; entrena con prendas reales etiquetadas.

Esta ficha resume el uso dentro del repositorio y no reemplaza la documentación oficial.
