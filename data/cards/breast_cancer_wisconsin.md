# Dataset Card — breast_cancer_wisconsin

## Identificación

- **Fuente:** UCI
- **Identificador:** `17`
- **Referencia oficial:** https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- **Licencia/condiciones:** CC BY 4.0
- **Estado de revisión de licencia:** `declared`
- **Usado por:** `00_numpy_neuron`, `22_uncertainty_calibration`

## Descripción

Datos clínicos reales derivados de imágenes digitalizadas de aspirados de masas mamarias.

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

Datos clínicos reales derivados de imágenes digitalizadas de aspirados de masas mamarias.

Esta ficha resume el uso dentro del repositorio y no reemplaza la documentación oficial.
