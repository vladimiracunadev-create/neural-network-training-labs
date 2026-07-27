# Contribuir

## Reglas obligatorias

1. Cree una rama descriptiva y limite el cambio a un objetivo verificable.
2. Instale `.[full,dev]` y active `pre-commit`.
3. No agregue datasets, credenciales, checkpoints ni artefactos grandes.
4. Toda fuente nueva debe ser real, trazable y acompañarse por licencia o condiciones.
5. Nunca sustituya una descarga fallida por datos generados.
6. Mantenga `train`, `validation` y `test` separados.
7. Ajuste transformadores solo con `train`; seleccione con `validation`; evalúe `test` al final.
8. Agregue línea base, métricas, notebook, teoría, plan experimental, rúbrica y pruebas.
9. Documente riesgos, poblaciones no representadas y usos fuera de alcance.
10. Ejecute `make validate`, `make lint` y `make test`.

## Crear un laboratorio

```bash
neural-labs new-lab semantic_segmentation --title "Segmentación semántica"
```

El scaffold no modifica automáticamente el catálogo: el contribuidor debe definir dataset, arquitectura, adaptador, configuración y pruebas conscientemente.

## Pull request

Incluya:

- problema resuelto;
- fuente y licencia del dataset;
- evidencia de cero fuga de datos;
- comandos ejecutados;
- resultados de tests;
- costo y limitaciones;
- capturas o artefactos solo cuando no expongan datos restringidos.

## Definición de terminado

Un laboratorio no está terminado porque el notebook abre. Debe descargar datos reales, ejecutar en modo rápido, producir artefactos, pasar tests y explicar qué aprendió el estudiante.
