# Agregar un laboratorio

## 1. Registrar el dataset

Agregue una entrada a `configs/labs.yaml` y regenere `configs/datasets.yaml`. La fuente debe ser pública, verificable y contar con licencia o condiciones consultables.

## 2. Implementar o reutilizar el adaptador

El adaptador debe devolver `DataBundle` con identificadores únicos y tres particiones. No descargue datos desde el notebook.

## 3. Agregar el modelo

Use `build_model` cuando sea una arquitectura reutilizable. Las variantes especializadas pueden implementarse en `experiments.py`, manteniendo selección por validación.

## 4. Crear la carpeta

```text
labs/XX_nombre/
├── README.md
├── train.py
├── notebook.ipynb
├── configs/
│   ├── baseline.yaml
│   └── improved.yaml
└── data/
    └── dataset.yaml
```

## 5. Línea base y métricas

Declare una referencia sencilla y métricas que correspondan al problema. Evite usar accuracy como único criterio en datos desbalanceados.

## 6. Pruebas

Agregue pruebas de:

- forma de entrada y salida;
- pérdida finita y gradientes;
- integridad de particiones;
- guardado y carga;
- ejecución rápida;
- ausencia de archivos grandes versionados.

## 7. Validar

```bash
python scripts/validate_repository.py
pytest -q
python scripts/smoke_test.py --lab XX_nombre
```
