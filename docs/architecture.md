# Arquitectura

```text
src/neural_labs/
├── core/
│   ├── protocol.py       # semillas y test lock
│   └── registry.py       # factorías extensibles
├── domains/
│   ├── tabular/
│   ├── vision/
│   ├── text/
│   ├── time_series/
│   ├── graphs/
│   ├── generative/
│   └── reinforcement/
├── datasets.py           # adaptadores de fuentes reales
├── experiments.py        # orquestación compatible
├── inference.py          # contrato y carga de ejecución
├── exporting.py          # ONNX, INT8 y ExecuTorch
├── model_registry.py     # versiones y alias locales
├── mlflow_registry.py    # backend MLflow opcional
├── deployment/api.py     # API FastAPI
├── distributed.py        # contexto DDP/FSDP2
├── distributed_training.py
├── telemetry.py
└── supply_chain.py
```

El registro desacopla nombres de arquitectura de sus implementaciones. Agregar un modelo requiere registrar una factoría en su dominio, sin ampliar una cadena central de `if/elif`.

La orquestación histórica permanece para compatibilidad, pero las nuevas capacidades se construyen sobre contratos separados y módulos por dominio.
