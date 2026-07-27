# Entrenamiento distribuido

`train-distributed` funciona bajo `torchrun` y soporta:

- DDP para replicar el modelo y sincronizar gradientes.
- FSDP2 para fragmentar parámetros cuando la versión de PyTorch lo permite.

La semilla base se ajusta por rango para evitar secuencias idénticas, mientras el sampler distribuido mantiene particiones reproducibles.

```bash
torchrun --standalone --nproc-per-node=2 -m neural_labs train-distributed --lab 03_cnn_vision --strategy ddp
```

Cada rango guarda un checkpoint. El rango cero produce el checkpoint principal y el manifiesto. Para producción multinodo se recomienda almacenamiento compartido y `torch.distributed.checkpoint`.
