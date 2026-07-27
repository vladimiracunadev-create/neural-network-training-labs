# Especializaciones avanzadas

El proyecto agrega seis rutas con datasets públicos reales y protocolos específicos.

| Ruta | Dataset | Modelo | Evaluación principal |
|---|---|---|---|
| Transformer preentrenado | AG News | DistilBERT completo o LoRA | accuracy, macro-F1, parámetros entrenables |
| Segmentación | Oxford-IIIT Pet trimaps | U-Net | IoU por clase, mean IoU, Dice |
| Audio | SpeechCommands v0.02 | CNN sobre log-mel | accuracy, macro-F1, robustez al ruido |
| Generación adversarial | Fashion-MNIST | WGAN-GP | estabilidad, diversidad y proxy de distancia |
| Difusión | Fashion-MNIST | DDPM compacto | MSE de ruido, diversidad y latencia de muestreo |
| Autosupervisión | CIFAR-10 | SimCLR + linear probe | NT-Xent y accuracy del probe |

```bash
neural-labs advanced
neural-labs train-advanced --track 26_segmentation_unet --quick
```

Cada carpeta bajo `advanced_labs/` contiene teoría, plan experimental, rúbrica, manifiesto de datos y tres notebooks.
