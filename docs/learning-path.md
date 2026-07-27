# Ruta de aprendizaje

1. **Neurona con NumPy** — `00_numpy_neuron`: Implementar propagación, entropía cruzada y descenso de gradiente sin autograd.
2. **Perceptrón con PyTorch** — `01_pytorch_perceptron`: Aprender tensores, autograd, optimizadores y un clasificador lineal.
3. **MLP multiclase** — `02_mlp_nonlinear`: Resolver clasificación no lineal con capas densas, activaciones y regularización.
4. **CNN para visión** — `03_cnn_vision`: Entrenar una CNN y analizar errores sobre fotografías reales de diez clases.
5. **RNN para texto** — `04_rnn_sequences`: Clasificar sentimiento en reseñas reales usando embeddings y recurrencia.
6. **LSTM para series temporales** — `05_lstm_time_series`: Pronosticar demanda horaria respetando el orden temporal.
7. **Autoencoder para fraude** — `06_autoencoder_anomaly`: Detectar transacciones fraudulentas mediante error de reconstrucción.
8. **Transformer para noticias** — `07_transformer_attention`: Aplicar atención multi-cabeza a clasificación de noticias reales.
9. **GAN generativa** — `08_gan_generation`: Generar prendas a partir de imágenes reales de Fashion-MNIST.
10. **GNN sobre red de citas** — `09_gnn_graphs`: Clasificar publicaciones científicas usando texto y enlaces de citas.
11. **DQN para inventario con demanda real** — `10_dqn_reinforcement`: Aprender una política de reposición usando una secuencia de demanda observada en transacciones reales.
12. **Transfer learning con mascotas** — `11_transfer_learning`: Comparar extracción de características, fine-tuning y entrenamiento desde cero.
13. **Fusión de sensores** — `12_multimodal_fusion`: Fusionar acelerómetro y giroscopio de smartphones para reconocer actividades.
14. **Búsqueda de hiperparámetros** — `13_hyperparameter_search`: Optimizar profundidad, ancho, dropout y learning rate sin tocar test.
15. **Destilación de conocimiento** — `14_knowledge_distillation`: Transferir conocimiento de una CNN profesora a una estudiante compacta.
16. **Aprendizaje federado por participante** — `15_federated_learning`: Aplicar FedAvg usando participantes reales como clientes naturales.
17. **Backpropagation manual** — `16_backpropagation_manual`: Derivar y programar backpropagation en una MLP pequeña.
18. **Activaciones y funciones de pérdida** — `17_activations_and_losses`: Comparar ReLU, GELU, Tanh y pérdidas apropiadas en clases desbalanceadas.
19. **Optimizadores y schedulers** — `18_optimizers_and_schedulers`: Comparar SGD, Momentum, Adam y reducción de tasa de aprendizaje.
20. **Regularización** — `19_regularization_dropout_batchnorm`: Medir dropout, weight decay y batch normalization.
21. **Aumento de datos** — `20_data_augmentation`: Comparar recortes, volteos y perturbaciones sobre imágenes reales.
22. **Explicabilidad** — `21_explainability`: Explicar predicciones con Integrated Gradients y permutación.
23. **Incertidumbre y calibración** — `22_uncertainty_calibration`: Medir confianza, Brier score, ECE y temperature scaling.
24. **Exportación e inferencia** — `23_model_export_and_inference`: Exportar ONNX, validar paridad y medir latencia por lotes.
25. **Proyecto final: churn de telecomunicaciones** — `24_capstone_real_project`: Resolver de extremo a extremo un problema real de abandono de clientes con documentación, evaluación y despliegue.

## Proyecto final

El laboratorio 24 integra descarga gobernada, análisis tabular, línea base, MLP, selección por validación, evaluación final, model card y reporte. Debe ampliarse con evaluación de sesgo, costos de error y monitoreo de deriva antes de cualquier aplicación real.

## Especializaciones avanzadas

26. **Fine-tuning eficiente** — `25_transformer_finetuning`: comparar DistilBERT completo y LoRA sobre AG News.
27. **Segmentación U-Net** — `26_segmentation_unet`: segmentar mascota, fondo y borde en Oxford-IIIT Pet.
28. **Audio SpeechCommands** — `27_audio_speechcommands`: clasificar comandos reales mediante espectrogramas log-mel.
29. **WGAN-GP** — `28_wgan_gp`: estudiar estabilidad generativa sobre Fashion-MNIST.
30. **Difusión DDPM** — `29_diffusion_ddpm`: aprender predicción de ruido y muestreo iterativo.
31. **SimCLR** — `30_self_supervised_simclr`: preentrenar representaciones de CIFAR-10 y evaluarlas con linear probe.
