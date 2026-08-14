# 🗺️ Índice del recorrido

> 🧭 [📘 Portada del repositorio](../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/) · [🖥️ Índice HTML offline](../index.html)

Las **31 rutas** se estudian en orden, de la **00** a la **30**.
Las siete partes de abajo son tramos **contiguos** de esa misma secuencia: cada una agrupa
las clases consecutivas que comparten propósito, y termina justo donde empieza la siguiente.

| Parte | Título | Rutas | Clases | Qué llevas al terminar |
|:---:|---|:---:|:---:|---|
| 🟢 **1** | [Fundamentos: de la derivada a la primera red](01-fundamentos.md) | 00–02 | 3 | entiendes qué calcula, qué deriva y qué actualiza un entrenamiento. |
| 🔵 **2** | [Arquitecturas según la forma del dato](02-arquitecturas.md) | 03–07 | 5 | eliges arquitectura por la forma del problema, no por la moda. |
| 🟣 **3** | [Familias especializadas: generar, decidir, relacionar](03-familias-especializadas.md) | 08–12 | 5 | evalúas sistemas que no tienen una única etiqueta correcta. |
| 🟠 **4** | [Entrenar mejor, más barato y sin centralizar datos](04-entrenamiento-eficiente.md) | 13–15 | 3 | mejoras un modelo sin tocar `test` y sabes qué cuesta cada mejora. |
| 🔴 **5** | [La mecánica fina, ahora en profundidad](05-mecanica-fina.md) | 16–20 | 5 | explicas por qué un entrenamiento converge, se estanca o sobreajusta. |
| ⚫ **6** | [Confiar en el modelo y sacarlo del cuaderno](06-confianza-y-despliegue.md) | 21–24 | 4 | respondes «¿por qué predijo esto?», «¿cuánto te fías?» y «¿cuánto tarda?». |
| 🔬 **7** | [Especializaciones avanzadas](07-especializaciones-avanzadas.md) | 25–30 | 6 | trabajas con arquitecturas actuales sin renunciar al protocolo. |

## 📚 Todas las clases, en orden

| # | Clase | Parte | Dataset |
|---:|---|---|---|
| 00 | 🔢 [Neurona con NumPy](../labs/00_numpy_neuron/README.md) | 🟢 [1](01-fundamentos.md) | `breast_cancer_wisconsin` |
| 01 | 🧩 [Perceptrón con PyTorch](../labs/01_pytorch_perceptron/README.md) | 🟢 [1](01-fundamentos.md) | `banknote_authentication` |
| 02 | 🌀 [MLP multiclase](../labs/02_mlp_nonlinear/README.md) | 🟢 [1](01-fundamentos.md) | `dry_bean` |
| 03 | 🖼️ [CNN para visión](../labs/03_cnn_vision/README.md) | 🔵 [2](02-arquitecturas.md) | `cifar10` |
| 04 | 🔁 [RNN para texto](../labs/04_rnn_sequences/README.md) | 🔵 [2](02-arquitecturas.md) | `imdb` |
| 05 | 📈 [LSTM para series temporales](../labs/05_lstm_time_series/README.md) | 🔵 [2](02-arquitecturas.md) | `seoul_bike` |
| 06 | 🧬 [Autoencoder para fraude](../labs/06_autoencoder_anomaly/README.md) | 🔵 [2](02-arquitecturas.md) | `credit_card_fraud` |
| 07 | 🔭 [Transformer para noticias](../labs/07_transformer_attention/README.md) | 🔵 [2](02-arquitecturas.md) | `ag_news` |
| 08 | 🎨 [GAN generativa](../labs/08_gan_generation/README.md) | 🟣 [3](03-familias-especializadas.md) | `fashion_mnist` |
| 09 | 🕸️ [GNN sobre red de citas](../labs/09_gnn_graphs/README.md) | 🟣 [3](03-familias-especializadas.md) | `cora` |
| 10 | 🕹️ [DQN para inventario con demanda real](../labs/10_dqn_reinforcement/README.md) | 🟣 [3](03-familias-especializadas.md) | `online_retail` |
| 11 | ♻️ [Transfer learning con mascotas](../labs/11_transfer_learning/README.md) | 🟣 [3](03-familias-especializadas.md) | `oxford_iiit_pet` |
| 12 | 🔀 [Fusión de sensores](../labs/12_multimodal_fusion/README.md) | 🟣 [3](03-familias-especializadas.md) | `uci_har` |
| 13 | 🎛️ [Búsqueda de hiperparámetros](../labs/13_hyperparameter_search/README.md) | 🟠 [4](04-entrenamiento-eficiente.md) | `adult_census` |
| 14 | ⚗️ [Destilación de conocimiento](../labs/14_knowledge_distillation/README.md) | 🟠 [4](04-entrenamiento-eficiente.md) | `cifar10` |
| 15 | 🌐 [Aprendizaje federado por participante](../labs/15_federated_learning/README.md) | 🟠 [4](04-entrenamiento-eficiente.md) | `uci_har_subjects` |
| 16 | ∂ [Backpropagation manual](../labs/16_backpropagation_manual/README.md) | 🔴 [5](05-mecanica-fina.md) | `iris` |
| 17 | 📐 [Activaciones y funciones de pérdida](../labs/17_activations_and_losses/README.md) | 🔴 [5](05-mecanica-fina.md) | `wine_quality` |
| 18 | ⚙️ [Optimizadores y schedulers](../labs/18_optimizers_and_schedulers/README.md) | 🔴 [5](05-mecanica-fina.md) | `california_housing` |
| 19 | 🛡️ [Regularización](../labs/19_regularization_dropout_batchnorm/README.md) | 🔴 [5](05-mecanica-fina.md) | `fashion_mnist` |
| 20 | 🔄 [Aumento de datos](../labs/20_data_augmentation/README.md) | 🔴 [5](05-mecanica-fina.md) | `cifar10` |
| 21 | 🔍 [Explicabilidad](../labs/21_explainability/README.md) | ⚫ [6](06-confianza-y-despliegue.md) | `adult_census` |
| 22 | 🎯 [Incertidumbre y calibración](../labs/22_uncertainty_calibration/README.md) | ⚫ [6](06-confianza-y-despliegue.md) | `breast_cancer_wisconsin` |
| 23 | 📦 [Exportación e inferencia](../labs/23_model_export_and_inference/README.md) | ⚫ [6](06-confianza-y-despliegue.md) | `cifar10` |
| 24 | 🏁 [Proyecto final: churn de telecomunicaciones](../labs/24_capstone_real_project/README.md) | ⚫ [6](06-confianza-y-despliegue.md) | `iranian_churn` |
| 25 | 🔧 [Fine-tuning eficiente de transformer](../advanced_labs/25_transformer_finetuning/README.md) | 🔬 [7](07-especializaciones-avanzadas.md) | `ag_news` |
| 26 | 🧷 [Segmentación semántica con U-Net](../advanced_labs/26_segmentation_unet/README.md) | 🔬 [7](07-especializaciones-avanzadas.md) | `oxford_iiit_pet_segmentation` |
| 27 | 🎙️ [Clasificación de audio con SpeechCommands](../advanced_labs/27_audio_speechcommands/README.md) | 🔬 [7](07-especializaciones-avanzadas.md) | `speechcommands_v0.02` |
| 28 | 🖌️ [WGAN-GP sobre Fashion-MNIST](../advanced_labs/28_wgan_gp/README.md) | 🔬 [7](07-especializaciones-avanzadas.md) | `fashion_mnist` |
| 29 | 🌫️ [Difusión DDPM sobre Fashion-MNIST](../advanced_labs/29_diffusion_ddpm/README.md) | 🔬 [7](07-especializaciones-avanzadas.md) | `fashion_mnist` |
| 30 | 🪞 [Aprendizaje autosupervisado SimCLR](../advanced_labs/30_self_supervised_simclr/README.md) | 🔬 [7](07-especializaciones-avanzadas.md) | `cifar10` |

---

[📘 Portada del repositorio](../README.md) · [▶️ Empezar por la ruta 00](../labs/00_numpy_neuron/README.md)
