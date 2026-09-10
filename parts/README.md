# 🗺️ Índice del recorrido

> 🧭 [📘 Portada del repositorio](../README.md) · [🌐 Sitio de estudio](https://vladimiracunadev-create.github.io/neural-network-training-labs/) · [🖥️ Índice HTML offline](../index.html)

Las **31 clases** se estudian en orden, de la **01** a la **31**.
Los siete módulos de abajo son tramos **contiguos** de esa misma secuencia: cada uno agrupa
las clases consecutivas que comparten propósito, y termina justo donde empieza la siguiente.

| Módulo | Título | Clases | Cantidad | Qué llevas al terminar |
|:---:|---|:---:|:---:|---|
| 🟢 **1** | [Fundamentos: de la derivada a la primera red](01-fundamentos.md) | 01–03 | 3 | entiendes qué calcula, qué deriva y qué actualiza un entrenamiento. |
| 🔵 **2** | [Arquitecturas según la forma del dato](02-arquitecturas.md) | 04–08 | 5 | eliges arquitectura por la forma del problema, no por la moda. |
| 🟣 **3** | [Familias especializadas: generar, decidir, relacionar](03-familias-especializadas.md) | 09–13 | 5 | evalúas sistemas que no tienen una única etiqueta correcta. |
| 🟠 **4** | [Entrenar mejor, más barato y sin centralizar datos](04-entrenamiento-eficiente.md) | 14–16 | 3 | mejoras un modelo sin tocar `test` y sabes qué cuesta cada mejora. |
| 🔴 **5** | [La mecánica fina, ahora en profundidad](05-mecanica-fina.md) | 17–21 | 5 | explicas por qué un entrenamiento converge, se estanca o sobreajusta. |
| ⚫ **6** | [Confiar en el modelo y sacarlo del cuaderno](06-confianza-y-despliegue.md) | 22–25 | 4 | respondes «¿por qué predijo esto?», «¿cuánto te fías?» y «¿cuánto tarda?». |
| 🔬 **7** | [Especializaciones avanzadas](07-especializaciones-avanzadas.md) | 26–31 | 6 | trabajas con arquitecturas actuales sin renunciar al protocolo. |

## 📚 Todas las clases, en orden

| # | Clase | Módulo | Dataset |
|---:|---|---|---|
| 01 | 🔢 [Neurona con NumPy](../labs/00_numpy_neuron/README.md) | 🟢 [1](01-fundamentos.md) | `breast_cancer_wisconsin` |
| 02 | 🧩 [Perceptrón con PyTorch](../labs/01_pytorch_perceptron/README.md) | 🟢 [1](01-fundamentos.md) | `banknote_authentication` |
| 03 | 🌀 [MLP multiclase](../labs/02_mlp_nonlinear/README.md) | 🟢 [1](01-fundamentos.md) | `dry_bean` |
| 04 | 🖼️ [CNN para visión](../labs/03_cnn_vision/README.md) | 🔵 [2](02-arquitecturas.md) | `cifar10` |
| 05 | 🔁 [RNN para texto](../labs/04_rnn_sequences/README.md) | 🔵 [2](02-arquitecturas.md) | `imdb` |
| 06 | 📈 [LSTM para series temporales](../labs/05_lstm_time_series/README.md) | 🔵 [2](02-arquitecturas.md) | `seoul_bike` |
| 07 | 🧬 [Autoencoder para fraude](../labs/06_autoencoder_anomaly/README.md) | 🔵 [2](02-arquitecturas.md) | `credit_card_fraud` |
| 08 | 🔭 [Transformer para noticias](../labs/07_transformer_attention/README.md) | 🔵 [2](02-arquitecturas.md) | `ag_news` |
| 09 | 🎨 [GAN generativa](../labs/08_gan_generation/README.md) | 🟣 [3](03-familias-especializadas.md) | `fashion_mnist` |
| 10 | 🕸️ [GNN sobre red de citas](../labs/09_gnn_graphs/README.md) | 🟣 [3](03-familias-especializadas.md) | `cora` |
| 11 | 🕹️ [DQN para inventario con demanda real](../labs/10_dqn_reinforcement/README.md) | 🟣 [3](03-familias-especializadas.md) | `online_retail` |
| 12 | ♻️ [Transfer learning con mascotas](../labs/11_transfer_learning/README.md) | 🟣 [3](03-familias-especializadas.md) | `oxford_iiit_pet` |
| 13 | 🔀 [Fusión de sensores](../labs/12_multimodal_fusion/README.md) | 🟣 [3](03-familias-especializadas.md) | `uci_har` |
| 14 | 🎛️ [Búsqueda de hiperparámetros](../labs/13_hyperparameter_search/README.md) | 🟠 [4](04-entrenamiento-eficiente.md) | `adult_census` |
| 15 | ⚗️ [Destilación de conocimiento](../labs/14_knowledge_distillation/README.md) | 🟠 [4](04-entrenamiento-eficiente.md) | `cifar10` |
| 16 | 🌐 [Aprendizaje federado por participante](../labs/15_federated_learning/README.md) | 🟠 [4](04-entrenamiento-eficiente.md) | `uci_har_subjects` |
| 17 | ∂ [Backpropagation manual](../labs/16_backpropagation_manual/README.md) | 🔴 [5](05-mecanica-fina.md) | `iris` |
| 18 | 📐 [Activaciones y funciones de pérdida](../labs/17_activations_and_losses/README.md) | 🔴 [5](05-mecanica-fina.md) | `wine_quality` |
| 19 | ⚙️ [Optimizadores y schedulers](../labs/18_optimizers_and_schedulers/README.md) | 🔴 [5](05-mecanica-fina.md) | `california_housing` |
| 20 | 🛡️ [Regularización](../labs/19_regularization_dropout_batchnorm/README.md) | 🔴 [5](05-mecanica-fina.md) | `fashion_mnist` |
| 21 | 🔄 [Aumento de datos](../labs/20_data_augmentation/README.md) | 🔴 [5](05-mecanica-fina.md) | `cifar10` |
| 22 | 🔍 [Explicabilidad](../labs/21_explainability/README.md) | ⚫ [6](06-confianza-y-despliegue.md) | `adult_census` |
| 23 | 🎯 [Incertidumbre y calibración](../labs/22_uncertainty_calibration/README.md) | ⚫ [6](06-confianza-y-despliegue.md) | `breast_cancer_wisconsin` |
| 24 | 📦 [Exportación e inferencia](../labs/23_model_export_and_inference/README.md) | ⚫ [6](06-confianza-y-despliegue.md) | `cifar10` |
| 25 | 🏁 [Proyecto final: churn de telecomunicaciones](../labs/24_capstone_real_project/README.md) | ⚫ [6](06-confianza-y-despliegue.md) | `iranian_churn` |
| 26 | 🔧 [Fine-tuning eficiente de transformer](../advanced_labs/25_transformer_finetuning/README.md) | 🔬 [7](07-especializaciones-avanzadas.md) | `ag_news` |
| 27 | 🧷 [Segmentación semántica con U-Net](../advanced_labs/26_segmentation_unet/README.md) | 🔬 [7](07-especializaciones-avanzadas.md) | `oxford_iiit_pet_segmentation` |
| 28 | 🎙️ [Clasificación de audio con SpeechCommands](../advanced_labs/27_audio_speechcommands/README.md) | 🔬 [7](07-especializaciones-avanzadas.md) | `speechcommands_v0.02` |
| 29 | 🖌️ [WGAN-GP sobre Fashion-MNIST](../advanced_labs/28_wgan_gp/README.md) | 🔬 [7](07-especializaciones-avanzadas.md) | `fashion_mnist` |
| 30 | 🌫️ [Difusión DDPM sobre Fashion-MNIST](../advanced_labs/29_diffusion_ddpm/README.md) | 🔬 [7](07-especializaciones-avanzadas.md) | `fashion_mnist` |
| 31 | 🪞 [Aprendizaje autosupervisado SimCLR](../advanced_labs/30_self_supervised_simclr/README.md) | 🔬 [7](07-especializaciones-avanzadas.md) | `cifar10` |

---

[📘 Portada del repositorio](../README.md) · [▶️ Empezar por la clase 01](../labs/00_numpy_neuron/README.md)
