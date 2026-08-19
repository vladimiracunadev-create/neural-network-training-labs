# Datasets reales y gobierno de datos

El repositorio no versiona archivos de datos grandes. Versiona **manifiestos**, reglas de partición y código de preparación. La descarga se realiza desde la fuente declarada por cada laboratorio.

## Catálogo

| Dataset | Fuente | Adaptador | Licencia/condiciones declaradas | Referencia oficial |
|---|---|---|---|---|
| `breast_cancer_wisconsin` | UCI | `uci` | CC BY 4.0 | https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic |
| `banknote_authentication` | UCI | `uci` | Consultar ficha UCI | https://archive.ics.uci.edu/dataset/267/banknote+authentication |
| `dry_bean` | UCI | `uci` | CC BY 4.0 | https://archive.ics.uci.edu/dataset/602/dry+bean+dataset |
| `cifar10` | Torchvision / University of Toronto | `torchvision` | Consultar términos CIFAR-10 | https://www.cs.toronto.edu/~kriz/cifar.html |
| `imdb` | Hugging Face / Stanford | `huggingface` | Consultar dataset card | https://huggingface.co/datasets/stanfordnlp/imdb |
| `seoul_bike` | UCI | `uci` | CC BY 4.0 | https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand |
| `credit_card_fraud` | Kaggle / ULB | `kaggle` | Uso sujeto a términos de Kaggle y autor | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud |
| `ag_news` | Hugging Face | `huggingface` | Consultar dataset card | https://huggingface.co/datasets/fancyzhx/ag_news |
| `fashion_mnist` | Torchvision / Zalando Research | `torchvision` | MIT | https://github.com/zalandoresearch/fashion-mnist |
| `cora` | PyTorch Geometric / Planetoid | `pyg` | Consultar dataset original | https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.datasets.Planetoid.html |
| `online_retail` | UCI | `uci_retail` | CC BY 4.0 | https://archive.ics.uci.edu/dataset/352/online+retail |
| `oxford_iiit_pet` | Torchvision / Oxford | `torchvision` | Uso académico según fuente | https://www.robots.ox.ac.uk/~vgg/data/pets/ |
| `uci_har` | UCI | `uci_har` | CC BY 4.0 | https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones |
| `adult_census` | UCI | `uci` | CC BY 4.0 | https://archive.ics.uci.edu/dataset/2/adult |
| `uci_har_subjects` | UCI | `uci_har` | CC BY 4.0 | https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones |
| `iris` | UCI | `uci` | CC BY 4.0 | https://archive.ics.uci.edu/dataset/53/iris |
| `wine_quality` | UCI | `uci` | CC BY 4.0 | https://archive.ics.uci.edu/dataset/186/wine+quality |
| `california_housing` | scikit-learn / StatLib | `sklearn` | Consultar fuente StatLib | https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html |
| `iranian_churn` | UCI | `uci` | CC BY 4.0 | https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset |

## Registro de fuentes

La tabla de arriba es el **catálogo operativo**: lo que el adaptador necesita para
descargar. La procedencia completa de cada dataset —autoridad que responde por él,
licencia tal como la declara la fuente, versión concreta, SHA-256 de cada artefacto y la
cita académica que pide su autor— vive en el registro de fuentes:

- [`sources/bibliography.json`](https://github.com/vladimiracunadev-create/neural-network-training-labs/blob/main/sources/bibliography.json) — el registro;
- [`sources/README.md`](https://github.com/vladimiracunadev-create/neural-network-training-labs/blob/main/sources/README.md) — cómo se lee y cómo se amplía.

```bash
python scripts/verify-sources     # offline, bloquea en CI
python scripts/refresh-sources    # en red, informa, no bloquea
```

El verificador falla si un dataset entra al catálogo o a una ruta sin entrada en el
registro. Donde esta tabla dice «consultar», el registro dice qué licencia declara
realmente la fuente; cuando no declara ninguna, también lo dice.

## Política de almacenamiento

```text
data/
├── raw/          # caché local de la descarga; ignorada por Git
└── processed/    # hashes, manifiestos y particiones; ignorada por Git
```

Los datasets no se copian al ZIP. Esto evita redistribuir contenido sin autorización, reduce el tamaño del repositorio y permite obtener la versión vigente desde el proveedor.

## Trazabilidad

Cada ejecución guarda `dataset_manifest.json` con:

- nombre y fuente;
- referencia y licencia declarada;
- tamaño de cada partición;
- hash de los identificadores de `train`, `validation` y `test`;
- resumen de columnas, clases o forma de entrada;
- semilla y estrategia de partición;
- metadatos entregados por el proveedor cuando están disponibles.

## Regla de partición

1. Cuando el proveedor publica un test oficial, se conserva.
2. La validación se extrae solamente del entrenamiento oficial.
3. En series temporales se usa orden cronológico.
4. En clasificación tabular se usa partición estratificada cuando es posible.
5. En Cora se usan las máscaras públicas de Planetoid.
6. En HAR federado, los clientes se forman a partir de sujetos reales.
7. En DQN de inventario, la señal de demanda proviene de transacciones reales y se divide cronológicamente.
8. `test` jamás se concatena con `train` o `validation`.

## Datos Kaggle

Kaggle puede exigir autenticación, aceptación de condiciones o ambos. El repositorio no evita esos controles. `kagglehub.dataset_download(...)` se ejecuta con la identidad configurada por el usuario y los archivos permanecen en su caché local.

## Integridad

```bash
neural-labs dataset --lab 05_lstm_time_series
neural-labs audit --lab 05_lstm_time_series
python scripts/audit_splits.py --all
```

`audit_bundle` falla cuando encuentra identificadores compartidos entre particiones.

## Incorporar otra fuente

Una fuente nueva debe aportar:

- procedencia verificable;
- licencia o condiciones explícitas;
- identificadores estables por muestra;
- estrategia de partición documentada;
- preprocesamiento ajustado solo con `train`;
- una prueba de integridad;
- una ficha de limitaciones.

## Fuentes de las especializaciones avanzadas

| Ruta | Dataset | Proveedor | Política |
|---|---|---|---|
| Transformer preentrenado | AG News | Hugging Face Datasets | Test oficial; validación extraída de train |
| Segmentación U-Net | Oxford-IIIT Pet trimaps | Torchvision / Oxford | Test oficial |
| Audio | SpeechCommands v0.02 | Torchaudio | Listas oficiales de validation y testing |
| WGAN-GP y DDPM | Fashion-MNIST | Torchvision / Zalando | Test oficial |
| SimCLR | CIFAR-10 | Torchvision / Toronto | Test oficial; dos vistas solo en train |

Los modelos preentrenados se descargan desde Hugging Face Hub. Ningún peso ni dataset grande se redistribuye dentro del repositorio.
