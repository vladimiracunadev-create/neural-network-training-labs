import torch

from neural_labs.models import Autoencoder, SensorFusionNet, build_model


def test_autoencoder_and_sensor_fusion() -> None:
    autoencoder = Autoencoder(10, 3)
    assert autoencoder(torch.randn(4, 10)).shape == (4, 10)
    fusion = SensorFusionNet(6)
    assert fusion(torch.randn(2, 9, 128)).shape == (2, 6)
    registered = build_model("sensor_fusion", (9, 128), 6, {})
    assert registered(torch.randn(1, 9, 128)).shape == (1, 6)
