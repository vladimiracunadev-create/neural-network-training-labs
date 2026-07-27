import torch

from neural_labs.domains.graphs.models import GraphModelUnavailable, build_graph_model
from neural_labs.domains.reinforcement.models import DuelingDQN
from neural_labs.domains.time_series.models import TemporalConvNet
from neural_labs.domains.vision.models import DCGANDiscriminator, DCGANGenerator, MobileStudent, TeacherCNN
from neural_labs.models import build_model


def test_modern_vision_models_forward() -> None:
    generator = DCGANGenerator(latent_dim=16)
    fake = generator(torch.randn(2, 16))
    assert fake.shape == (2, 1, 28, 28)
    discriminator = DCGANDiscriminator()
    assert discriminator(fake).shape == (2,)
    student = MobileStudent(3, 10)
    teacher = TeacherCNN(3, 10)
    x = torch.randn(2, 3, 32, 32)
    assert student(x).shape == (2, 10)
    assert teacher(x).shape == (2, 10)


def test_masked_rnn_and_attention_maps() -> None:
    rnn = build_model("rnn_text", (8,), 2, {"vocab_size": 50})
    tokens = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]])
    assert rnn(tokens).shape == (2,)
    transformer = build_model("transformer_text", (5,), 4, {"vocab_size": 50})
    assert transformer(tokens).shape == (2, 4)
    maps = transformer.attention_maps()
    assert maps and maps[0].shape[0] == 2


def test_tcn_and_dueling_dqn() -> None:
    tcn = TemporalConvNet(4, 16)
    assert tcn(torch.randn(3, 12, 4)).shape == (3,)
    dqn = DuelingDQN(4, 5)
    output = dqn(torch.randn(6, 4))
    assert output.shape == (6, 5)


def test_graph_model_missing_extra_is_explicit() -> None:
    model = build_graph_model("gcn", 10, 8, 3)
    assert isinstance(model, GraphModelUnavailable)
    try:
        model(None, None)
    except RuntimeError as exc:
        assert "graph" in str(exc)
