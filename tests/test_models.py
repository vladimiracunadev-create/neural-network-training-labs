import torch

from neural_labs.models import DQN, build_model


def test_tabular_and_vision_forward_shapes() -> None:
    tabular = build_model("mlp", (12,), 3, {})
    assert tabular(torch.randn(5, 12)).shape == (5, 3)
    binary = build_model("linear", (6,), 2, {})
    assert binary(torch.randn(5, 6)).shape in {(5,), (5, 1)}
    cnn = build_model("cnn", (3, 32, 32), 10, {})
    assert cnn(torch.randn(4, 3, 32, 32)).shape == (4, 10)


def test_sequence_forward_shapes() -> None:
    rnn = build_model("rnn_text", (256,), 2, {"vocab_size": 1000})
    assert rnn(torch.randint(0, 1000, (3, 32))).shape in {(3,), (3, 1)}
    transformer = build_model("transformer_text", (256,), 4, {"vocab_size": 1000})
    assert transformer(torch.randint(0, 1000, (3, 32))).shape == (3, 4)
    lstm = build_model("lstm_regression", (24, 8), None, {})
    assert lstm(torch.randn(3, 24, 8)).reshape(-1).shape == (3,)


def test_specialized_forward_shapes_and_gradients() -> None:
    autoencoder = build_model("autoencoder", (20,), 2, {})
    values = autoencoder(torch.randn(5, 20))
    assert values.shape == (5, 20)
    loss = values.square().mean()
    loss.backward()
    assert any(parameter.grad is not None for parameter in autoencoder.parameters())
    dqn = DQN(4, 4)
    assert dqn(torch.randn(6, 4)).shape == (6, 4)
