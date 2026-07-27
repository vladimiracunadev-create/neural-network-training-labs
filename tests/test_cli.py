from neural_labs.cli import build_parser


def test_catalog_command_parses() -> None:
    args = build_parser().parse_args(["catalog"])
    assert args.command == "catalog"


def test_train_command_parses() -> None:
    args = build_parser().parse_args(["train", "--lab", "00_numpy_neuron", "--quick", "--device", "cpu"])
    assert args.lab == "00_numpy_neuron"
    assert args.quick is True
