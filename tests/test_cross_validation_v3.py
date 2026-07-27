import pytest

from neural_labs.cross_validation import walk_forward_windows


def test_walk_forward_windows() -> None:
    windows = walk_forward_windows(100, minimum_train=40, validation_size=10, step=10)
    assert len(windows) == 6
    assert windows[0][0] == slice(0, 40)
    assert windows[-1][1] == slice(90, 100)
    with pytest.raises(ValueError):
        walk_forward_windows(10, minimum_train=0, validation_size=2, step=1)
