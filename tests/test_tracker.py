import greenpatch.config as cfg
from greenpatch.tracker import Rect


def test_validate_tracker():
    assert cfg.validate_tracker("planar") == "planar"
    try:
        cfg.validate_tracker("invalid")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_validate_blend():
    assert cfg.validate_blend("seamless") == "seamless"
    try:
        cfg.validate_blend("bad")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_rect_coords():
    assert Rect(1.7, 2.3, 10.6, 20.9).x == 1.7
