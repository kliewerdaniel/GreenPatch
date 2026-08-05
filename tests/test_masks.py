import numpy as np
from greenpatch.config import build_rect_mask, apply_padding


def test_build_rect_mask():
    shape = (100, 200)
    mask = build_rect_mask(shape, (10, 20, 30, 40))
    assert mask.shape == shape
    assert mask[20:60, 10:40].sum() == 255 * 30 * 40


def test_apply_padding():
    rect = apply_padding((10, 20, 30, 40), 5, (100, 100))
    assert rect[0] == 5
    assert rect[2] == 40
