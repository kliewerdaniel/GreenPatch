import cv2
import numpy as np

from greenpatch.algorithms.seamless_clone import warp_source_patch


def test_warp_source_patch_resizes():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    source_rect = type("obj", (object,), {"x": 10, "y": 10, "width": 20, "height": 20})()
    target_rect = type("obj", (object,), {"x": 40, "y": 40, "width": 30, "height": 30})()
    out = warp_source_patch(frame, source_rect, target_rect, None)
    assert out.shape == (30, 30, 3)


def test_warp_source_patch_empty():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    source_rect = type("obj", (object,), {"x": 200, "y": 200, "width": 10, "height": 10})()
    target_rect = type("obj", (object,), {"x": 10, "y": 10, "width": 20, "height": 20})()
    out = warp_source_patch(frame, source_rect, target_rect, None)
    assert out is None or out.size == 0
