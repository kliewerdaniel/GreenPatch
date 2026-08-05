import numpy as np
import pytest

from greenpatch.ai.auto_mask import AutoMasker
from greenpatch.ai.segmentation import DummySegmenter
from greenpatch.config import RepairConfig


def test_combined_mask_contains_original():
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    target_mask = np.zeros((64, 64), dtype=np.uint8)
    target_mask[10:20, 10:20] = 255
    cfg = RepairConfig(green_threshold_hue_low=30, green_threshold_hue_high=90)
    auto = AutoMasker(DummySegmenter())
    combined = auto.combined_repair_mask(frame, target_mask, cfg)
    assert combined[15, 15] == 255


def test_dummy_segmenter_returns_mask():
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    mask = DummySegmenter().segment(frame)
    assert mask.shape == frame.shape[:2]
    assert mask.sum() == 0
