import numpy as np

from greenpatch.config import RepairConfig
from greenpatch.ai.auto_mask import AutoMasker
from greenpatch.ai.segmentation import DummySegmenter


def test_auto_masker_combined_mask():
    frame = np.full((50, 50, 3), (0, 255, 0), dtype=np.uint8)
    target_mask = np.zeros((50, 50), dtype=np.uint8)
    target_mask[10:20, 10:20] = 255
    cfg = RepairConfig(green_threshold_hue_low=30, green_threshold_hue_high=90)
    auto = AutoMasker(DummySegmenter())
    combined = auto.combined_repair_mask(frame, target_mask, cfg)
    assert combined.sum() >= target_mask.sum()
