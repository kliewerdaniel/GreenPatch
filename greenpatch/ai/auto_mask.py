from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from greenpatch.ai.segmentation import BaseSegmenter, DummySegmenter
from greenpatch.masks import green_contamination_mask


class AutoMasker:
    def __init__(self, segmenter: Optional[BaseSegmenter] = None) -> None:
        self.segmenter = segmenter or DummySegmenter()

    def auto_target_mask(self, frame, prompt: Optional[str] = None) -> np.ndarray:
        mask = self.segmenter.segment(frame, prompt=prompt)
        if mask is None or mask.sum() == 0:
            return np.zeros(frame.shape[:2], dtype=np.uint8)
        return mask

    def combined_repair_mask(self, frame, target_mask: np.ndarray, config) -> np.ndarray:
        green = green_contamination_mask(
            frame,
            target_mask,
            hue_low=config.green_threshold_hue_low,
            hue_high=config.green_threshold_hue_high,
            sat_low=config.green_threshold_sat_low,
            sat_high=config.green_threshold_sat_high,
            val_low=config.green_threshold_val_low,
            val_high=config.green_threshold_val_high,
        )
        return cv2.bitwise_or(target_mask, green)
