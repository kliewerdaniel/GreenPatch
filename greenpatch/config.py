from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from dataclasses import dataclass


@dataclass(frozen=True)
class RepairConfig:
    tracker: str = "planar"
    blend: str = "seamless"
    padding: int = 15
    feather: int = 8
    green_threshold_hue_low: int = 35
    green_threshold_hue_high: int = 95
    green_threshold_sat_low: int = 20
    green_threshold_sat_high: int = 255
    green_threshold_val_low: int = 20
    green_threshold_val_high: int = 255
    max_corners: int = 500
    quality_level: float = 0.01
    min_distance: int = 10
    block_size: int = 7
    orb_features: int = 2000
    homography_reproj_thresh: float = 5.0

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "RepairConfig":
        data = {}
        if path and path.exists():
            with path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def as_dict(self) -> dict:
        return {
            "tracker": self.tracker,
            "blend": self.blend,
            "padding": self.padding,
            "feather": self.feather,
            "green_threshold": {
                "hue_low": self.green_threshold_hue_low,
                "hue_high": self.green_threshold_hue_high,
                "sat_low": self.green_threshold_sat_low,
                "sat_high": self.green_threshold_sat_high,
                "val_low": self.green_threshold_val_low,
                "val_high": self.green_threshold_val_high,
            },
        }


def validate_tracker(value: str) -> str:
    allowed = {"optical", "orb", "planar"}
    if value not in allowed:
        raise ValueError(f"tracker must be one of {sorted(allowed)}")
    return value


def validate_blend(value: str) -> str:
    allowed = {"copy", "feather", "seamless"}
    if value not in allowed:
        raise ValueError(f"blend must be one of {sorted(allowed)}")
    return value


def build_rect_mask(shape, rect) -> np.ndarray:
    mask = np.zeros(shape[:2], dtype=np.uint8)
    x, y, w, h = rect
    mask[int(y) : int(y + h), int(x) : int(x + w)] = 255
    return mask


def dilate_mask(mask: np.ndarray, padding: int) -> np.ndarray:
    if padding <= 0:
        return mask
    import cv2

    kernel = np.ones((padding, padding), np.uint8)
    return cv2.dilate(mask, kernel, iterations=1)


def apply_padding(rect, padding: int, frame_shape):
    x, y, w, h = rect
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(frame_shape[1] - x, w + 2 * padding)
    h = min(frame_shape[0] - y, h + 2 * padding)
    return (x, y, w, h)
