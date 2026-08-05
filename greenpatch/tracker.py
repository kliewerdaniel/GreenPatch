from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    width: float
    height: float


@dataclass
class TrackResult:
    frame_index: int
    target_rect: Rect
    source_rect: Rect
    target_mask: np.ndarray
    source_mask: np.ndarray
    H_target: Optional[np.ndarray] = None
    H_source: Optional[np.ndarray] = None
