from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class BaseSegmenter(ABC):
    @abstractmethod
    def segment(self, frame: np.ndarray, prompt: Optional[str] = None) -> np.ndarray:
        raise NotImplementedError


class DummySegmenter(BaseSegmenter):
    def segment(self, frame: np.ndarray, prompt=None) -> np.ndarray:
        return np.zeros(frame.shape[:2], dtype=np.uint8)
