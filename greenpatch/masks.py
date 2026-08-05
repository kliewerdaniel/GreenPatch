from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def mask_to_rect(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 25:
        return None
    x, y, w, h = cv2.boundingRect(cnt)
    return x, y, w, h


def green_contamination_mask(
    frame: np.ndarray,
    target_mask: np.ndarray,
    hue_low: int = 35,
    hue_high: int = 95,
    sat_low: int = 20,
    sat_high: int = 255,
    val_low: int = 20,
    val_high: int = 255,
) -> np.ndarray:
    blurred = cv2.medianBlur(frame, 3)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    lower = np.array([hue_low, sat_low, val_low], dtype=np.uint8)
    upper = np.array([hue_high, sat_high, val_high], dtype=np.uint8)
    green = cv2.inRange(hsv, lower, upper)
    green = cv2.bitwise_and(green, target_mask)
    green = cv2.dilate(green, np.ones((3, 3), np.uint8), iterations=1)
    return green
