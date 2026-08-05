from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from greenpatch.tracker import Rect


def track_planar(
    prev_gray,
    frame_gray,
    prev_rect: Rect,
    max_corners=500,
    quality_level=0.01,
    min_distance=10,
    block_size=7,
    reproj_thresh=5.0,
) -> Tuple[Rect, Optional[np.ndarray]]:
    prev_bbox = [
        int(prev_rect.x),
        int(prev_rect.y),
        int(prev_rect.x + prev_rect.width),
        int(prev_rect.y + prev_rect.height),
    ]
    mask = np.zeros(prev_gray.shape, dtype=np.uint8)
    cv2.rectangle(mask, (prev_bbox[0], prev_bbox[1]), (prev_bbox[2], prev_bbox[3]), 255, -1)
    corners = cv2.goodFeaturesToTrack(
        prev_gray, max_corners, quality_level, min_distance, mask=mask, blockSize=block_size
    )
    if corners is None:
        return prev_rect, None

    new_corners, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, corners, None)
    valid = status.flatten() == 1
    if valid.sum() < 4:
        return prev_rect, None

    src = corners[valid].reshape(-1, 1, 2)
    dst = new_corners[valid].reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, reproj_thresh)
    if H is None:
        return prev_rect, None

    x, y, w, h = prev_rect.x, prev_rect.y, prev_rect.width, prev_rect.height
    pts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    x_min, y_min = transformed.min(axis=0)
    x_max, y_max = transformed.max(axis=0)
    return Rect(float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)), H
