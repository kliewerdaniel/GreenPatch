from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from greenpatch.tracker import Rect


def warp_source_patch(frame, source_rect: Rect, target_rect: Rect, H: Optional[np.ndarray]):
    src = _crop_by_rect(frame, source_rect)
    if src.size == 0:
        return None
    dst_shape = (int(target_rect.width), int(target_rect.height))
    if dst_shape[0] <= 0 or dst_shape[1] <= 0:
        return None
    if H is not None and _homography_shape_ok(H, src.shape[:2], dst_shape):
        return cv2.warpPerspective(src, H, dst_shape)
    return cv2.resize(src, dst_shape, interpolation=cv2.INTER_LINEAR)


def _crop_by_rect(image, rect: Rect):
    x = int(max(0, rect.x))
    y = int(max(0, rect.y))
    x2 = int(min(image.shape[1], rect.x + rect.width))
    y2 = int(min(image.shape[0], rect.y + rect.height))
    if x2 <= x or y2 <= y:
        return np.zeros((0, 0, 3), dtype=image.dtype)
    return image[y:y2, x:x2].copy()


def _homography_shape_ok(H, src_shape, dst_shape):
    try:
        h, w = src_shape
        pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
        return transformed.min(axis=0).min() >= -100 and transformed[:, 0].max() < w + 100 and transformed[:, 1].max() < h + 100
    except Exception:
        return False
