from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from greenpatch.tracker import Rect


def estimate_homography(prev_gray, frame_gray, prev_rect: Rect, reproj_thresh=5.0) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    prev_bbox = [
        int(prev_rect.x),
        int(prev_rect.y),
        int(prev_rect.x + prev_rect.width),
        int(prev_rect.y + prev_rect.height),
    ]
    mask = np.zeros(prev_gray.shape, dtype=np.uint8)
    cv2.rectangle(mask, (prev_bbox[0], prev_bbox[1]), (prev_bbox[2], prev_bbox[3]), 255, -1)
    detector = cv2.ORB_create(nfeatures=1500)
    kp1, des1 = detector.detectAndCompute(prev_gray, mask)
    kp2, des2 = detector.detectAndCompute(frame_gray, None)
    if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
        return None, None
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    if len(matches) < 4:
        return None, None
    matches = sorted(matches, key=lambda m: m.distance)[: min(len(matches), 300)]
    src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, reproj_thresh)
    return H, None
