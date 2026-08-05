from __future__ import annotations

import cv2
import numpy as np

from greenpatch.tracker import Rect


def track_optical(prev_gray, frame_gray, prev_rect: Rect, max_corners, quality_level, min_distance, block_size):
    corners = cv2.goodFeaturesToTrack(prev_gray, max_corners, quality_level, min_distance, blockSize=block_size)
    if corners is None:
        return prev_rect, None

    new_corners, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, corners, None)
    valid = status.flatten() == 1
    prev_pts = corners[valid]
    curr_pts = new_corners[valid]
    if len(prev_pts) < 4:
        return prev_rect, None

    dx = curr_pts[:, 0, 0] - prev_pts[:, 0, 0]
    dy = curr_pts[:, 0, 1] - prev_pts[:, 0, 1]
    tx = float(np.median(dx))
    ty = float(np.median(dy))
    return Rect(prev_rect.x + tx, prev_rect.y + ty, prev_rect.width, prev_rect.height), None


def track_orb(prev_gray, frame_gray, prev_rect: Rect, n_features=2000):
    prev_bbox = [
        int(prev_rect.x),
        int(prev_rect.y),
        int(prev_rect.x + prev_rect.width),
        int(prev_rect.y + prev_rect.height),
    ]
    mask = np.zeros(prev_gray.shape, dtype=np.uint8)
    cv2.rectangle(mask, (prev_bbox[0], prev_bbox[1]), (prev_bbox[2], prev_bbox[3]), 255, -1)
    detector = cv2.ORB_create(nfeatures=n_features)
    kp1, des1 = detector.detectAndCompute(prev_gray, mask)
    if des1 is None or len(kp1) < 8:
        return prev_rect, None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, detector.detectAndCompute(frame_gray, None)[1])
    if len(matches) < 8:
        return prev_rect, None
    matches = sorted(matches, key=lambda m: m.distance)[: min(len(matches), 200)]
    src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp1[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC)
    if M is None:
        return prev_rect, None
    dx, dy = float(M[0, 2]), float(M[1, 2])
    return Rect(prev_rect.x + dx, prev_rect.y + dy, prev_rect.width, prev_rect.height), M
