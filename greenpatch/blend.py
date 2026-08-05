from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from greenpatch.tracker import Rect


def blend_copies(patched, frame, target_mask):
    if target_mask is None:
        return patched
    if target_mask.shape[:2] != frame.shape[:2]:
        target_mask = cv2.resize(target_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask3 = target_mask.astype(bool)
    out = frame.copy()
    out[mask3] = patched[mask3]
    return out


def blend_feather(patched, frame, target_mask, feather=8):
    out = blend_copies(patched, frame, target_mask)
    if feather <= 0:
        return out
    feathered_mask = cv2.GaussianBlur(target_mask.astype(np.float32) / 255.0, (feather * 2 + 1, feather * 2 + 1), feather)
    feathered_mask = np.clip(feathered_mask, 0, 1)
    mask3 = feathered_mask[..., None]
    return (mask3 * patched + (1.0 - mask3) * frame).astype(np.uint8)


def blend_seamless(frame, patched, target_mask, target_rect: Rect):
    tx, ty, tw, th = int(target_rect.x), int(target_rect.y), int(target_rect.width), int(target_rect.height)
    tx = max(0, min(tx, frame.shape[1] - 1))
    ty = max(0, min(ty, frame.shape[0] - 1))
    tw = max(1, min(tw, frame.shape[1] - tx))
    th = max(1, min(th, frame.shape[0] - ty))
    src_patch = patched[ty : ty + th, tx : tx + tw]
    mask = cv2.resize(target_mask[ty : ty + th, tx : tx + tw], (tw, th), interpolation=cv2.INTER_NEAREST)
    if mask.sum() == 0:
        return blend_copies(patched, frame, target_mask)
    center = (tw // 2, th // 2)
    try:
        out = frame.copy()
        cv2.seamlessClone(src_patch, out, mask, center, cv2.NORMAL_CLONE)
        return out
    except cv2.error:
        return blend_feather(patched, frame, target_mask, feather=8)
