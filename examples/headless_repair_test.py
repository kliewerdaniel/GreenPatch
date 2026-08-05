"""Headless end-to-end test for GreenPatch using the REAL repair_video() code path.

Builds a tiny synthetic "greenscreen" clip, performs a fake first-frame selection
(simulating the manual left/right drag), and runs the exact CLI repair loop so we
exercise VideoIO.fps/.width/.height, the trackers, and the writer without a GUI.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from greenpatch.config import RepairConfig
from greenpatch.video import VideoPipeline
from greenpatch.ui.first_frame_selector import FirstFrameSelector
from greenpatch.tracker import Rect, TrackResult
from greenpatch.cli import repair_video

INPUT = Path("/Users/danielkliewer/Downloads/gp_synth.mov")
OUT_DIR = Path("/Users/danielkliewer/Downloads/gp_test")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def make_synthetic_clip(path: Path, n_frames: int = 40, w: int = 320, h: int = 240):
    """A moving green panel with a white target patch and a clean source patch."""
    if path.exists():
        return
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 25, (w, h))
    for i in range(n_frames):
        frame = np.full((h, w, 3), (0, 200, 0), dtype=np.uint8)  # green bg
        dx = int(20 * np.sin(i / 5.0))
        # target (damaged) region, drifting
        cv2.rectangle(frame, (50 + dx, 80), (110 + dx, 140), (255, 255, 255), -1)
        # source clean patch to the right
        cv2.rectangle(frame, (180, 80), (240, 140), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()


def fake_selection(frame, target, source):
    h, w = frame.shape[:2]
    t = Rect(*target)
    s = Rect(*source)
    tmask = np.zeros((h, w), np.uint8)
    smask = np.zeros((h, w), np.uint8)
    cv2.rectangle(tmask, (int(t.x), int(t.y)), (int(t.x + t.width), int(t.y + t.height)), 255, -1)
    cv2.rectangle(smask, (int(s.x), int(s.y)), (int(s.x + s.width), int(s.y + s.height)), 255, -1)
    return type("Sel", (), {"target_rect": t, "source_rect": s, "target_mask": tmask, "source_mask": smask})()


def run(tracker, blend, name):
    out = OUT_DIR / f"synth_{name}.mp4"
    cfg = RepairConfig(tracker=tracker, blend=blend)
    with VideoPipeline(INPUT, out) as pipeline:
        first = next(pipeline.frames())
        sel = fake_selection(first, (50, 80, 60, 60), (180, 80, 60, 60))
        n = repair_video(pipeline, sel, cfg)
    cap = cv2.VideoCapture(str(out))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok = cap.isOpened() and frames > 0
    cap.release()
    print(f"  {name}: wrote {n} frames, output {frames} frames, {out.stat().st_size} bytes -> {'OK' if ok else 'FAIL'}")
    return ok and frames > 0


def main():
    make_synthetic_clip(INPUT)
    print(f"synthetic input: {INPUT}")
    ok1 = run("planar", "seamless", "planar_seamless")
    ok2 = run("optical", "feather", "optical_feather")
    ok3 = run("planar", "copy", "planar_copy")
    print("ALL DONE" if (ok1 and ok2 and ok3) else "SOME FAILED")


if __name__ == "__main__":
    main()
