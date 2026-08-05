from __future__ import annotations

import subprocess
from pathlib import Path

import cv2


def run_ffmpeg(command: str) -> None:
    import subprocess as _sp

    try:
        proc = _sp.run(command, shell=True, check=True, capture_output=True, text=True)
    except _sp.CalledProcessError as exc:
        raise RuntimeError(
            f"ffmpeg failed (exit {exc.returncode}): {exc.stderr[-2000:]}"
        ) from exc
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr}")


class VideoIO:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.path}")

    @property
    def frame_count(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def fps(self) -> float:
        return float(self.cap.get(cv2.CAP_PROP_FPS))

    @property
    def width(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def has_audio(self) -> bool:
        return bool(self.cap.get(cv2.CAP_PROP_AUDIO_STREAM)) if hasattr(cv2, "CAP_PROP_AUDIO_STREAM") else True

    def read_frame(self):
        ret, frame = self.cap.read()
        return frame if ret else None

    def seek(self, index: int) -> None:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)

    def release(self) -> None:
        self.cap.release()
