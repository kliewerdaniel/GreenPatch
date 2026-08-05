from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import cv2
import numpy as np

from greenpatch.io import VideoIO


class VideoPipeline:
    def __init__(self, input_path: Path, output_path: Path) -> None:
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.reader = VideoIO(self.input_path)
        self.writer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def prepare_writer(self, frame: np.ndarray, fps: Optional[float] = None) -> None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            fps or self.reader.fps,
            (frame.shape[1], frame.shape[0]),
        )

    def write_frame(self, frame: np.ndarray) -> None:
        if self.writer is None:
            self.prepare_writer(frame)
        self.writer.write(frame)

    def frames(self) -> Iterator[np.ndarray]:
        while True:
            frame = self.reader.read_frame()
            if frame is None:
                break
            yield frame

    def frame_count(self) -> int:
        return self.reader.frame_count

    def fps(self) -> float:
        return self.reader.fps

    def close(self) -> None:
        self.reader.release()
        if self.writer is not None:
            self.writer.release()


def encode_frames_to_video(frame_paths, output_path: Path, fps: float) -> None:
    if not frame_paths:
        return
    first = cv2.imread(str(frame_paths[0]))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (first.shape[1], first.shape[0]))
    for path in frame_paths:
        frame = cv2.imread(str(path))
        writer.write(frame)
    writer.release()


def ffmpeg_extract_frames(input_path: Path, frames_dir: Path) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    cmd = f"ffmpeg -i {input_path} -qscale:v 2 {frames_dir}/%06d.png"
    from greenpatch.io import run_ffmpeg

    run_ffmpeg(cmd)


def ffmpeg_encode(frames_dir: Path, output_path: Path, fps: float, pattern: str = "%06d.png") -> None:
    pattern_path = str(frames_dir / pattern)
    cmd = (
        f"ffmpeg -framerate {fps} -i {pattern_path} -c:v libx264 -crf 18 -pix_fmt yuv420p {output_path}"
    )
    from greenpatch.io import run_ffmpeg

    run_ffmpeg(cmd)
