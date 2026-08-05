import cv2
import numpy as np
from greenpatch.io import VideoIO, run_ffmpeg
from pathlib import Path


def test_video_io_missing(tmp_path):
    try:
        VideoIO(Path("does_not_exist.mp4"))
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError")
