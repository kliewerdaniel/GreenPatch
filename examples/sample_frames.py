from pathlib import Path

from PIL import Image

from greenpatch.video import VideoPipeline


def sample_frames(input_path: Path, output_dir: Path, count: int = 6) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with VideoPipeline(input_path, output_dir / "dummy.mp4") as pipeline:
        for idx, frame in enumerate(pipeline.frames()):
            if idx >= count:
                break
            Image.fromarray(frame[:, :, ::-1]).save(output_dir / f"frame_{idx + 1:03d}.png")
