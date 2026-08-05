from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import typer
from rich.console import Console

from greenpatch.config import RepairConfig, validate_blend, validate_tracker, apply_padding, dilate_mask
from greenpatch.video import VideoPipeline, encode_frames_to_video
from greenpatch.ui.first_frame_selector import FirstFrameSelector
from greenpatch.tracker import Rect, TrackResult
from greenpatch.ai.auto_mask import AutoMasker
from greenpatch.blend import blend_copies, blend_feather, blend_seamless
from greenpatch.algorithms.optical_flow import track_optical
from greenpatch.algorithms.planar_tracker import track_planar

console = Console()
app = typer.Typer(add_completion=False, help="GreenPatch: automatic greenscreen repair for reflective/damaged regions.")


def _track_next(prev_gray, frame_gray, prev: TrackResult, cfg: RepairConfig) -> TrackResult:
    if cfg.tracker == "orb":
        raise ValueError("ORB backend is not wired to CLI yet; use optical or planar.")

    if cfg.tracker == "optical":
        target_rect, _ = track_optical(prev_gray, frame_gray, prev.target_rect, cfg.max_corners, cfg.quality_level, cfg.min_distance, cfg.block_size)
        source_rect, _ = track_optical(prev_gray, frame_gray, prev.source_rect, cfg.max_corners, cfg.quality_level, cfg.min_distance, cfg.block_size)
        H_target = H_source = None
    else:
        target_rect, H_target = track_planar(prev_gray, frame_gray, prev.target_rect, cfg.max_corners, cfg.quality_level, cfg.min_distance, cfg.block_size, cfg.homography_reproj_thresh)
        source_rect, H_source = track_planar(prev_gray, frame_gray, prev.source_rect, cfg.max_corners, cfg.quality_level, cfg.min_distance, cfg.block_size, cfg.homography_reproj_thresh)

    target_rect = Rect(
        max(0.0, target_rect.x),
        max(0.0, target_rect.y),
        max(1.0, target_rect.width),
        max(1.0, target_rect.height),
    )
    source_rect = Rect(
        max(0.0, source_rect.x),
        max(0.0, source_rect.y),
        max(1.0, source_rect.width),
        max(1.0, source_rect.height),
    )
    frame_h, frame_w = frame_gray.shape
    target_rect = Rect(
        min(target_rect.x, frame_w - target_rect.width),
        min(target_rect.y, frame_h - target_rect.height),
        min(target_rect.width, frame_w),
        min(target_rect.height, frame_h),
    )
    source_rect = Rect(
        min(source_rect.x, frame_w - source_rect.width),
        min(source_rect.y, frame_h - source_rect.height),
        min(source_rect.width, frame_w),
        min(source_rect.height, frame_h),
    )

    target_mask = np.zeros(prev.target_mask.shape, dtype=np.uint8)
    source_mask = np.zeros(prev.source_mask.shape, dtype=np.uint8)
    tx, ty, tw, th = int(target_rect.x), int(target_rect.y), int(target_rect.width), int(target_rect.height)
    sx, sy, sw, sh = int(source_rect.x), int(source_rect.y), int(source_rect.width), int(source_rect.height)
    target_mask[ty : ty + th, tx : tx + tw] = 255
    source_mask[sy : sy + sh, sx : sx + sw] = 255
    return TrackResult(prev.frame_index + 1, target_rect, source_rect, target_mask, source_mask, H_target, H_source)


@app.command()
def repair(
    input: Path = typer.Argument(..., exists=True, help="Input video path."),
    output: Path = typer.Argument(..., help="Output repaired video path."),
    tracker: str = typer.Option("planar", "--tracker", "-t", help="Tracker backend: optical, orb, planar."),
    blend: str = typer.Option("seamless", "--blend", "-b", help="Blend mode: copy, feather, seamless."),
    padding: int = typer.Option(15, "--padding", "-p", help="Padding around selected regions."),
    feather: int = typer.Option(8, "--feather", help="Feather radius for feather mode."),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c", help="YAML config file."),
) -> None:
    validate_tracker(tracker)
    validate_blend(blend)
    cfg = RepairConfig.load(config_path)
    cfg = RepairConfig(
        tracker=tracker,
        blend=blend,
        padding=padding,
        feather=feather,
        green_threshold_hue_low=cfg.green_threshold_hue_low,
        green_threshold_hue_high=cfg.green_threshold_hue_high,
        green_threshold_sat_low=cfg.green_threshold_sat_low,
        green_threshold_sat_high=cfg.green_threshold_sat_high,
        green_threshold_val_low=cfg.green_threshold_val_low,
        green_threshold_val_high=cfg.green_threshold_val_high,
    )
    with VideoPipeline(input, output) as pipeline:
        first = next(pipeline.frames())
        selector = FirstFrameSelector(first, cfg)
        selection = selector.run()
        if selection is None:
            console.print("[yellow]Quit before selection.[/yellow]")
            raise typer.Exit()
        repair_video(pipeline, selection, cfg)
    console.print(f"[green]Wrote repaired video -> {output}[/green]")


def repair_video(pipeline: "VideoPipeline", selection, cfg: RepairConfig) -> int:
    """Process every frame after the first: track, clone the source patch onto the
    target region, blend, and write. Returns the number of frames written."""
    first = None
    prev_gray = None
    frame_h = frame_w = 0
    written = 0
    for frame in pipeline.frames():
        if first is None:
            first = frame
            prev_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
            frame_h, frame_w = prev_gray.shape
            prev = TrackResult(0, selection.target_rect, selection.source_rect, selection.target_mask, selection.source_mask)
            pipeline.write_frame(first)
            written += 1
            continue
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev = _track_next(prev_gray, frame_gray, prev, cfg)

        # Bounds-guard the source slice; if the tracker drifted it off-frame, fall
        # back to the previous frame's content for this frame instead of crashing.
        sy0 = max(0, int(prev.source_rect.y))
        sx0 = max(0, int(prev.source_rect.x))
        sy1 = min(frame_h, int(prev.source_rect.y + prev.source_rect.height))
        sx1 = min(frame_w, int(prev.source_rect.x + prev.source_rect.width))
        src_slice = frame[sy0:sy1, sx0:sx1]
        if src_slice.size == 0:
            pipeline.write_frame(frame)
            written += 1
            prev_gray = frame_gray
            continue

        ty0 = max(0, int(prev.target_rect.y))
        tx0 = max(0, int(prev.target_rect.x))
        th = max(1, min(int(prev.target_rect.height), frame_h - ty0))
        tw = max(1, min(int(prev.target_rect.width), frame_w - tx0))
        src_patch = cv2.resize(src_slice, (tw, th), interpolation=cv2.INTER_LINEAR)
        ty1, tx1 = ty0 + th, tx0 + tw
        patched = frame.copy()
        patched[ty0:ty1, tx0:tx1] = src_patch

        if cfg.blend == "copy":
            out = blend_copies(patched, frame, prev.target_mask)
        elif cfg.blend == "feather":
            out = blend_feather(patched, frame, prev.target_mask, cfg.feather)
        else:
            out = blend_seamless(frame, patched, prev.target_mask, prev.target_rect)
        pipeline.write_frame(out)
        prev_gray = frame_gray
        written += 1
    return written


@app.command()
def auto(input: Path = typer.Argument(...), output: Path = typer.Argument(...)) -> None:
    console.print("[yellow]Auto mode is not implemented yet. Use repair with manual selection.[/yellow]")
    raise typer.Exit(code=1)


def main():
    app()


if __name__ == "__main__":
    main()
