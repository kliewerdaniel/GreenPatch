# GreenPatch Architecture

```text
greenpatch/
  cli.py         -> Typer CLI entrypoint
  config.py      -> RepairConfig dataclass + helpers
  io.py          -> VideoIO + ffmpeg wrappers
  tracker.py     -> Rect/TrackResult dataclasses
  video.py       -> VideoPipeline / ffmpeg encode helpers
  masks.py       -> green contamination + bounding rect masks
  blend.py       -> copy/feather/seamless blend functions
  algorithms/    -> optical flow, planar tracker, homography, seamless clone
  ai/            -> segmenter interface + auto masker
  ui/            -> OpenCV first-frame selector
```
