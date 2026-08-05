# API Documentation

## `greenpatch.cli.repair(...)`

Main command. Starts the first-frame selector, then tracks and repairs the video.

Parameters:
- `input`: input video path
- `output`: output video path
- `tracker`: `optical`, `orb`, or `planar`
- `blend`: `copy`, `feather`, or `seamless`
- `padding`: padding added to tracked regions
- `feather`: feather radius for feather blending
- `config_path`: optional YAML config

## `greenpatch.config.RepairConfig`

Dataclass with defaults and `load(path)`/`as_dict()` helpers.

## `greenpatch.video.VideoPipeline`

Context manager around a source video and output writer.
