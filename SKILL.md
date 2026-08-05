---
name: greenpatch
description: "Use when repairing green-screen keyed regions in video with GreenPatch."
version: 1.0.0
author: GreenPatch Authors
license: MIT
metadata:
  hermes:
    tags: [greenpatch, video, repair, tracking, opencv]
    related_skills: []
---

# GreenPatch Usage Skill

## Overview

GreenPatch repairs small subject regions accidentally keyed away by a green background. The default workflow is manual first-frame selection, tracking, and seamless repair.

Current CLI state: `--tracker orb` is still a work-in-progress. Use `optical` or `planar`.

## When to Use

- Repairing greenscreen footage with reflective/damaged regions.
- Running the repo-local CLI and verification commands.
- Extending trackers, blend modes, or AI modules.

## Setup

```bash
cd /path/to/GreenPatch
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
```

On macOS with leaked agent env vars:
```bash
env -u PYTHONPATH -u PYTHONHOME python3.12 -m venv .venv
```

## Tested command sequence

```bash
python3.12 -m compileall greenpatch tests examples
python3.12 -m pytest tests -q
```

## CLI Usage

```bash
greenpatch repair input.mp4 output.mp4
greenpatch repair input.mp4 output.mp4 --tracker optical --blend seamless --padding 15 --feather 8 --config greenpatch.yaml
greenpatch repair input.mp4 output.mp4 --no-audio   # keep silent output (skip audio mux)
```

By default the repaired clip keeps the **original audio track** — the frame writer
emits a silent clip, then FFmpeg re-muxes the source audio in (stream copy). Use
`--no-audio` to skip that step. If the input has no audio, the mux is skipped
automatically.

### First-frame selector

| Action | Control |
| --- | --- |
| Draw damaged region | Left drag |
| Draw source patch | Right drag |
| Pan | Middle mouse |
| Zoom | Scroll wheel |
| Accept selection | Space |
| Reset selection | R |
| Quit | Esc or Q |

### Tracker options

- `planar` — Lucas-Kanade points + homography
- `optical` — dense-ish LK point tracking with median motion
- `orb` — placeholder validation only

### Blend modes

- `copy` — raw patch replacement
- `feather` — Gaussian alpha feathering
- `seamless` — `cv2.seamlessClone` with fallback

## YAML config

`greenpatch.yaml`:
```yaml
tracker: planar
blend: seamless
padding: 15
feather: 8
green_threshold:
  hue_low: 35
  hue_high: 95
  sat_low: 20
  sat_high: 255
  val_low: 20
  val_high: 255
```

## Architecture

See `docs/architecture.md`.

## Extending

New AI modules should implement `greenpatch.ai.segmentation.BaseSegmenter` and plug into `AutoMasker` in `greenpatch/ai/auto_mask.py`.

## Troubleshooting

- Missing `ffmpeg`: `brew install ffmpeg`
- Headless server: use `opencv-python-headless` and ensure a display or virtual framebuffer for the selector
- macOS Python env leaks: strip `PYTHONPATH` and `PYTHONHOME` when creating/running child processes

## Verification checklist

- [ ] `python3.12 -m compileall greenpatch tests examples`
- [ ] `python3.12 -m pytest tests -q`
- [ ] `greenpatch repair --help` shows options (incl. `--no-audio`)
- [ ] `python examples/headless_repair_test.py` writes valid clips for planar/seamless, optical/feather, planar/copy
