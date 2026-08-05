# GreenPatch

GreenPatch is an AI-assisted greenscreen repair tool that automatically restores small subject regions accidentally keyed away by matching the green background (reflective tin foil, chrome, jewelry, glasses, metallic costumes, or green reflections).

```bash
pip install greenpatch
greenpatch repair input.mp4 output.mp4
```

## Features

- Manual first-frame selection with OpenCV viewer
- Multiple trackers: optical flow, ORB, planar homography
- Multiple blend modes: copy, feather, seamless
- Automatic green contamination detection
- Optional AI modules for future extension
- YAML configuration

## Documentation

See `docs/` for architecture, API docs, examples, and installation guide.
