# GreenPatch

GreenPatch repairs small **subject regions accidentally keyed away** by a green
screen — reflective foil, chrome, jewelry, glasses, metallic costumes, or green
reflections that matched the background. You select the damaged spot and a clean
source patch on frame 0; GreenPatch tracks both regions through the clip, clones
the source onto the target, and blends it back in.

```bash
pip install greenpatch
greenpatch repair input.mp4 output.mp4
```

A repo-local usage skill lives at [`SKILL.md`](SKILL.md) — written for both humans
and AI agents. The `docs/` folder has deeper reference material
([installation](docs/installation.md), [configuration](docs/configuration.md),
[architecture](docs/architecture.md), [API](docs/api.md)).

---

## Install

```bash
cd /path/to/GreenPatch
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Requirements: Python 3.12+, OpenCV, NumPy, FFmpeg on `PATH`, Typer, Rich, PyYAML,
Pillow. (On macOS, if an agent shell leaks `PYTHONPATH`/`PYTHONHOME`, create the
venv with `env -u PYTHONPATH -u PYTHONHOME python3.12 -m venv .venv` so a stray
3.11 numpy doesn't break the 3.12 cv2.)

---

## Quickstart

```bash
python -m greenpatch.cli repair input.mp4 output.mp4
```

1. A window titled **greenpatch** opens on frame 0.
2. **Left-drag** a box around the damaged region.
3. **Right-drag** a box around a nearby clean patch to clone from (similar
   lighting/texture).
4. Press **Space** to accept and repair the whole clip.
   - `R` resets the selection, `Esc`/`Q` quits, middle-drag pans, scroll zooms.

| Action | Control |
| --- | --- |
| Draw damaged (target) region | Left drag |
| Draw source patch | Right drag |
| Pan | Middle mouse drag |
| Zoom | Scroll wheel |
| Accept selection | Space |
| Reset selection | R |
| Quit | Esc or Q |

---

## CLI options

```bash
greenpatch repair INPUT OUTPUT [OPTIONS]
```

| Option | Default | Description |
| --- | --- | --- |
| `INPUT` | — (required) | Source video. |
| `OUTPUT` | — (required) | Repaired video to write. |
| `--tracker`, `-t` | `planar` | Tracking backend: `planar` or `optical`. (`orb` is not wired yet.) |
| `--blend`, `-b` | `seamless` | Blend mode: `copy`, `feather`, `seamless`. |
| `--padding`, `-p` | `15` | Padding (px) around selected regions. |
| `--feather` | `8` | Feather radius (px) for `feather` mode. |
| `--config`, `-c` | — | Path to a YAML config file (overrides defaults below). |
| `--no-audio` | off | Skip copying the original audio track (output stays silent). |

Examples:

```bash
greenpatch repair in.mp4 out.mp4 --tracker optical --blend feather --feather 12
greenpatch repair in.mp4 out.mp4 --config greenpatch.yaml
greenpatch repair in.mp4 out.mp4 --no-audio      # keep silent output
```

### Trackers
- **`planar`** (default) — Lucas–Kanade corner points + homography. Robust for
  roughly planar regions that rotate/scale with the camera.
- **`optical`** — LK point tracking with median motion. Good for general motion.
- `orb` — validated but **not wired to the CLI** yet; raises an error.

### Blend modes
- **`seamless`** (default) — `cv2.seamlessClone` with a fallback to feathered copy.
- **`feather`** — Gaussian alpha feather of the patch edge.
- **`copy`** — raw patch replacement (hard edge).

---

## Audio

GreenPatch's frame writer emits a **silent** clip, then re-muxes the **original
audio track** from the source into the output with FFmpeg (stream copy, no
re-encode). So `output.mp4` keeps the source's sound by default. Use `--no-audio`
to skip the mux and keep the silent clip. If the input has no audio, the mux is
skipped automatically and the silent clip is kept.

---

## Configuration (YAML)

`--config greenpatch.yaml` overrides defaults. All keys are optional.

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

See [`docs/configuration.md`](docs/configuration.md) for the full list of tunable
thresholds (corner detection, homography RANSAC, etc.).

---

## Architecture

- `cli.py` — Typer CLI; `repair` runs selection → `repair_video()` (track → clone →
  blend per frame) → optional audio mux.
- `ui/first_frame_selector.py` — interactive first-frame region selector.
- `algorithms/` — `optical_flow`, `planar_tracker`, `homography`, `seamless_clone`.
- `blend.py` — `copy` / `feather` / `seamless` blend modes.
- `masks.py` — green-contamination mask + rect extraction.
- `ai/` — segmentation + `AutoMasker` (auto repair mask); `auto` CLI mode is a stub.

Deeper detail in [`docs/architecture.md`](docs/architecture.md).

---

## Verify / test

```bash
python3.12 -m compileall greenpatch tests examples
python3.12 -m pytest tests -q
python -m greenpatch.cli repair --help
```

`examples/headless_repair_test.py` drives the real `repair_video()` path without a
GUI (synthetic clip + fake selection, all tracker/blend combos). Run it with the
venv active:

```bash
python examples/headless_repair_test.py
```

---

## Limitations

- The interactive selector needs a display (or virtual framebuffer); the rest of
  the pipeline can run headless.
- `orb` tracker and the `auto` (fully automatic) mode are not implemented yet.
- Audio is copied losslessly from the source; if the source uses an audio codec
  FFmpeg can't copy into the chosen container, the mux falls back to keeping the
  silent clip (with a warning).
