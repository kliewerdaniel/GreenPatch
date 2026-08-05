# Installation

Install from source:
```bash
git clone https://github.com/danielkliewer/GreenPatch.git
cd GreenPatch
pip install -e .
```

## Requirements

- Python 3.12+
- OpenCV
- NumPy
- FFmpeg
- Typer
- Rich
- PyYAML
- Pillow

## Optional AI dependencies
```bash
pip install greenpatch[ai]
```

# Configuration

See `docs/configuration.md` for YAML options.

# Troubleshooting

Ensure `ffmpeg` is installed and on your PATH:
```bash
ffmpeg -version
```
