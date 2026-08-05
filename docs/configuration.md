# Configuration

GreenPatch supports a YAML configuration file via `--config`.

Example `greenpatch.yaml`:
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

CLI options override the YAML values.
