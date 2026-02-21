# HueCLI - Multi-Colour 3D Print STL Generator

Generate multi-colour 3D print STLs from images. Converts images into stacked topographical STL files using Beer-Lambert colour simulation.

## How It Works

HueCLI converts an image into one STL per filament colour. Each colour layer sits directly on top of the previous one with per-pixel varying thickness based on image brightness. The cumulative stack creates colour through light transmission (Beer-Lambert physics).

- Extracts dominant colours via K-means clustering
- Matches colours to your filament library using CIEDE2000 deltaE
- Sorts filaments dark-to-light and allocates layer height proportional to transmission distance
- Applies adaptive contrast enhancement and Beer-Lambert gamma curves
- Generates watertight shared-vertex grid meshes (manifold, no gaps)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Interactive mode (prompts for all parameters)
python3 huecli.py image.png

# Non-interactive with all flags
python3 huecli.py image.png \
  -f filaments.csv \
  -c 5 \
  -n 0.4 \
  -l 0.04 \
  -m 3.0 \
  -s 120x160 \
  -o output_name.stl \
  -d 10.0 \
  --cap-layers yes
```

| Flag | Description | Default |
|------|-------------|---------|
| `-f, --filaments` | Filament library CSV path | `filaments.csv` |
| `-c, --colors` | Number of colours/filaments | `4` |
| `-n, --nozzle` | Nozzle diameter (mm) | `0.2` |
| `-l, --layer-height` | Layer height (mm) | `0.08` |
| `-m, --model-height` | Total model height (mm) | `2.0` |
| `-s, --size` | Print size WxH (mm) | `100x140` |
| `-o, --output` | Output filename | `<input_stem>.stl` |
| `-d, --min-delta-e` | Min colour difference (delta-E) | `5.0` |
| `--cap-layers` | Cap layers: black base + auto colours + clear top (yes/no) | `no` |

Any flag left out will be prompted interactively. A preview window shows the expected result before generating STLs.

## Output

All files are saved to `output/`:
- One STL per colour (e.g. `image_Black.stl`, `image_Orange.stl`)
- A `.txt` file with printing instructions

Load all STLs into your slicer, right-click each part to assign the corresponding filament.

## Filament Library

Edit `filaments.csv` to match your available filaments. Each row needs:
- `Brand`, `Type`, `Color` (hex), `Name`, `TD` (transmission distance in mm), `Tags`

Transmission distance (TD) controls how much material is needed to show the colour. Low TD = opaque, high TD = translucent.

## Slicer Setup

- Layer height: match what you specified (default 0.08mm)
- Infill: 100% (solid)
- Top/bottom layers: 0 (STLs are already capped)

## License

MIT
