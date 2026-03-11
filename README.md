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
| `--face-down` | Face-down mode: inverted heightmap, flip in slicer (yes/no) | `no` |
| `--exploded` | Exploded mode: standalone transparent sandwiches per colour (yes/no) | `no` |

Any flag left out will be prompted interactively. A preview window shows the expected result before generating STLs.

## Modes

### Standard (default)

Multi-colour stacked topographical STLs. Requires a multi-material printer. Load all STLs into your slicer and assign each part to its filament.

### Cap Layers (`--cap-layers yes`)

Adds a dark base and clear top layer for enhanced contrast. Best for images with fine detail.

### Face-Down (`--face-down yes`)

Flat voxel grid printed face-down. Flip the print after removal for a smooth viewing surface.

### Exploded (`--exploded yes`)

Each colour becomes a standalone 3-layer sandwich: transparent bottom, colour middle, transparent top. This removes the multi-material printer requirement — each sandwich can be printed individually on any single-material printer, then stacked by hand.

- **No colour limit**: Automatically determines the optimal number of colours from the image using iterative K-means clustering (stops when a new colour adds less than `--min-delta-e` difference). You can also override with `-c`.
- **Binary pixel assignment**: Each pixel is assigned to exactly one colour (the best match).
- **Auto transparent selection**: Picks the most transparent filament from your library automatically.
- **2 STLs per colour**: `_color.stl` (colour pixels at middle layer) and `_transparent.stl` (carrier: full bottom + inverse middle fill + full top).

```bash
# Exploded mode with auto colour count
python3 huecli.py image.png -f filaments.csv -l 0.08 -m 2.0 -n 0.4 -s 120x160 -d 10.0 -o output.stl --exploded yes
```

**Printing each sandwich:**
1. Load both STLs (`_color.stl` and `_transparent.stl`) into your slicer
2. Assign the colour filament to the `_color` part, transparent to the `_transparent` part
3. Print as a single 3-layer print (3 x layer_height tall)
4. Repeat for each colour, then stack all sandwiches and backlight

Cannot be combined with `--cap-layers` or `--face-down`.

## Output

All files are saved to `output/`:
- One STL per colour (e.g. `image_Black.stl`, `image_Orange.stl`), or in exploded mode, two STLs per colour (`_color.stl` + `_transparent.stl`)
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
