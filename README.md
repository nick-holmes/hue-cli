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
python3 huecli.py image.png
```

You'll be prompted for print parameters (filament library, colours, layer height, model height, print size, etc.). A preview window shows the expected result before generating STLs.

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
