# HueCLI - Multi-Colour 3D Print STL Generator

Convert images into stacked topographical STL files using Beer-Lambert colour simulation.

## How It Works

HueCLI extracts dominant colours from an image via K-means clustering, then selects filaments using physics-aware Beer-Lambert scoring — each candidate filament is evaluated at its estimated print thickness, not just its raw colour. Selection is automatically refined via simulated annealing. One STL is generated per filament, with per-pixel thickness varying based on image brightness, creating colour through light transmission (Beer-Lambert physics). Filaments are sorted dark-to-light with layer height proportional to transmission distance.

## Installation

```bash
pip3 install -e .
```

## Quick Start

```bash
# Interactive mode — prompts for anything not provided
python3 -m huecli image.png

# Fully non-interactive
python3 -m huecli image.png -f data/filaments.csv -c 5 -n 0.4 -l 0.04 -m 3.0 -s 120x160 -o output.stl --mode standard
```

A 3D preview opens in your browser before generating STLs. From the preview you can generate, adjust colours (reduce delta-E), or cancel.

## 3D Preview

The browser preview is an interactive Three.js viewer that renders a downsampled version of your model.

**Controls:**

- **Drag** to rotate
- **Scroll** to zoom
- **Right-drag** to pan

**Toolbar (bottom of screen):**

- **Gap slider** — separates layers along the Z axis so you can inspect individual colour bands. Defaults to 0 for standard/flat modes (showing the finished product) and 2mm for exploded modes.
- **Realistic / Filaments toggle** — switches between two colour modes:
  - **Realistic** — each layer shows the cumulative front-lit appearance of all layers beneath it. Separate layers with the gap slider to see how colour builds up from bottom to top.
  - **Filaments** — shows each layer in its raw filament colour so you can see which filament is assigned where. Automatically applies a small gap so all layers are visible.
- **Background picker** — row of colour dots next to the toggle. Changes the scene background colour (dark gray, white, mid-gray, warm beige, slate) so transparent regions and edges are easier to inspect.
- **i button** — click for a description of each colour mode.

## Which Mode Should I Use?

| Situation | Recommended Mode |
|-----------|-----------------|
| Multi-material printer, general use | `standard` |
| Multi-material, flat slabs with multi-colour mixing | `flat` or `flat-cap` |
| Single-material printer | `exploded` |
| Multi-material, highest fidelity stacking | `exploded-multi` |
| Subtractive CMYK colour mixing | `exploded-cmyk` |

## Modes

### Standard (default)

Stacked topographical STLs — one per colour. Load all into your slicer and assign filaments.

### Flat (`--mode flat`)

Flat uniform-thickness colour slabs. For each pixel, brute-force tests all 2^N filament presence/absence combinations to find the subset that best reproduces the target colour via Beer-Lambert transmission. Multiple colours can stack on the same pixel (e.g., red under blue = purple).

### Flat-Cap (`--mode flat-cap`)

Same as flat + transparent cap layer on top and transparent fill in gaps. Reserves one filament slot for transparent. `--base-layers` controls cap thickness (default 1).

### Exploded (`--mode exploded`)

Each colour gets standalone sandwiches (transparent base + colour middle). **No multi-material printer required** — print each sandwich individually, then stack and backlight.

- **Auto colour count**: tests K=3..12 via Beer-Lambert delta-E scoring, picks fewest colours within 2.0 delta-E of best. Override with `-c`.
- **2 STLs per sandwich**: `_color.stl` + `_transparent.stl`
- Use `--fill yes` to add inverse fill + top transparent layer

### Exploded-Multi (`--mode exploded-multi`)

Up to 3 different colours per sandwich middle layer. Requires multi-material or manual swaps, but fewer total sandwiches and better accuracy.

- Up to 4 STLs per sandwich: 1 transparent + up to 3 colours
- Colour-level pairs are bin-packed into sandwiches to minimise count

### Exploded CMYK (`--mode exploded-cmyk`)

Fixed CMYK primaries (auto-selected from your library by delta-E). Each primary gets up to N intensity levels. Defaults to 3 colour layers per sandwich for high-TD filaments.

## Parameter Reference

### General Flags

| Flag | Description | Default |
|------|-------------|---------|
| `-f, --filaments` | Filament library CSV | `data/filaments.csv` |
| `-c, --colors` | Number of colours | `4` (exploded: auto) |
| `-n, --nozzle` | Nozzle diameter (mm) | `0.2` |
| `-l, --layer-height` | Layer height (mm) | `0.08` |
| `-m, --model-height` | Total model height (mm) | `2.0` |
| `-s, --size` | Print size WxH (mm) | `100x140` |
| `-o, --output` | Output filename | `<input_stem>.stl` |
| `-d, --min-delta-e` | Min colour difference | `5.0` |
| `--mode` | Generation mode | `standard` |
| `--scheme` | Remap image to a colour scheme palette | none |
| `--flip` | Flip image: `horizontal`, `vertical`, or `both` | none |
| `--dither` | Dithering method for flat/exploded modes: `none`, `floyd-steinberg`, `ordered` | `none` |
| `--enhance-detail` | Extend darkest filament through edges for sharper detail (standard mode only) | off |
| `--use-filaments` | Comma-separated filament names to use (bypasses auto-selection and SA) | none |

### Exploded Mode Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--sandwich-layers` | Colour layers per sandwich | `1` (CMYK: `3`) |
| `--base-layers` | Transparent base layers per sandwich | `3` |
| `--max-color-sandwiches` | Max sandwiches per colour | `3` (multi: `5`, CMYK: `1`) |
| `--fill` | Fill non-colour areas with transparent + top layer | `no` |

### Flat-Cap Flag

| Flag | Description | Default |
|------|-------------|---------|
| `--base-layers` | Transparent cap layer count | `1` |

## Colour Schemes

Use `--scheme <name>` to remap your image to a preset colour palette before processing. The image pixels are reassigned to the nearest palette colour (in perceptual LAB space), then the normal pipeline runs on the remapped image. `-c` defaults to the number of colours in the scheme but can be overridden to use fewer.

Works with all modes.

| Scheme | Colours | Palette |
|--------|---------|---------|
| `greyscale` | 5 | ![#000000](https://placehold.co/20x20/000000/000000) ![#404040](https://placehold.co/20x20/404040/404040) ![#808080](https://placehold.co/20x20/808080/808080) ![#B0B0B0](https://placehold.co/20x20/B0B0B0/B0B0B0) ![#FFFFFF](https://placehold.co/20x20/FFFFFF/FFFFFF) |
| `cyberpunk` | 6 | ![#0D0221](https://placehold.co/20x20/0D0221/0D0221) ![#FF00FF](https://placehold.co/20x20/FF00FF/FF00FF) ![#00FFFF](https://placehold.co/20x20/00FFFF/00FFFF) ![#FF6600](https://placehold.co/20x20/FF6600/FF6600) ![#FFFF00](https://placehold.co/20x20/FFFF00/FFFF00) ![#8B00FF](https://placehold.co/20x20/8B00FF/8B00FF) |
| `sepia` | 5 | ![#2B1700](https://placehold.co/20x20/2B1700/2B1700) ![#6B3A1F](https://placehold.co/20x20/6B3A1F/6B3A1F) ![#A0724A](https://placehold.co/20x20/A0724A/A0724A) ![#C4A47A](https://placehold.co/20x20/C4A47A/C4A47A) ![#F5E6C8](https://placehold.co/20x20/F5E6C8/F5E6C8) |
| `sunset` | 5 | ![#1A0533](https://placehold.co/20x20/1A0533/1A0533) ![#8B1A4A](https://placehold.co/20x20/8B1A4A/8B1A4A) ![#E94E3D](https://placehold.co/20x20/E94E3D/E94E3D) ![#F49D37](https://placehold.co/20x20/F49D37/F49D37) ![#FFD662](https://placehold.co/20x20/FFD662/FFD662) |
| `ocean` | 5 | ![#001B2E](https://placehold.co/20x20/001B2E/001B2E) ![#014F6B](https://placehold.co/20x20/014F6B/014F6B) ![#0496A8](https://placehold.co/20x20/0496A8/0496A8) ![#5CC8D4](https://placehold.co/20x20/5CC8D4/5CC8D4) ![#D1F0F0](https://placehold.co/20x20/D1F0F0/D1F0F0) |
| `vaporwave` | 5 | ![#2B0A3D](https://placehold.co/20x20/2B0A3D/2B0A3D) ![#FF71CE](https://placehold.co/20x20/FF71CE/FF71CE) ![#B967FF](https://placehold.co/20x20/B967FF/B967FF) ![#01CDFE](https://placehold.co/20x20/01CDFE/01CDFE) ![#05FFA1](https://placehold.co/20x20/05FFA1/05FFA1) |
| `autumn` | 6 | ![#2D1B00](https://placehold.co/20x20/2D1B00/2D1B00) ![#8B2500](https://placehold.co/20x20/8B2500/8B2500) ![#CC5500](https://placehold.co/20x20/CC5500/CC5500) ![#E09540](https://placehold.co/20x20/E09540/E09540) ![#FFD700](https://placehold.co/20x20/FFD700/FFD700) ![#556B2F](https://placehold.co/20x20/556B2F/556B2F) |
| `nordic` | 5 | ![#1C2833](https://placehold.co/20x20/1C2833/1C2833) ![#4A6A7A](https://placehold.co/20x20/4A6A7A/4A6A7A) ![#8EB8C4](https://placehold.co/20x20/8EB8C4/8EB8C4) ![#C8DDE0](https://placehold.co/20x20/C8DDE0/C8DDE0) ![#F0F5F5](https://placehold.co/20x20/F0F5F5/F0F5F5) |

```bash
# Example: sunset-themed print
python3 -m huecli photo.png --scheme sunset --mode flat -s 120x160
```

## Output Files

All files saved to `output/`:

| Mode | Naming | Example |
|------|--------|---------|
| Standard / Flat / Flat-Cap | One STL per colour | `image_Black.stl` |
| Exploded | 2 STLs per colour per level | `image_Black_1_color.stl`, `image_Black_1_transparent.stl` |
| Exploded-multi | Up to 4 per sandwich | `image_S01_Black_color.stl`, `image_S01_transparent.stl` |
| Exploded-CMYK | 2 per primary per level | `image_Aquatic_Blue_1_color.stl` |

A `.txt` file with printing instructions is also generated.

## Slicer Setup

- **Layer height**: match your `-l` value (default 0.08mm)
- **Infill**: 100% (solid)
- **Top/bottom layers**: 0 (STLs are already capped)
- **Exploded sandwiches**: load both `_color.stl` and `_transparent.stl`, assign filaments, print as one

## Filament Library

Edit `data/filaments.csv`. Required columns: `Brand`, `Type`, `Color` (hex), `Name`, `TD` (transmission distance in mm), `Tags`.

TD controls material needed to show colour: low TD = opaque, high TD = translucent.

## Project Structure

```
print3r/
├── huecli/                        # Python package
│   ├── __init__.py                # Package marker + version
│   ├── __main__.py                # Entry point (python3 -m huecli)
│   ├── config.py                  # Typed dataclasses, COLOR_SCHEMES
│   ├── color_science.py           # Beer-Lambert, sRGB/linear, delta-E, dithering
│   ├── mesh.py                    # Topographical height-fields, greedy flat layers
│   ├── filaments.py               # FilamentLibrary: CSV, selection, SA optimization
│   ├── image.py                   # ImageProcessor: load, resize, smooth, flip
│   ├── preview.py                 # Three.js GLB browser preview
│   ├── generator.py               # STLGenerator facade
│   ├── cli.py                     # Argparse wrapper
│   ├── interactive.py             # Interactive prompts for missing config
│   └── modes/
│       ├── __init__.py            # Registry + dispatcher
│       ├── standard.py            # Brightness heightmap + TD z-bands
│       ├── flat.py                # Flat + flat-cap (Beer-Lambert 2^N)
│       ├── exploded.py            # Single-color sandwiches
│       ├── exploded_multi.py      # Multi-color sandwiches
│       └── exploded_cmyk.py       # CMYK wrapper
├── tests/                         # Unit + regression tests
├── scripts/                       # Dev/debug tools
├── data/                          # Filament library CSVs
├── examples/                      # Sample input images
├── output/                        # Generated STLs
├── setup.py
├── requirements.txt
└── CLAUDE.md
```

## Testing

```bash
# Unit tests
python3 -m pytest tests/

# Regression tests (preview scene metrics vs baselines)
python3 tests/test_regression.py
```

## License

MIT
