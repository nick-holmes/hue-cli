# HueCLI - Multi-Colour 3D Print STL Generator

Convert images into stacked topographical STL files using Beer-Lambert colour simulation.

## How It Works

HueCLI extracts dominant colours from an image via K-means clustering, matches them to your filament library using CIEDE2000 delta-E, then generates one STL per filament. Each layer's thickness varies per-pixel based on image brightness, creating colour through light transmission (Beer-Lambert physics). Filaments are sorted dark-to-light with layer height proportional to transmission distance.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Interactive mode — prompts for anything not provided
python3 huecli.py image.png

# Fully non-interactive
python3 huecli.py image.png -f filaments.csv -c 5 -n 0.4 -l 0.04 -m 3.0 -s 120x160 -o output.stl --mode standard
```

A 3D preview opens in your browser before generating STLs. From the preview you can generate, adjust colours (reduce delta-E), or cancel.

## Which Mode Should I Use?

| Situation | Recommended Mode |
|-----------|-----------------|
| Multi-material printer, general use | `standard` |
| Multi-material, fine detail / high contrast | `cap-layers` |
| Multi-material, smooth viewing surface | `face-down` or `face-down-cap` |
| Single-material printer | `exploded` |
| Multi-material, highest fidelity stacking | `exploded-multi` |
| Subtractive CMYK colour mixing | `exploded-cmyk` |

## Modes

### Standard (default)

Stacked topographical STLs — one per colour. Load all into your slicer and assign filaments.

### Cap Layers (`--mode cap-layers`)

Dark base + shaped colour middles + clear top. Enhanced contrast for fine detail. The `--base-layers` flag controls the number of flat base and top layers (default 2).

### Face-Down (`--mode face-down`)

Inverted heightmap (dark=tall). Flip after printing for a smooth viewing surface. Combine with `--mode face-down-cap` for cap layers + face-down.

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
| `-f, --filaments` | Filament library CSV | `filaments.csv` |
| `-c, --colors` | Number of colours | `4` (exploded: auto) |
| `-n, --nozzle` | Nozzle diameter (mm) | `0.2` |
| `-l, --layer-height` | Layer height (mm) | `0.08` |
| `-m, --model-height` | Total model height (mm) | `2.0` |
| `-s, --size` | Print size WxH (mm) | `100x140` |
| `-o, --output` | Output filename | `<input_stem>.stl` |
| `-d, --min-delta-e` | Min colour difference | `5.0` |
| `--mode` | Generation mode | `standard` |

### Exploded Mode Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--sandwich-layers` | Colour layers per sandwich | `1` (CMYK: `3`) |
| `--base-layers` | Transparent base layers per sandwich | `3` |
| `--max-color-sandwiches` | Max sandwiches per colour | `3` (multi: `5`, CMYK: `1`) |
| `--fill` | Fill non-colour areas with transparent + top layer | `no` |

### Cap Layers / Face-Down-Cap Flag

| Flag | Description | Default |
|------|-------------|---------|
| `--base-layers` | Flat base/top layer count | `2` |

## Output Files

All files saved to `output/`:

| Mode | Naming | Example |
|------|--------|---------|
| Standard / Cap / Face-down | One STL per colour | `image_Black.stl` |
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

Edit `filaments.csv`. Required columns: `Brand`, `Type`, `Color` (hex), `Name`, `TD` (transmission distance in mm), `Tags`.

TD controls material needed to show colour: low TD = opaque, high TD = translucent.

## License

MIT
