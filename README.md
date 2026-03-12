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
  --mode cap-layers
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
| `--mode` | Generation mode (see below) | `standard` |
| `--sandwich-layers` | Colour layers per sandwich in exploded modes | `1` (CMYK: `3`) |
| `--base-layers` | Transparent base layers per sandwich in exploded modes | `3` |
| `--fill` | Fill sandwiches with transparent (inverse middle + top layer) | `no` |

**Modes:** `standard`, `cap-layers`, `face-down`, `face-down-cap`, `exploded`, `exploded-multi`, `exploded-cmyk`

Any flag left out will be prompted interactively. A preview window shows the expected result before generating STLs.

## Modes

### Standard (default)

Multi-colour stacked topographical STLs. Requires a multi-material printer (e.g. AMS/MMU). Load all STLs into your slicer and assign each part to its filament.

```bash
python3 huecli.py image.png -f filaments.csv -c 5 -n 0.4 -l 0.08 -m 2.0 -s 120x160 -o output.stl
```

### Cap Layers (`--mode cap-layers`)

Adds a dark base and clear top layer for enhanced contrast. Best for images with fine detail. Requires a multi-material printer.

```bash
python3 huecli.py image.png -f filaments.csv -c 5 -n 0.4 -l 0.08 -m 2.0 -s 120x160 -o output.stl --mode cap-layers
```

### Face-Down (`--mode face-down`)

Flat voxel grid printed face-down. Flip the print after removal for a smooth viewing surface. Combine with cap layers using `--mode face-down-cap`.

```bash
python3 huecli.py image.png -f filaments.csv -c 5 -n 0.4 -l 0.08 -m 2.0 -s 120x160 -o output.stl --mode face-down
```

### Exploded (`--mode exploded`)

Each colour gets its own standalone sandwiches: transparent bottom, colour middle, transparent top (when `--fill yes`), or just transparent bottom + colour middle (default). **No multi-material printer required** — each sandwich is printed individually with just one colour + transparent filament, then stacked by hand.

- **Auto colour count**: Iteratively tests K=3..12 colours using Beer-Lambert delta-E scoring. Picks the fewest colours within 2.0 delta-E of the best score. Override with `-c`.
- **Per-colour cap of 3 sandwiches**: Each colour can have up to 3 intensity levels (0/1/2/3 layers per pixel).
- **Auto transparent selection**: Picks the most transparent filament from your library.
- **2 STLs per sandwich**: `_color.stl` (colour pixels) + `_transparent.stl` (bottom plate, plus inverse middle fill and top plate when `--fill yes`).
- **Configurable sandwich thickness**: Use `--sandwich-layers N` to set the number of colour layers per sandwich (default 1).

```bash
# Exploded mode with auto colour count
python3 huecli.py image.png -f filaments.csv -l 0.08 -m 2.0 -n 0.4 -s 120x160 -d 10.0 -o output.stl --mode exploded

# Exploded mode with manual colour count
python3 huecli.py image.png -f filaments.csv -c 8 -l 0.08 -m 2.0 -n 0.4 -s 120x160 -o output.stl --mode exploded
```

**Printing each sandwich:**
1. Load both STLs (`_color.stl` and `_transparent.stl`) into your slicer
2. Assign the colour filament to the `_color` part, transparent to the `_transparent` part
3. Print as a single print (base_layers + sandwich_layers + 1 with `--fill`, base_layers + sandwich_layers without)
4. Repeat for each sandwich, then stack all sandwiches and backlight

### Exploded-Multi (`--mode exploded-multi`)

Higher-fidelity variant of exploded mode. Each sandwich can have **up to 3 different colours** in its middle layer (different pixels, different colours). This requires a multi-material printer or manual filament swaps for each sandwich, but produces better colour accuracy.

- **Higher intensity resolution**: Per-colour cap of 5 (auto-reduced to fit budget). Each colour gets 0-5 intensity levels vs 0-3 in standard exploded.
- **Multi-colour packing**: Colour-level pairs are bin-packed into physical sandwiches (up to 3 non-overlapping colour masks per sandwich), reducing total sandwich count.
- **Auto budget fitting**: If packing exceeds the sandwich budget (model_height / layer_height), caps are iteratively reduced until it fits.
- **Up to 4 STLs per sandwich**: 1 transparent + up to 3 colour STLs, named `_S01_ColorName_color.stl`, `_S01_transparent.stl`, etc.

```bash
# Exploded-multi with auto colour count
python3 huecli.py image.png -f filaments.csv -l 0.08 -m 2.0 -n 0.4 -s 120x160 -d 10.0 -o output.stl --mode exploded-multi

# Exploded-multi with manual colour count
python3 huecli.py image.png -f filaments.csv -c 8 -l 0.08 -m 2.0 -n 0.4 -s 120x160 -o output.stl --mode exploded-multi
```

**Printing each sandwich:**
1. Load all STLs for that sandwich number (`_S01_*.stl`) into your slicer
2. Assign each colour filament to its `_color.stl` part
3. Assign transparent filament to the `_transparent.stl` part
4. Print as a single print, then repeat for each sandwich and stack

### Exploded CMYK (`--mode exploded-cmyk`)

Uses 4 fixed CMYK subtractive primaries (Cyan, Magenta, Yellow, Black) instead of K-means colour selection. Each primary gets up to 2 intensity levels, producing a **maximum of 8 sandwiches** (vs 15-24 for regular exploded with many colours). Fewer sandwiches means less transparent material and a brighter result.

- Auto-selects the best C/M/Y/K filaments from your library by delta-E to ideal primaries
- Uses the same Beer-Lambert simulation as other exploded modes
- Defaults to 3 colour layers per sandwich (`--sandwich-layers 3`) for visible colour from high-TD filaments
- Default no-fill (`--fill no`): sandwiches are just bottom transparent + colour layers, minimising transparent material

```bash
python3 huecli.py image.png -f filaments.csv -n 0.4 -l 0.08 -m 2.0 -s 120x160 -o output.stl --mode exploded-cmyk
```

**Printing:** Same as regular exploded — print each sandwich separately on any single-material printer, then stack all sandwiches and backlight. Each sandwich is 6 layers tall by default (3 base + 3 colour), or 7 with `--fill yes`.

## Exploded Mode Options

These flags apply to all exploded modes (`exploded`, `exploded-multi`, `exploded-cmyk`):

| Flag | Description | Default |
|------|-------------|---------|
| `--sandwich-layers N` | Colour layers in the middle of each sandwich. Higher values give more colour opacity per sandwich, useful for high-TD (translucent) filaments. | `1` (`exploded-cmyk`: `3`) |
| `--base-layers N` | Transparent base layers at the bottom of each sandwich. More layers = more structural rigidity for freestanding prints. | `3` |
| `--fill yes/no` | When `yes`, fills non-colour areas in the middle with transparent and adds a top transparent layer. When `no`, sandwiches are just bottom transparent + colour pixels. | `no` |

## Output

All files are saved to `output/`:
- **Standard/Cap/Face-down**: One STL per colour (e.g. `image_Black.stl`, `image_Orange.stl`)
- **Exploded**: Two STLs per colour per level (e.g. `image_Black_1_color.stl` + `image_Black_1_transparent.stl`)
- **Exploded-multi**: Up to 4 STLs per sandwich (e.g. `image_S01_Black_color.stl` + `image_S01_transparent.stl`)
- **Exploded-CMYK**: Two STLs per primary per level (e.g. `image_Aquatic_Blue_1_color.stl` + `image_Aquatic_Blue_1_transparent.stl`)
- A `.txt` file with printing instructions

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
