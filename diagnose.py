#!/usr/bin/env python3
"""Diagnostic tool: visualize every stage of the Beer-Lambert color pipeline.

Shows side-by-side images at each processing step to identify where
color inversion occurs.

Usage:
    python3 diagnose.py image.png -f filaments.csv -c 4 -n 0.2 -l 0.12 -m 1.44 -s 100x140
"""

import argparse
import base64
import io
import json
import logging
import sys
import tempfile
import webbrowser
from pathlib import Path

import numpy as np
from PIL import Image

from filaments import FilamentLibrary
from image import ImageProcessor
from color_science import (
    sort_filaments_by_luminosity,
    apply_contrast_enhancement,
    apply_unsharp_mask,
    allocate_layers_td_proportional,
    compute_heightmap,
    render_standard_preview,
    srgb_to_linear,
    linear_to_srgb,
    auto_contrast_preview,
    compute_contrast_params,
    apply_contrast_params,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def to_b64_png(rgb_float):
    """Convert (H,W,3) float RGB to base64 PNG. Flips for display."""
    uint8 = (np.clip(rgb_float, 0, 1) * 255).astype(np.uint8)
    uint8 = np.flipud(uint8)
    img = Image.fromarray(uint8)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def grayscale_to_rgb(gray):
    """Convert (H,W) grayscale to (H,W,3) RGB for display."""
    return np.stack([gray, gray, gray], axis=-1)


def heightmap_to_heatmap(height, max_h):
    """Convert height array to a blue→red heatmap for visualization."""
    normalized = np.clip(height / max(max_h, 1e-6), 0, 1)
    r = normalized
    g = np.zeros_like(normalized)
    b = 1.0 - normalized
    return np.stack([r, g, b], axis=-1)


def main():
    parser = argparse.ArgumentParser(description='Diagnose color pipeline')
    parser.add_argument('image', help='Input image path')
    parser.add_argument('-f', '--filaments', default='filaments.csv')
    parser.add_argument('-c', '--colors', type=int, default=4)
    parser.add_argument('-n', '--nozzle', type=float, default=0.2)
    parser.add_argument('-l', '--layer-height', type=float, default=0.12)
    parser.add_argument('-m', '--model-height', type=float, default=1.44)
    parser.add_argument('-s', '--size', default='100x140')
    parser.add_argument('-d', '--min-delta-e', type=float, default=5.0)
    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"Image not found: {args.image}")
        return 1

    width_mm, height_mm = [float(x) for x in args.size.split('x')]
    num_layers = int(args.model_height / args.layer_height)

    # Load
    filament_lib = FilamentLibrary(args.filaments)
    img_processor = ImageProcessor(args.image, width_mm, args.colors)
    img_processor.load_and_prepare(nozzle_diameter=args.nozzle)

    alpha_mask = img_processor.alpha_mask
    alpha_pixels = alpha_mask >= 0.5

    # Select filaments
    dominant_colors, _, _ = img_processor.quantize_colors()
    selected = filament_lib.select_best_filaments(
        dominant_colors, args.colors,
        layer_height=args.layer_height,
        min_color_difference=args.min_delta_e,
        model_height=args.model_height,
    )
    selected = filament_lib.optimize_filament_set(
        selected, img_processor.image, alpha_mask,
        args.layer_height, args.model_height, num_layers, mode='standard',
    )

    sorted_filaments = sort_filaments_by_luminosity(selected)

    # Print filament info
    print("\nSelected filaments (sorted dark → light):")
    for i, (_, f) in enumerate(sorted_filaments.iterrows()):
        rgb = f['rgb']
        lab = f['lab']
        td = f['transmission_distance']
        print(f"  {i}: {f['name']:20s}  RGB=({rgb[0]:.2f},{rgb[1]:.2f},{rgb[2]:.2f})  "
              f"L={lab[0]:.1f}  TD={td:.1f}mm")

    # Pipeline stages
    stages = []
    image_rgb = img_processor.image
    H, W = image_rgb.shape[:2]

    # Stage 0: Original image
    stages.append(('Original image', image_rgb))

    # Stage 1: Contrast-enhanced grayscale
    grayscale = (0.2126 * image_rgb[:, :, 0] +
                 0.7152 * image_rgb[:, :, 1] +
                 0.0722 * image_rgb[:, :, 2])
    if grayscale.max() > grayscale.min():
        grayscale = (grayscale - grayscale.min()) / (grayscale.max() - grayscale.min())

    enhanced = apply_contrast_enhancement(grayscale.copy(), alpha_mask, 2.0)
    enhanced = apply_unsharp_mask(enhanced, alpha_mask)
    stages.append(('Contrast-enhanced grayscale', grayscale_to_rgb(enhanced)))

    # Stage 2: Heightmap + z-band allocation
    filament_tds = np.array([f['transmission_distance']
                              for _, f in sorted_filaments.iterrows()])
    layer_counts_arr, layer_boundaries, z_boundaries = allocate_layers_td_proportional(
        filament_tds, num_layers, args.layer_height)
    pixel_height = compute_heightmap(enhanced, alpha_pixels, args.model_height, args.layer_height)

    # Print z-band info
    print(f"\nZ-band allocation (total {num_layers} layers, {args.model_height}mm):")
    for i, (_, f) in enumerate(sorted_filaments.iterrows()):
        z_lo = z_boundaries[i]
        z_hi = z_boundaries[i + 1]
        print(f"  {f['name']:20s}: z=[{z_lo:.3f}, {z_hi:.3f}]mm ({layer_counts_arr[i]} layers)")

    stages.append(('Heightmap (blue=low, red=high)', heightmap_to_heatmap(pixel_height, args.model_height)))

    # Stage 3: Per-band thickness visualization
    for i in range(len(sorted_filaments)):
        z_lo = z_boundaries[i]
        z_hi = z_boundaries[i + 1]
        band_thickness = np.clip(pixel_height, z_lo, z_hi) - z_lo
        max_thick = z_hi - z_lo
        name = sorted_filaments.iloc[i]['name']
        stages.append((f'Band {i} thickness: {name} (max={max_thick:.3f}mm)',
                        heightmap_to_heatmap(band_thickness, max_thick)))

    # Stage 4: Cumulative Beer-Lambert at each band
    filament_rgbs = np.array([f['rgb'] for _, f in sorted_filaments.iterrows()])
    filament_rgbs_lin = srgb_to_linear(filament_rgbs)

    light = np.ones((H, W, 3))
    for i in range(len(sorted_filaments)):
        z_lo = z_boundaries[i]
        z_hi = z_boundaries[i + 1]
        thickness = np.clip(pixel_height, z_lo, z_hi) - z_lo
        td = max(filament_tds[i], 0.1)
        rgb_lin = filament_rgbs_lin[i]
        transmission = np.exp(-thickness / td)[:, :, np.newaxis]
        light = light * transmission + rgb_lin * (1.0 - transmission)

        cumulative_srgb = linear_to_srgb(np.clip(light.copy(), 0, 1))
        name = sorted_filaments.iloc[i]['name']
        stages.append((f'After band {i}: {name}', cumulative_srgb))

    # Stage 5: Raw Beer-Lambert output (no auto-contrast)
    raw_preview = render_standard_preview(pixel_height, z_boundaries, sorted_filaments)
    stages.append(('Raw Beer-Lambert (no contrast adjust)', raw_preview))

    # Stage 6: Auto-contrast params
    params = compute_contrast_params(raw_preview, alpha_pixels)

    if params:
        print(f"\nAuto-contrast params: p_lo={params['p_lo']:.1f}, rng={params['rng']:.1f}")
        print(f"  Maps L=[{params['p_lo']:.1f}, {params['p_lo']+params['rng']:.1f}] → L=[5, 95]")
        chroma_boost = min(90.0 / max(params['rng'], 1.0), 5.0)
        print(f"  Chroma boost: {chroma_boost:.1f}x")
    else:
        print("\nAuto-contrast: no adjustment needed (range > 70)")

    final_preview = apply_contrast_params(raw_preview, alpha_pixels, params)
    final_preview[~alpha_pixels] = 0.1
    stages.append(('After auto-contrast (final)', final_preview))

    # Build HTML (pass dummy z_boundaries/layer_counts for backward compat)
    z_boundaries = np.zeros(len(sorted_filaments) + 1)
    layer_counts_dummy = np.ones(len(sorted_filaments), dtype=int)
    print(f"\nGenerating diagnostic page with {len(stages)} stages...")
    html = build_diagnostic_html(stages, sorted_filaments, z_boundaries,
                                  layer_counts_dummy, params)

    with tempfile.NamedTemporaryFile('w', suffix='.html', delete=False) as f:
        f.write(html)
        tmp_path = f.name

    webbrowser.open('file://' + tmp_path)
    print(f"Opened diagnostic page: {tmp_path}")
    return 0


def build_diagnostic_html(stages, sorted_filaments, z_boundaries, layer_counts, contrast_params):
    """Build scrollable diagnostic page showing all pipeline stages."""
    panels = []
    for i, (label, img) in enumerate(stages):
        b64 = to_b64_png(img)
        panels.append({'label': label, 'png': b64})

    # Filament info for header
    fil_info = []
    for idx, (_, f) in enumerate(sorted_filaments.iterrows()):
        z_lo = z_boundaries[idx]
        z_hi = z_boundaries[idx + 1]
        fil_info.append({
            'name': f['name'],
            'hex': '#%02x%02x%02x' % tuple((np.array(f['rgb']) * 255).astype(int)),
            'td': round(float(f['transmission_distance']), 2),
            'L': round(float(f['lab'][0]), 1),
            'z_lo': round(float(z_lo), 3),
            'z_hi': round(float(z_hi), 3),
            'layers': int(layer_counts[idx]),
        })

    contrast_info = "No adjustment needed"
    if contrast_params:
        p_lo = contrast_params['p_lo']
        rng = contrast_params['rng']
        boost = min(90.0 / max(rng, 1.0), 5.0)
        contrast_info = (f"L=[{p_lo:.1f}, {p_lo+rng:.1f}] → [5, 95], "
                         f"chroma boost: {boost:.1f}x")

    panels_json = json.dumps(panels)
    fil_json = json.dumps(fil_info)

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>HueCLI — Pipeline Diagnostic</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #1a1a1a; color: #ddd; font: 14px/1.5 system-ui; padding: 20px; }}
  h1 {{ color: #fff; margin-bottom: 10px; }}
  .info-bar {{
    background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
    border-radius: 8px; padding: 12px 16px; margin-bottom: 20px;
  }}
  .info-bar h3 {{ color: #fff; margin-bottom: 8px; }}
  .filament-row {{
    display: flex; align-items: center; gap: 8px; margin: 4px 0;
    font: 12px monospace;
  }}
  .fil-swatch {{
    width: 16px; height: 16px; border-radius: 3px;
    border: 1px solid rgba(255,255,255,0.3); flex-shrink: 0;
  }}
  .contrast-info {{ color: #f90; margin-top: 8px; font: 12px monospace; }}
  .stage-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
    gap: 12px;
  }}
  .stage {{
    background: #2a2a2a; border-radius: 8px; overflow: hidden;
    border: 1px solid rgba(255,255,255,0.1);
  }}
  .stage-label {{
    background: rgba(0,0,0,0.5); color: #fff; font: bold 12px system-ui;
    padding: 6px 12px;
  }}
  .stage img {{
    width: 100%; display: block;
    image-rendering: pixelated;
  }}
</style>
</head>
<body>
<h1>Pipeline Diagnostic</h1>

<div class="info-bar">
  <h3>Filaments (sorted dark → light)</h3>
  <div id="filament-list"></div>
  <div class="contrast-info">Auto-contrast: {contrast_info}</div>
</div>

<div class="stage-grid" id="stages"></div>

<script>
const panels = {panels_json};
const filaments = {fil_json};

// Filament list
const filList = document.getElementById('filament-list');
filaments.forEach((f, i) => {{
  const row = document.createElement('div');
  row.className = 'filament-row';
  row.innerHTML = `
    <div class="fil-swatch" style="background:${{f.hex}}"></div>
    <span>${{i}}: ${{f.name}} &nbsp; L=${{f.L}} &nbsp; TD=${{f.td}}mm &nbsp; z=[${{f.z_lo}}, ${{f.z_hi}}]mm &nbsp; (${{f.layers}} layers)</span>
  `;
  filList.appendChild(row);
}});

// Stage grid
const grid = document.getElementById('stages');
panels.forEach(p => {{
  const div = document.createElement('div');
  div.className = 'stage';
  div.innerHTML = `
    <div class="stage-label">${{p.label}}</div>
    <img src="data:image/png;base64,${{p.png}}">
  `;
  grid.appendChild(div);
}});
</script>
</body>
</html>'''


if __name__ == '__main__':
    sys.exit(main())
