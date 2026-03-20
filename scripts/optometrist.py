#!/usr/bin/env python3
"""Optometrist-style A/B comparison tool for refining standard mode pipeline.

Shows two side-by-side previews with ONE variable changed. The developer
picks A or B, and the winner becomes the baseline for the next round.

Usage:
    python3 scripts/optometrist.py examples/charizard.png.webp [options]

This is a development tool, not user-facing.
"""

import argparse
import base64
import io
import json
import logging
import sys
import tempfile
import time
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage import color as skcolor

from huecli.filaments import FilamentLibrary
from huecli.image import ImageProcessor
from huecli.color_science import (
    sort_filaments_by_luminosity,
    apply_contrast_enhancement,
    apply_unsharp_mask,
    median_smooth,
    allocate_layers_standard,
    compute_heightmap,
    render_standard_preview,
    auto_contrast_preview,
    apply_edge_inset,
    score_standard_filament_set,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def render_standard_image(image_rgb, alpha_mask, sorted_filaments,
                          layer_height, model_height, num_layers,
                          contrast_strength=2.0,
                          unsharp_strength=1.5, unsharp_radius=1.5,
                          median_radius=2, spike_tolerance=0.15,
                          edge_inset_rate=0.25, nozzle_diameter=0.2,
                          min_region_mm2=0.8,
                          contrast_mode='auto', bright_gamma=1.5,
                          bright_scurve=4.0):
    """Render a standard mode preview with fully parameterized pipeline.

    Every tunable is an explicit parameter so the optometrist can sweep them.

    contrast_mode: 'auto' (default adaptive), 'scurve' (always S-curve),
                   'gamma' (always gamma), 'none' (passthrough)
    """
    alpha_pixels = alpha_mask >= 0.5
    pixel_size = nozzle_diameter  # 1 pixel = 1 nozzle width

    # Grayscale
    grayscale = (0.2126 * image_rgb[:, :, 0] +
                 0.7152 * image_rgb[:, :, 1] +
                 0.0722 * image_rgb[:, :, 2])
    if grayscale.max() > grayscale.min():
        grayscale = (grayscale - grayscale.min()) / (grayscale.max() - grayscale.min())

    # Contrast enhancement with selectable algorithm
    if contrast_mode == 'none' or contrast_strength == 1.0:
        enhanced = grayscale.copy()
    elif contrast_mode == 'scurve':
        center = 0.5
        enhanced = 1.0 / (1.0 + np.exp(-contrast_strength * (grayscale - center)))
        enhanced = (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min())
    elif contrast_mode == 'gamma':
        enhanced = np.power(grayscale, bright_gamma)
    else:
        # 'auto' — use the existing adaptive function
        enhanced = apply_contrast_enhancement(grayscale.copy(), alpha_mask, contrast_strength)

    # Unsharp mask with custom params
    if unsharp_strength > 0:
        from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter
        blurred = gaussian_filter(enhanced, sigma=unsharp_radius)
        sharpened = enhanced + unsharp_strength * (enhanced - blurred)
        sharpened = np.clip(sharpened, 0, 1)

        # Spike suppression with custom tolerance
        orig_local_max = maximum_filter(enhanced, size=3)
        orig_local_min = minimum_filter(enhanced, size=3)
        upper_bound = orig_local_max + spike_tolerance
        lower_bound = orig_local_min - spike_tolerance
        sharpened = np.clip(sharpened, lower_bound, upper_bound)
        sharpened = np.clip(sharpened, 0, 1)
        sharpened = np.where(alpha_pixels, sharpened, enhanced)
        enhanced = sharpened

    # Median smooth with custom radius
    if median_radius > 0:
        enhanced = median_smooth(enhanced, alpha_mask, radius=median_radius)

    # Allocate z-bands
    filament_tds = np.array([f['transmission_distance']
                              for _, f in sorted_filaments.iterrows()])
    _, _, z_boundaries = allocate_layers_standard(
        filament_tds, num_layers, layer_height)

    # Heightmap
    pixel_height = compute_heightmap(
        enhanced, alpha_pixels, model_height, layer_height,
        min_height=float(z_boundaries[1]))

    # Edge inset with custom rate
    if nozzle_diameter is not None and edge_inset_rate > 0:
        pixel_height = apply_edge_inset(
            pixel_height, z_boundaries, nozzle_diameter, pixel_size)

    # Render preview
    preview = render_standard_preview(pixel_height, z_boundaries, sorted_filaments)
    result = auto_contrast_preview(preview, alpha_pixels)
    result[~alpha_pixels] = 0.1

    return result


def image_to_b64_png(rgb_float):
    """Convert (H,W,3) float RGB to base64 PNG."""
    uint8 = (np.clip(rgb_float, 0, 1) * 255).astype(np.uint8)
    uint8 = np.flipud(uint8)
    img = Image.fromarray(uint8)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def build_ab_html(img_a_b64, img_b_b64, label_a, label_b, question, round_info):
    """Build side-by-side A/B comparison HTML."""
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>HueCLI Optometrist — {round_info}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #1a1a1a; color: #fff; font-family: system-ui; overflow: hidden; }}
  #header {{
    position: fixed; top: 0; left: 0; right: 0; z-index: 10;
    background: rgba(0,0,0,0.85); padding: 12px 20px;
    border-bottom: 1px solid rgba(255,255,255,0.15);
    text-align: center;
  }}
  #header h2 {{ font-size: 16px; margin-bottom: 6px; color: #4fc3f7; }}
  #header p {{ font-size: 14px; color: #ccc; }}
  #grid {{
    display: grid; grid-template-columns: 1fr 1fr;
    width: 100vw; height: 100vh; padding-top: 70px; gap: 2px;
  }}
  .panel {{
    position: relative; background: #2a2a2a; overflow: hidden;
    display: flex; align-items: center; justify-content: center;
  }}
  .panel img {{
    max-width: 100%; max-height: 100%; object-fit: contain;
    image-rendering: pixelated;
  }}
  .panel-label {{
    position: absolute; top: 8px; left: 50%; transform: translateX(-50%);
    background: rgba(0,0,0,0.7); color: #fff; font: bold 18px system-ui;
    padding: 6px 24px; border-radius: 8px; z-index: 5;
  }}
</style>
</head>
<body>
<div id="header">
  <h2>{round_info}</h2>
  <p>{question}</p>
</div>
<div id="grid">
  <div class="panel">
    <div class="panel-label">{label_a}</div>
    <img src="data:image/png;base64,{img_a_b64}">
  </div>
  <div class="panel">
    <div class="panel-label">{label_b}</div>
    <img src="data:image/png;base64,{img_b_b64}">
  </div>
</div>
</body>
</html>'''


def show_ab(img_a, img_b, label_a, label_b, question, round_info):
    """Show A/B comparison and get user choice."""
    b64_a = image_to_b64_png(img_a)
    b64_b = image_to_b64_png(img_b)
    html = build_ab_html(b64_a, b64_b, label_a, label_b, question, round_info)
    with tempfile.NamedTemporaryFile('w', suffix='.html', delete=False) as f:
        f.write(html)
        tmp_path = f.name
    webbrowser.open('file://' + tmp_path)


def parse_args():
    parser = argparse.ArgumentParser(description='HueCLI Optometrist')
    parser.add_argument('image', help='Input image path')
    parser.add_argument('-f', '--filaments', default='data/filaments.csv')
    parser.add_argument('-c', '--colors', type=int, default=6)
    parser.add_argument('-n', '--nozzle', type=float, default=0.2)
    parser.add_argument('-l', '--layer-height', type=float, default=0.08)
    parser.add_argument('-m', '--model-height', type=float, default=3.0)
    parser.add_argument('-s', '--size', default='171x240')
    parser.add_argument('-d', '--min-delta-e', type=float, default=12.0)
    # Override specific round
    parser.add_argument('--round', type=str, default=None,
                        help='Run specific round (e.g., "median", "unsharp_strength")')
    parser.add_argument('--a', type=str, default=None, help='Value for A')
    parser.add_argument('--b', type=str, default=None, help='Value for B')
    return parser.parse_args()


def main():
    args = parse_args()

    if not Path(args.image).exists():
        print(f"Image not found: {args.image}")
        return 1

    width_mm = float(args.size.split('x')[0])
    num_layers = int(args.model_height / args.layer_height)

    # Load
    filament_lib = FilamentLibrary(args.filaments)
    img_processor = ImageProcessor(args.image, width_mm, args.colors)
    img_processor.load_and_prepare(nozzle_diameter=args.nozzle)

    # Select filaments
    dominant_colors, _, _ = img_processor.quantize_colors()
    selected = filament_lib.select_best_filaments(
        dominant_colors, args.colors,
        layer_height=args.layer_height,
        min_color_difference=args.min_delta_e,
        model_height=args.model_height,
        mode='standard')

    # Compute grayscale for SA
    grayscale = (0.2126 * img_processor.image[:, :, 0] +
                 0.7152 * img_processor.image[:, :, 1] +
                 0.0722 * img_processor.image[:, :, 2])
    if grayscale.max() > grayscale.min():
        grayscale = (grayscale - grayscale.min()) / (grayscale.max() - grayscale.min())

    # SA optimization
    optimized = filament_lib.optimize_filament_set(
        selected, img_processor.image, img_processor.alpha_mask,
        args.layer_height, args.model_height, num_layers,
        mode='standard', grayscale=grayscale)
    sorted_filaments = sort_filaments_by_luminosity(optimized)

    fil_names = [f['name'] for _, f in sorted_filaments.iterrows()]
    print(f"\nFilaments: {', '.join(fil_names)}")

    base_kwargs = dict(
        image_rgb=img_processor.image,
        alpha_mask=img_processor.alpha_mask,
        sorted_filaments=sorted_filaments,
        layer_height=args.layer_height,
        model_height=args.model_height,
        num_layers=num_layers,
        nozzle_diameter=args.nozzle,
    )

    if args.round and args.a is not None and args.b is not None:
        # Single round mode — run one comparison
        try:
            va = float(args.a)
            vb = float(args.b)
        except ValueError:
            va = args.a
            vb = args.b
        run_single_round(args.round, va, vb,
                         base_kwargs, args,
                         filament_lib=filament_lib,
                         img_processor=img_processor)
    else:
        print("Use --round <name> --a <val> --b <val> to run a comparison")
        print("Available rounds: median, unsharp_strength, unsharp_radius,")
        print("  contrast, spike_tolerance, edge_inset_rate,")
        print("  colors, model_height, min_delta_e")
    return 0


def select_and_render(filament_lib, img_processor, color_count, layer_height,
                      model_height, min_delta_e, nozzle, **render_overrides):
    """Full pipeline: select filaments + render preview."""
    width_mm = img_processor.width_mm
    num_layers = int(model_height / layer_height)

    img_processor.color_count = color_count
    dominant_colors, _, _ = img_processor.quantize_colors()
    selected = filament_lib.select_best_filaments(
        dominant_colors, color_count,
        layer_height=layer_height,
        min_color_difference=min_delta_e,
        model_height=model_height,
        mode='standard')

    grayscale = (0.2126 * img_processor.image[:, :, 0] +
                 0.7152 * img_processor.image[:, :, 1] +
                 0.0722 * img_processor.image[:, :, 2])
    if grayscale.max() > grayscale.min():
        grayscale = (grayscale - grayscale.min()) / (grayscale.max() - grayscale.min())

    optimized = filament_lib.optimize_filament_set(
        selected, img_processor.image, img_processor.alpha_mask,
        layer_height, model_height, num_layers,
        mode='standard', grayscale=grayscale)
    sorted_filaments = sort_filaments_by_luminosity(optimized)

    fil_names = [f['name'] for _, f in sorted_filaments.iterrows()]

    img = render_standard_image(
        image_rgb=img_processor.image,
        alpha_mask=img_processor.alpha_mask,
        sorted_filaments=sorted_filaments,
        layer_height=layer_height,
        model_height=model_height,
        num_layers=num_layers,
        nozzle_diameter=nozzle,
        **render_overrides)

    return img, fil_names


def run_single_round(round_name, val_a, val_b, base_kwargs, args,
                     filament_lib=None, img_processor=None):
    """Run a single A/B comparison round."""
    # Rounds that only change render params (same filaments)
    param_map = {
        'median': ('median_radius', int),
        'unsharp_strength': ('unsharp_strength', float),
        'unsharp_radius': ('unsharp_radius', float),
        'contrast': ('contrast_strength', float),
        'spike_tolerance': ('spike_tolerance', float),
        'edge_inset_rate': ('edge_inset_rate', float),
        'bright_gamma': ('bright_gamma', float),
        'bright_scurve': ('bright_scurve', float),
    }

    # String-value rounds (contrast algorithm selection)
    string_rounds = {'contrast_mode'}
    if round_name in string_rounds:
        va_str = str(val_a) if not isinstance(val_a, str) else val_a
        vb_str = str(val_b) if not isinstance(val_b, str) else val_b
        # val_a/val_b were passed as floats, map back to strings
        mode_map = {0: 'none', 1: 'auto', 2: 'scurve', 3: 'gamma'}
        va_str = mode_map.get(int(val_a), str(val_a))
        vb_str = mode_map.get(int(val_b), str(val_b))

        print(f"\nRendering A (contrast_mode={va_str})...")
        t0 = time.time()
        img_a = render_standard_image(**base_kwargs, contrast_mode=va_str)
        print(f"  Done in {time.time()-t0:.1f}s")

        print(f"Rendering B (contrast_mode={vb_str})...")
        t0 = time.time()
        img_b = render_standard_image(**base_kwargs, contrast_mode=vb_str)
        print(f"  Done in {time.time()-t0:.1f}s")

        label_a = f"A: {va_str}"
        label_b = f"B: {vb_str}"
        question = "Which has better tonal range and detail?"
        round_info = f"Contrast algorithm: {va_str} vs {vb_str}"
        show_ab(img_a, img_b, label_a, label_b, question, round_info)
        print(f"\nOpened in browser. Which is better, A or B?")
        return

    # Rounds that require full re-selection
    full_pipeline_rounds = {'colors', 'model_height', 'min_delta_e'}

    if round_name in full_pipeline_rounds:
        if filament_lib is None or img_processor is None:
            print(f"Round '{round_name}' needs filament_lib and img_processor")
            return

        va = float(val_a)
        vb = float(val_b)

        overrides_a = {}
        overrides_b = {}
        sel_a = dict(color_count=args.colors, layer_height=args.layer_height,
                     model_height=args.model_height, min_delta_e=args.min_delta_e,
                     nozzle=args.nozzle)
        sel_b = dict(sel_a)

        if round_name == 'colors':
            sel_a['color_count'] = int(va)
            sel_b['color_count'] = int(vb)
        elif round_name == 'model_height':
            sel_a['model_height'] = va
            sel_b['model_height'] = vb
        elif round_name == 'min_delta_e':
            sel_a['min_delta_e'] = va
            sel_b['min_delta_e'] = vb

        print(f"\nRendering A ({round_name}={va})...")
        t0 = time.time()
        img_a, fils_a = select_and_render(filament_lib, img_processor, **sel_a)
        print(f"  Filaments: {', '.join(fils_a)}")
        print(f"  Done in {time.time()-t0:.1f}s")

        print(f"Rendering B ({round_name}={vb})...")
        t0 = time.time()
        img_b, fils_b = select_and_render(filament_lib, img_processor, **sel_b)
        print(f"  Filaments: {', '.join(fils_b)}")
        print(f"  Done in {time.time()-t0:.1f}s")

        label_a = f"A: {round_name}={int(va) if round_name == 'colors' else va}"
        label_b = f"B: {round_name}={int(vb) if round_name == 'colors' else vb}"
        question = f"Which looks better? Focus on color accuracy and overall fidelity."
        round_info = f"Round: {round_name} — {va} vs {vb}"

        show_ab(img_a, img_b, label_a, label_b, question, round_info)
        print(f"\nOpened in browser. Which is better, A or B?")
        return

    if round_name not in param_map:
        print(f"Unknown round: {round_name}")
        print(f"Available: {', '.join(list(param_map.keys()) + list(full_pipeline_rounds))}")
        return

    param_name, param_type = param_map[round_name]
    va = param_type(val_a)
    vb = param_type(val_b)

    print(f"\nRendering A ({param_name}={va})...")
    t0 = time.time()
    img_a = render_standard_image(**base_kwargs, **{param_name: va})
    print(f"  Done in {time.time()-t0:.1f}s")

    print(f"Rendering B ({param_name}={vb})...")
    t0 = time.time()
    img_b = render_standard_image(**base_kwargs, **{param_name: vb})
    print(f"  Done in {time.time()-t0:.1f}s")

    label_a = f"A: {param_name}={va}"
    label_b = f"B: {param_name}={vb}"
    question = f"Which looks better? Focus on text clarity, color accuracy, and detail."
    round_info = f"Round: {round_name} — {va} vs {vb}"

    show_ab(img_a, img_b, label_a, label_b, question, round_info)
    print(f"\nOpened in browser. Which is better, A or B?")


if __name__ == '__main__':
    sys.exit(main())
