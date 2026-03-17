#!/usr/bin/env python3
"""Tournament-style A/B comparison tool for iterating on color fidelity.

Each round isolates ONE variable (filaments, contrast, sharpness, fine-tune)
while keeping everything else identical, so you can judge fairly.

Uses full-resolution Beer-Lambert color simulation rendered as textures.

Usage:
    python3 tournament.py image.png -f filaments.csv -c 4 -n 0.2 -l 0.12 -m 1.44 -s 100x140
"""

import argparse
import base64
import io
import json
import logging
import random
import sys
import tempfile
import time
import webbrowser
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage import color

from filaments import FilamentLibrary
from image import ImageProcessor
from color_science import (
    sort_filaments_by_luminosity,
    apply_contrast_enhancement,
    apply_unsharp_mask,
    allocate_layers_td_proportional,
    compute_heightmap,
    render_standard_preview,
    auto_contrast_preview,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Beer-Lambert color rendering (no mesh — just numpy)
# ---------------------------------------------------------------------------

def render_beer_lambert_image(image_rgb, alpha_mask, sorted_filaments,
                              layer_height, model_height, num_layers,
                              contrast_strength=None, unsharp_strength=None,
                              unsharp_radius=None):
    """Render a full-resolution Beer-Lambert color simulation image.

    Uses heightmap + TD-proportional z-bands for standard topographical rendering.

    Returns (H, W, 3) sRGB float array.
    """
    alpha_pixels = alpha_mask >= 0.5

    # Convert to grayscale
    grayscale = (0.2126 * image_rgb[:, :, 0] +
                 0.7152 * image_rgb[:, :, 1] +
                 0.0722 * image_rgb[:, :, 2])
    if grayscale.max() > grayscale.min():
        grayscale = (grayscale - grayscale.min()) / (grayscale.max() - grayscale.min())

    # Apply contrast enhancement + unsharp mask
    cs = contrast_strength if contrast_strength is not None else 2.0
    enhanced = apply_contrast_enhancement(grayscale.copy(), alpha_mask, cs)
    enhanced = apply_unsharp_mask(enhanced, alpha_mask,
                                  strength=unsharp_strength or 1.5,
                                  radius=unsharp_radius or 1.5)

    # Allocate z-bands and compute heightmap
    filament_tds = np.array([f['transmission_distance'] for _, f in sorted_filaments.iterrows()])
    _, _, z_boundaries = allocate_layers_td_proportional(filament_tds, num_layers, layer_height)
    pixel_height = compute_heightmap(enhanced, alpha_pixels, model_height, layer_height)

    # Render via Beer-Lambert + auto-contrast
    preview = render_standard_preview(pixel_height, z_boundaries, sorted_filaments)
    result = auto_contrast_preview(preview, alpha_pixels)
    result[~alpha_pixels] = 0.1

    return result


def image_to_b64_png(rgb_float):
    """Convert (H,W,3) float RGB to base64-encoded PNG."""
    uint8 = (np.clip(rgb_float, 0, 1) * 255).astype(np.uint8)
    uint8 = np.flipud(uint8)  # Flip back from 3D coords to image coords
    img = Image.fromarray(uint8)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def filament_info_list(sorted_filaments):
    """Build filament info dicts for display."""
    info = []
    for _, f in sorted_filaments.iterrows():
        info.append({
            'name': f['name'],
            'hex': '#%02x%02x%02x' % tuple(
                (np.array(f['rgb']) * 255).astype(int)),
            'td': round(float(f['transmission_distance']), 2),
        })
    return info


# ---------------------------------------------------------------------------
# Round generators
# ---------------------------------------------------------------------------

def _rank_alternatives_per_slot(filament_lib, dominant_colors, base_config):
    """For each slot in the greedy selection, rank all filaments by suitability.

    Returns:
        base_selection: DataFrame of the deterministic best pick
        slot_rankings: list of lists — slot_rankings[slot] is a list of
                       (delta_e, filament_idx) sorted best-first, excluding
                       filaments already used in other slots.
    """
    from color_science import compute_effective_color

    count = base_config['color_count']
    estimated_thickness = base_config['model_height'] / max(count, 1)

    # Sort target colors by luminosity (same order as select_best_filaments)
    target_sorted = sorted(enumerate(dominant_colors),
                           key=lambda x: x[1][0])

    # Score every filament against every target color
    all_scores = {}  # (target_idx, filament_idx) -> delta_e
    for _, target_lab in target_sorted:
        for idx, row in filament_lib.df.iterrows():
            rendered_lab = compute_effective_color(
                row['rgb'], row['transmission_distance'], estimated_thickness)
            delta_e = color.deltaE_ciede2000(
                np.array([[target_lab]]),
                np.array([[rendered_lab]])
            )[0][0]

            # Opacity penalty (same as select_best_filaments)
            td = row['transmission_distance']
            opacity = 1.0 - np.exp(-estimated_thickness / max(td, 0.1))
            if opacity < 0.15:
                delta_e += 30.0 * (1.0 - opacity / 0.15)

            all_scores[(id(target_lab), idx)] = delta_e

    # Run greedy selection to get base pick + per-slot rankings
    selected_indices = []
    slot_rankings = []

    for _, target_lab in target_sorted:
        scored = []
        for idx, row in filament_lib.df.iterrows():
            if idx in selected_indices:
                continue
            scored.append((all_scores[(id(target_lab), idx)], idx))
        scored.sort()

        if scored:
            selected_indices.append(scored[0][1])
            # Keep top alternatives (skip the winner, it's already the base)
            slot_rankings.append(scored[1:])
        else:
            slot_rankings.append([])

    base_selection = filament_lib.df.iloc[selected_indices].reset_index(drop=True)
    return base_selection, slot_rankings, selected_indices


def run_filament_round(filament_lib, img_processor, base_config, n_candidates=4):
    """Round 1: Vary filament selection only. All processing params identical.

    Generates sensible candidates by swapping one filament at a time for its
    next-best alternative, rather than random perturbation across the whole library.
    """
    from skimage import color as skcolor

    dominant_colors, _, _ = img_processor.quantize_colors()

    # Get base selection and per-slot alternative rankings
    base_selection, slot_rankings, base_indices = _rank_alternatives_per_slot(
        filament_lib, dominant_colors, base_config)

    # Candidate 0: deterministic best pick + SA optimization
    candidate_sets = [base_selection]

    # Generate additional candidates by swapping one slot at a time
    # Pick the slot with the closest alternative (smallest delta-E gap)
    swap_options = []
    for slot, rankings in enumerate(slot_rankings):
        for alt_delta_e, alt_idx in rankings:
            # Skip if this filament is already in the base selection
            if alt_idx in base_indices:
                continue
            # Score = how close this alternative is (lower = more plausible swap)
            swap_options.append((alt_delta_e, slot, alt_idx))

    swap_options.sort()

    # Pick diverse swaps: don't swap the same slot twice in a row
    used_swaps = set()
    for delta_e, slot, alt_idx in swap_options:
        if len(candidate_sets) >= n_candidates:
            break
        # Create a unique key to avoid duplicate candidate sets
        swap_key = (slot, alt_idx)
        if swap_key in used_swaps:
            continue
        used_swaps.add(swap_key)

        # Build candidate by swapping one filament
        rows = []
        for r in range(len(base_selection)):
            if r == slot:
                rows.append(filament_lib.df.loc[alt_idx])
            else:
                rows.append(base_selection.iloc[r])
        candidate_df = pd.DataFrame(rows).reset_index(drop=True)
        candidate_sets.append(candidate_df)

    # Now render each candidate (with SA optimization)
    candidates = []
    for i, selected in enumerate(candidate_sets):
        t0 = time.time()

        optimized = filament_lib.optimize_filament_set(
            selected,
            img_processor.image,
            img_processor.alpha_mask,
            base_config['layer_height'],
            base_config['model_height'],
            base_config['num_layers'],
            mode='standard',
        )

        sorted_fils = sort_filaments_by_luminosity(optimized)
        preview = render_beer_lambert_image(
            img_processor.image, img_processor.alpha_mask, sorted_fils,
            base_config['layer_height'], base_config['model_height'],
            base_config['num_layers'])

        elapsed = time.time() - t0
        names = [f['name'] for _, f in sorted_fils.iterrows()]
        label = f"{chr(65+i)} — {', '.join(names)}"

        candidates.append({
            'label': label,
            'preview': preview,
            'sorted_filaments': sorted_fils,
            'elapsed': elapsed,
            'params': {
                'filaments': names,
                'filament_tds': [float(f['transmission_distance'])
                                  for _, f in sorted_fils.iterrows()],
            },
        })
        print(f"    {chr(65+i)}: {', '.join(names)} ({elapsed:.1f}s)")

    return candidates


def run_contrast_round(img_processor, base_config, sorted_filaments, n_candidates=4):
    """Round 2: Vary contrast only. Same filaments, same sharpness."""
    unsharp_strength = 1.5
    unsharp_radius = 1.5

    contrast_values = [1.0, 2.0, 3.0, 4.5]

    candidates = []
    for i, cs in enumerate(contrast_values[:n_candidates]):
        t0 = time.time()
        preview = render_beer_lambert_image(
            img_processor.image, img_processor.alpha_mask, sorted_filaments,
            base_config['layer_height'], base_config['model_height'],
            base_config['num_layers'])

        elapsed = time.time() - t0
        label = f"{chr(65+i)} — Contrast {cs:.1f}"

        candidates.append({
            'label': label,
            'preview': preview,
            'sorted_filaments': sorted_filaments,
            'elapsed': elapsed,
            'params': {
                'contrast_strength': cs,
                'unsharp_strength': unsharp_strength,
                'unsharp_radius': unsharp_radius,
            },
        })
        print(f"    {chr(65+i)}: contrast={cs:.1f} ({elapsed:.1f}s)")

    return candidates


def run_sharpness_round(img_processor, base_config, sorted_filaments,
                         contrast_strength, n_candidates=4):
    """Round 3: Vary sharpness only. Same filaments, same contrast."""
    sharpness_presets = [
        ('Soft', 0.5, 2.5),
        ('Medium', 1.5, 1.5),
        ('Sharp', 2.5, 1.0),
        ('Very sharp', 4.0, 0.7),
    ]

    candidates = []
    for i, (name, us, ur) in enumerate(sharpness_presets[:n_candidates]):
        t0 = time.time()
        preview = render_beer_lambert_image(
            img_processor.image, img_processor.alpha_mask, sorted_filaments,
            base_config['layer_height'], base_config['model_height'],
            base_config['num_layers'])

        elapsed = time.time() - t0
        label = f"{chr(65+i)} — {name} ({us:.1f}/{ur:.1f})"

        candidates.append({
            'label': label,
            'preview': preview,
            'sorted_filaments': sorted_filaments,
            'elapsed': elapsed,
            'params': {
                'contrast_strength': contrast_strength,
                'unsharp_strength': us,
                'unsharp_radius': ur,
            },
        })
        print(f"    {chr(65+i)}: {name} strength={us:.1f} radius={ur:.1f} ({elapsed:.1f}s)")

    return candidates


def run_finetune_round(img_processor, base_config, sorted_filaments,
                        contrast_strength, unsharp_strength, unsharp_radius,
                        n_candidates=4):
    """Round 4: Small variations around the winning params from rounds 2-3."""
    variations = [
        ('Winner', 1.0, 1.0, 1.0),
        ('Slightly more contrast + sharper', 1.15, 1.2, 0.9),
        ('Slightly less contrast + softer', 0.85, 0.8, 1.15),
        ('More contrast, less sharp', 1.2, 0.8, 1.2),
    ]

    candidates = []
    for i, (name, cs_mult, us_mult, ur_mult) in enumerate(variations[:n_candidates]):
        cs = contrast_strength * cs_mult
        us = unsharp_strength * us_mult
        ur = unsharp_radius * ur_mult

        t0 = time.time()
        preview = render_beer_lambert_image(
            img_processor.image, img_processor.alpha_mask, sorted_filaments,
            base_config['layer_height'], base_config['model_height'],
            base_config['num_layers'])

        elapsed = time.time() - t0
        label = f"{chr(65+i)} — {name}"

        candidates.append({
            'label': label,
            'preview': preview,
            'sorted_filaments': sorted_filaments,
            'elapsed': elapsed,
            'params': {
                'contrast_strength': cs,
                'unsharp_strength': us,
                'unsharp_radius': ur,
            },
        })
        print(f"    {chr(65+i)}: {name} c={cs:.2f} s={us:.2f} r={ur:.2f} ({elapsed:.1f}s)")

    return candidates


# ---------------------------------------------------------------------------
# Multi-panel HTML builder
# ---------------------------------------------------------------------------

def build_tournament_html(candidates, round_title):
    """Build a 2x2 grid HTML page with textured Three.js panels."""
    panel_data = []
    for i, c in enumerate(candidates):
        b64_png = image_to_b64_png(c['preview'])
        fil_info = filament_info_list(c['sorted_filaments'])
        panel_data.append(
            f'  {{ id: "panel{i}", label: {json.dumps(c["label"])}, '
            f'png: "{b64_png}", filaments: {json.dumps(fil_info)} }}'
        )
    panels_array = ',\n'.join(panel_data)

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HueCLI — {round_title}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #1a1a1a; overflow: hidden; }}
  #round-title {{
    position: fixed; top: 0; left: 0; right: 0; z-index: 10;
    background: rgba(0,0,0,0.8); color: #fff; font: bold 16px system-ui;
    text-align: center; padding: 8px 0;
    border-bottom: 1px solid rgba(255,255,255,0.15);
  }}
  #grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    grid-template-rows: 1fr 1fr;
    width: 100vw; height: 100vh;
    padding-top: 36px;
    gap: 2px;
  }}
  .panel {{
    position: relative;
    background: #2a2a2a;
    overflow: hidden;
  }}
  .panel canvas {{ width: 100% !important; height: 100% !important; }}
  .panel-label {{
    position: absolute; top: 8px; left: 50%; transform: translateX(-50%);
    background: rgba(0,0,0,0.7); color: #fff; font: bold 14px system-ui;
    padding: 4px 16px; border-radius: 6px; z-index: 5;
    pointer-events: none; white-space: nowrap;
  }}
  .panel-filaments {{
    position: absolute; bottom: 8px; left: 8px;
    background: rgba(0,0,0,0.6); color: #ccc; font: 11px system-ui;
    padding: 6px 10px; border-radius: 6px; z-index: 5;
    pointer-events: none; max-width: 90%;
  }}
  .panel-filaments .swatch {{
    display: inline-block; width: 12px; height: 12px;
    border-radius: 2px; margin-right: 4px; vertical-align: middle;
    border: 1px solid rgba(255,255,255,0.3);
  }}
</style>
</head>
<body>
<div id="round-title">{round_title}</div>
<div id="grid">
  <div class="panel" id="panel0"></div>
  <div class="panel" id="panel1"></div>
  <div class="panel" id="panel2"></div>
  <div class="panel" id="panel3"></div>
</div>

<script type="importmap">
{{
  "imports": {{
    "three": "https://unpkg.com/three@0.170.0/build/three.module.js",
    "three/addons/": "https://unpkg.com/three@0.170.0/examples/jsm/"
  }}
}}
</script>

<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

const panels = [
{panels_array}
];

panels.forEach(p => {{
  const container = document.getElementById(p.id);

  const labelDiv = document.createElement('div');
  labelDiv.className = 'panel-label';
  labelDiv.textContent = p.label;
  container.appendChild(labelDiv);

  const filDiv = document.createElement('div');
  filDiv.className = 'panel-filaments';
  filDiv.innerHTML = p.filaments.map(f =>
    `<span class="swatch" style="background:${{f.hex}}"></span>${{f.name}}`
  ).join(' &middot; ');
  container.appendChild(filDiv);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x1a1a1a);

  const rect = container.getBoundingClientRect();
  const renderer = new THREE.WebGLRenderer({{ antialias: true }});
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.toneMapping = THREE.NoToneMapping;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.setSize(rect.width, rect.height);
  container.appendChild(renderer.domElement);

  const img = new window.Image();
  img.onload = () => {{
    const texture = new THREE.Texture(img);
    texture.needsUpdate = true;
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.minFilter = THREE.LinearFilter;
    texture.magFilter = THREE.LinearFilter;

    const aspect = img.width / img.height;
    const planeH = 10;
    const planeW = planeH * aspect;
    const geometry = new THREE.PlaneGeometry(planeW, planeH);
    const material = new THREE.MeshBasicMaterial({{ map: texture }});
    const mesh = new THREE.Mesh(geometry, material);
    scene.add(mesh);

    const camera = new THREE.PerspectiveCamera(45, rect.width / rect.height, 0.1, 100);
    const dist = planeH * 1.1 / (2 * Math.tan(Math.PI / 8));
    camera.position.set(0, 0, dist);
    camera.lookAt(0, 0, 0);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.12;

    function animate() {{
      requestAnimationFrame(animate);
      controls.update();
      const r = container.getBoundingClientRect();
      if (renderer.domElement.width !== Math.round(r.width) ||
          renderer.domElement.height !== Math.round(r.height)) {{
        renderer.setSize(r.width, r.height);
        camera.aspect = r.width / r.height;
        camera.updateProjectionMatrix();
      }}
      renderer.render(scene, camera);
    }}
    animate();
  }};
  img.src = 'data:image/png;base64,' + p.png;
}});
</script>
</body>
</html>'''


# ---------------------------------------------------------------------------
# CLI + Main
# ---------------------------------------------------------------------------

def parse_tournament_args():
    parser = argparse.ArgumentParser(description='HueCLI Tournament — iterate on color fidelity')
    parser.add_argument('image', help='Input image path')
    parser.add_argument('-f', '--filaments', default='filaments.csv', help='Filament CSV')
    parser.add_argument('-c', '--colors', type=int, default=4, help='Number of colors')
    parser.add_argument('-n', '--nozzle', type=float, default=0.2, help='Nozzle diameter (mm)')
    parser.add_argument('-l', '--layer-height', type=float, default=0.12, help='Layer height (mm)')
    parser.add_argument('-m', '--model-height', type=float, default=1.44, help='Model height (mm)')
    parser.add_argument('-s', '--size', default='100x140', help='Print size WxH (mm)')
    parser.add_argument('-d', '--min-delta-e', type=float, default=5.0, help='Min color difference')
    parser.add_argument('--candidates', type=int, default=4, help='Candidates per round')
    parser.add_argument('--log', default='tournament_log.json', help='Output log file')
    return parser.parse_args()


def show_and_choose(candidates, round_title, n_candidates):
    """Show candidates in browser and get user choice."""
    html = build_tournament_html(candidates[:n_candidates], round_title)
    with tempfile.NamedTemporaryFile('w', suffix='.html', delete=False) as f:
        f.write(html)
        tmp_path = f.name
    webbrowser.open('file://' + tmp_path)

    print(f"\n  Which is best? (1-{n_candidates}): ", end='', flush=True)
    while True:
        try:
            choice = int(input().strip())
            if 1 <= choice <= n_candidates:
                return choice
            print(f"  Enter 1-{n_candidates}: ", end='', flush=True)
        except (ValueError, EOFError):
            print(f"  Enter 1-{n_candidates}: ", end='', flush=True)


def main():
    args = parse_tournament_args()

    if not Path(args.image).exists():
        print(f"Image not found: {args.image}")
        return 1

    width_mm, height_mm = [float(x) for x in args.size.split('x')]
    num_layers = int(args.model_height / args.layer_height)

    base_config = {
        'layer_height': args.layer_height,
        'model_height': args.model_height,
        'width_mm': width_mm,
        'height_mm': height_mm,
        'color_count': args.colors,
        'num_layers': num_layers,
        'min_color_difference': args.min_delta_e,
        'nozzle_diameter': args.nozzle,
    }

    n = args.candidates

    print(f"\nHueCLI Tournament")
    print(f"Image: {args.image}")
    print(f"Config: {args.colors} colors, {args.layer_height}mm layers, "
          f"{args.model_height}mm height, {args.size}mm ({num_layers} layers)")
    print(f"4 rounds: Filaments -> Contrast -> Sharpness -> Fine-tune")
    print()

    # Load image and filaments
    filament_lib = FilamentLibrary(args.filaments)
    img_processor = ImageProcessor(args.image, width_mm, args.colors)
    img_processor.load_and_prepare(nozzle_diameter=args.nozzle)

    img_processor.grayscale = (
        0.2126 * img_processor.image[:, :, 0] +
        0.7152 * img_processor.image[:, :, 1] +
        0.0722 * img_processor.image[:, :, 2])
    if img_processor.grayscale.max() > img_processor.grayscale.min():
        img_processor.grayscale = ((img_processor.grayscale - img_processor.grayscale.min()) /
                                    (img_processor.grayscale.max() - img_processor.grayscale.min()))

    log = {
        'timestamp': datetime.now().isoformat(),
        'image': args.image,
        'base_config': base_config,
        'rounds': [],
    }

    # ---- Round 1: Filament selection ----
    print(f"{'='*60}")
    print(f"ROUND 1/4 — FILAMENT SELECTION")
    print(f"  Same contrast & sharpness. Pick the best color match.")
    print(f"{'='*60}")

    fil_candidates = run_filament_round(filament_lib, img_processor, base_config, n)
    choice = show_and_choose(fil_candidates,
                              "Round 1/4 — Filament Selection (pick best colors)", n)
    best_filaments = fil_candidates[choice - 1]['sorted_filaments']
    log['rounds'].append({
        'round': 1, 'type': 'filaments', 'winner': choice,
        'winner_params': fil_candidates[choice - 1]['params'],
    })
    fil_names = [f['name'] for _, f in best_filaments.iterrows()]
    print(f"\n  Winner: {chr(64+choice)} — {', '.join(fil_names)}")

    # ---- Round 2: Contrast ----
    print(f"\n{'='*60}")
    print(f"ROUND 2/4 — CONTRAST")
    print(f"  Same filaments & sharpness. Pick the best contrast level.")
    print(f"{'='*60}")

    con_candidates = run_contrast_round(img_processor, base_config, best_filaments, n)
    choice = show_and_choose(con_candidates,
                              "Round 2/4 — Contrast (pick best tonal range)", n)
    best_contrast = con_candidates[choice - 1]['params']['contrast_strength']
    log['rounds'].append({
        'round': 2, 'type': 'contrast', 'winner': choice,
        'winner_params': con_candidates[choice - 1]['params'],
    })
    print(f"\n  Winner: {chr(64+choice)} — contrast={best_contrast:.1f}")

    # ---- Round 3: Sharpness ----
    print(f"\n{'='*60}")
    print(f"ROUND 3/4 — SHARPNESS")
    print(f"  Same filaments & contrast. Pick the crispest edges/text.")
    print(f"{'='*60}")

    sharp_candidates = run_sharpness_round(
        img_processor, base_config, best_filaments, best_contrast, n)
    choice = show_and_choose(sharp_candidates,
                              "Round 3/4 — Sharpness (pick crispest detail)", n)
    best_unsharp_strength = sharp_candidates[choice - 1]['params']['unsharp_strength']
    best_unsharp_radius = sharp_candidates[choice - 1]['params']['unsharp_radius']
    log['rounds'].append({
        'round': 3, 'type': 'sharpness', 'winner': choice,
        'winner_params': sharp_candidates[choice - 1]['params'],
    })
    print(f"\n  Winner: {chr(64+choice)} — unsharp={best_unsharp_strength:.1f}/{best_unsharp_radius:.1f}")

    # ---- Round 4: Fine-tune ----
    print(f"\n{'='*60}")
    print(f"ROUND 4/4 — FINE-TUNE")
    print(f"  Small variations around your picks. Final polish.")
    print(f"{'='*60}")

    ft_candidates = run_finetune_round(
        img_processor, base_config, best_filaments,
        best_contrast, best_unsharp_strength, best_unsharp_radius, n)
    choice = show_and_choose(ft_candidates,
                              "Round 4/4 — Fine-tune (pick overall best)", n)
    final_params = ft_candidates[choice - 1]['params']
    log['rounds'].append({
        'round': 4, 'type': 'finetune', 'winner': choice,
        'winner_params': final_params,
    })
    print(f"\n  Winner: {chr(64+choice)}")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"TOURNAMENT COMPLETE")
    print(f"{'='*60}")
    print(f"\nWinning filaments: {', '.join(fil_names)}")

    # Build CLI command
    filaments_arg = ','.join(fil_names)
    cli_cmd = (f"python3 huecli.py {args.image}"
               f" -f {args.filaments}"
               f" --use-filaments \"{filaments_arg}\""
               f" -c {base_config['color_count']}"
               f" -n {args.nozzle}"
               f" -l {args.layer_height}"
               f" -m {args.model_height}"
               f" -s {args.size}")

    print(f"\nCLI command:")
    print(f"  {cli_cmd}")

    log_path = Path(args.log)
    log['cli_command'] = cli_cmd
    log_path.write_text(json.dumps(log, indent=2))
    print(f"\nLog saved to {log_path}")

    # Run 3D preview with winning filaments
    print(f"\nOpening 3D preview with winning filaments...")
    from config import PipelineConfig, ProcessedImage
    from preview import show_3d_preview

    pipeline_config = PipelineConfig(
        layer_height=base_config['layer_height'],
        model_height=base_config['model_height'],
        width_mm=base_config['width_mm'],
        mode='standard',
        sandwich_layers=1,
        base_layers=2,
    )
    from skimage import color as skcolor
    processed_image = ProcessedImage(
        image_rgb=img_processor.image,
        image_lab=skcolor.rgb2lab(img_processor.image),
        grayscale=img_processor.grayscale,
        alpha_mask=img_processor.alpha_mask,
    )

    show_3d_preview(pipeline_config, processed_image, best_filaments)

    print(f"\nTo generate STLs, run:")
    print(f"  {cli_cmd}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
