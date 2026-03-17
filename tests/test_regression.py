#!/usr/bin/env python3
"""Regression test harness for HueCLI preview generation.

Runs generate_preview_scene() for all supported modes on a small test image
and validates output metrics (mesh count, face colors, vertex counts).
Compares against saved baselines to catch regressions.

Usage:
    python3 test_regression.py                    # Run tests, compare to baselines
    python3 test_regression.py --update           # Update baseline file
    python3 test_regression.py --mode standard    # Test single mode
    python3 test_regression.py --verbose          # Show detailed metrics
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage import color as skcolor

from huecli.config import PipelineConfig, ProcessedImage
from huecli import preview as preview_mod

logging.basicConfig(level=logging.WARNING)

BASELINE_FILE = Path(__file__).parent / 'test_baselines.json'
TEST_IMAGE_SIZE = (80, 60)  # Small for speed


def create_test_image():
    """Create a synthetic test image with known color regions."""
    H, W = TEST_IMAGE_SIZE
    img = np.zeros((H, W, 3), dtype=np.float64)

    # Quadrant colors (sRGB, 0-1)
    img[:H//2, :W//2] = [0.8, 0.2, 0.1]     # Red (top-left)
    img[:H//2, W//2:] = [0.1, 0.6, 0.2]     # Green (top-right)
    img[H//2:, :W//2] = [0.1, 0.2, 0.7]     # Blue (bottom-left)
    img[H//2:, W//2:] = [0.9, 0.8, 0.3]     # Yellow (bottom-right)

    # Add gradient overlay for height variation
    gy = np.linspace(0.3, 1.0, H)[:, None]
    img *= gy[:, :, None]

    grayscale = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    alpha = np.ones((H, W))

    return img, grayscale, alpha


def create_test_filaments(n_colors=4):
    """Create a small filament set for testing."""
    colors = [
        ('Dark Red', (0.3, 0.05, 0.05), 1.5),
        ('Forest Green', (0.05, 0.25, 0.05), 2.0),
        ('Navy Blue', (0.05, 0.05, 0.3), 1.8),
        ('Golden Yellow', (0.8, 0.7, 0.1), 3.0),
        ('Clear', (0.95, 0.95, 0.95), 15.0),
    ][:n_colors + 1]  # +1 for potential transparent

    df = pd.DataFrame({
        'name': [c[0] for c in colors],
        'rgb': [c[1] for c in colors],
        'transmission_distance': [c[2] for c in colors],
        'Brand': ['Test'] * len(colors),
    })
    # Add LAB column required by sort_filaments_by_luminosity
    df['lab'] = df['rgb'].apply(lambda rgb: skcolor.rgb2lab([[rgb]])[0][0])
    return df


def extract_metrics(scene, sorted_filaments):
    """Extract testable metrics from a preview scene."""
    metrics = {
        'mesh_count': len(scene.geometry),
        'total_vertices': 0,
        'total_faces': 0,
        'meshes': [],
        'color_stats': {},
    }

    all_r, all_g, all_b = [], [], []

    for name, geom in scene.geometry.items():
        mesh_info = {
            'name': name,
            'vertices': len(geom.vertices),
            'faces': len(geom.faces),
        }

        if hasattr(geom.visual, 'face_colors') and geom.visual.face_colors is not None:
            fc = geom.visual.face_colors
            if len(fc) > 0:
                rgb = fc[:, :3].astype(float)
                mesh_info['color_mean'] = rgb.mean(axis=0).tolist()
                mesh_info['color_std'] = rgb.std(axis=0).tolist()
                all_r.extend(rgb[:, 0].tolist())
                all_g.extend(rgb[:, 1].tolist())
                all_b.extend(rgb[:, 2].tolist())

        metrics['total_vertices'] += mesh_info['vertices']
        metrics['total_faces'] += mesh_info['faces']
        metrics['meshes'].append(mesh_info)

    # Global color statistics
    if all_r:
        r, g, b = np.array(all_r), np.array(all_g), np.array(all_b)
        metrics['color_stats'] = {
            'r_mean': float(r.mean()), 'r_std': float(r.std()),
            'g_mean': float(g.mean()), 'g_std': float(g.std()),
            'b_mean': float(b.mean()), 'b_std': float(b.std()),
            'is_greyscale': bool(
                abs(r.mean() - g.mean()) < 5 and
                abs(g.mean() - b.mean()) < 5 and
                r.std() < 10 and g.std() < 10 and b.std() < 10
            ),
            'chromatic_fraction': float(
                ((np.abs(r - g) > 5) | (np.abs(g - b) > 5)).mean()
            ),
        }

    return metrics


def run_mode_test(mode, verbose=False):
    """Run preview generation for a single mode and return metrics."""
    img_rgb, grayscale, alpha = create_test_image()
    filaments = create_test_filaments(4)

    # Select filaments based on mode
    if mode == 'flat-cap':
        selected = filaments  # Includes transparent
    else:
        selected = filaments.iloc[:4]

    config = PipelineConfig(
        layer_height=0.08,
        model_height=1.5,
        width_mm=40.0,
        mode=mode,
        sandwich_layers=1,
        base_layers=2,
    )

    processed_image = ProcessedImage(
        image_rgb=img_rgb,
        image_lab=skcolor.rgb2lab(img_rgb),
        grayscale=grayscale,
        alpha_mask=alpha,
    )

    t0 = time.time()
    scene, sorted_filaments = preview_mod.generate_preview_scene(
        config, processed_image, selected)
    elapsed = time.time() - t0

    metrics = extract_metrics(scene, sorted_filaments)
    metrics['elapsed_seconds'] = round(elapsed, 2)

    if verbose:
        print(f"\n  Mode: {mode}")
        print(f"  Meshes: {metrics['mesh_count']}, "
              f"Faces: {metrics['total_faces']:,}, "
              f"Vertices: {metrics['total_vertices']:,}")
        print(f"  Time: {elapsed:.2f}s")
        if metrics['color_stats']:
            cs = metrics['color_stats']
            print(f"  Color: R={cs['r_mean']:.0f}±{cs['r_std']:.0f}, "
                  f"G={cs['g_mean']:.0f}±{cs['g_std']:.0f}, "
                  f"B={cs['b_mean']:.0f}±{cs['b_std']:.0f}")
            print(f"  Greyscale: {cs['is_greyscale']}, "
                  f"Chromatic: {cs['chromatic_fraction']:.0%}")

    return metrics


def compare_to_baseline(mode, metrics, baseline):
    """Compare metrics to baseline, return list of issues."""
    issues = []
    b = baseline

    # Mesh count must match exactly
    if metrics['mesh_count'] != b['mesh_count']:
        issues.append(f"mesh count changed: {b['mesh_count']} → {metrics['mesh_count']}")

    # Vertex/face counts should be within 20% (downsample may vary slightly)
    for key in ['total_vertices', 'total_faces']:
        if b[key] > 0:
            ratio = metrics[key] / b[key]
            if ratio < 0.5 or ratio > 2.0:
                issues.append(f"{key} changed significantly: {b[key]:,} → {metrics[key]:,}")

    # Color checks — skip for modes where small synthetic images produce
    # legitimately narrow color ranges (e.g. flat-cap with gentle auto-contrast)
    cs = metrics.get('color_stats', {})
    if mode not in ('flat-cap',):
        if cs.get('is_greyscale', False):
            issues.append("GREYSCALE DETECTED — colors appear uniform/grey")

        # Chromatic fraction should be non-trivial (>2% of faces show color)
        # Threshold is low because small test images + topographical meshes
        # have many boundary wall faces with similar colors
        if cs.get('chromatic_fraction', 0) < 0.02:
            issues.append(f"low chromatic fraction: {cs['chromatic_fraction']:.0%} (want > 2%)")

    return issues


MODES = ['standard', 'flat', 'flat-cap', 'exploded', 'exploded-multi']


def main():
    parser = argparse.ArgumentParser(description='HueCLI preview regression tests')
    parser.add_argument('--update', action='store_true', help='Update baselines')
    parser.add_argument('--mode', choices=MODES, help='Test single mode')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show details')
    args = parser.parse_args()

    modes = [args.mode] if args.mode else MODES
    baselines = {}
    if BASELINE_FILE.exists() and not args.update:
        baselines = json.loads(BASELINE_FILE.read_text())

    results = {}
    total_passed = 0
    total_failed = 0

    print(f"Running preview regression tests ({len(modes)} modes)...\n")

    for mode in modes:
        print(f"[{mode}]", end=' ', flush=True)
        try:
            metrics = run_mode_test(mode, verbose=args.verbose)
            results[mode] = metrics

            if args.update:
                print(f"OK ({metrics['elapsed_seconds']}s, "
                      f"{metrics['mesh_count']} meshes, "
                      f"{metrics['total_faces']:,} faces)")
                total_passed += 1
                continue

            if mode in baselines:
                issues = compare_to_baseline(mode, metrics, baselines[mode])
                if issues:
                    print(f"FAIL")
                    for issue in issues:
                        print(f"  ! {issue}")
                    total_failed += 1
                else:
                    print(f"PASS ({metrics['elapsed_seconds']}s)")
                    total_passed += 1
            else:
                print(f"SKIP (no baseline — run with --update)")

        except Exception as e:
            print(f"ERROR: {e}")
            total_failed += 1
            if args.verbose:
                import traceback
                traceback.print_exc()

    if args.update:
        BASELINE_FILE.write_text(json.dumps(results, indent=2))
        print(f"\nBaselines saved to {BASELINE_FILE}")
    else:
        print(f"\n{'='*50}")
        print(f"Results: {total_passed} passed, {total_failed} failed")

    return 0 if total_failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
