#!/usr/bin/env python3
"""Verify preview contrast system preserves colors correctly.

Tests the Lab-space contrast approach that stretches lightness and boosts
a/b chrominance proportionally, so colors remain visible at all lightness
levels (not washed out to greyscale).

Run: python3 verify_preview_colors.py
"""
import numpy as np
import sys
sys.path.insert(0, '.')
from generator import STLGenerator, srgb_to_linear, linear_to_srgb

passed = 0
failed = 0


def make_generator():
    """Create minimal STLGenerator for testing contrast methods."""
    dummy_grey = np.zeros((10, 10))
    dummy_filaments = __import__('pandas').DataFrame({
        'name': ['test'], 'rgb': [(0.5, 0.5, 0.5)],
        'transmission_distance': [2.0]
    })
    return STLGenerator(dummy_grey, 10.0, 0.08, 2.0, dummy_filaments)


def test(name, condition, detail):
    global passed, failed
    if condition:
        print(f"  PASS — {name}: {detail}")
        passed += 1
    else:
        print(f"  FAIL — {name}: {detail}")
        failed += 1


g = make_generator()

# --- Test 1: Near-white subtle colors (the greyscale bug case) ---
print("Test 1: Near-white pixels preserve hue (greyscale bug regression)")
# Simulates Beer-Lambert output with translucent filaments: subtle warm tint
preview = np.full((20, 20, 3), [0.95, 0.90, 0.85])  # Subtle warm
preview[10:, :] = [0.92, 0.88, 0.82]  # Slightly more absorbed
mask = np.ones((20, 20), dtype=bool)
result = g._auto_contrast_preview(preview, mask)
# After contrast, the warm tint should be AMPLIFIED, not greyed out
pixel = result[5, 5]
sat = max(pixel) - min(pixel)
test("warm tint saturation", sat > 0.05,
     f"sat={sat:.3f} (want > 0.05), pixel=[{pixel[0]:.3f}, {pixel[1]:.3f}, {pixel[2]:.3f}]")

# Check that R > G > B ordering preserved (warm hue)
test("warm hue order R>G>B", pixel[0] > pixel[1] > pixel[2],
     f"R={pixel[0]:.3f} G={pixel[1]:.3f} B={pixel[2]:.3f}")

# --- Test 2: Grey pixels stay grey ---
print("\nTest 2: Neutral grey stays neutral")
preview = np.full((10, 10, 3), [0.5, 0.5, 0.5])
mask = np.ones((10, 10), dtype=bool)
result = g._auto_contrast_preview(preview, mask)
pixel = result[5, 5]
ratio = max(pixel) / max(min(pixel), 1e-10)
test("channel ratio near 1.0", ratio < 1.02,
     f"ratio={ratio:.4f} (want < 1.02)")

# --- Test 3: Saturated colors retain saturation ---
print("\nTest 3: Saturated colors retain saturation")
preview = np.zeros((20, 20, 3))
preview[:5, :] = [0.0, 0.0, 0.8]    # blue
preview[5:10, :] = [0.9, 0.5, 0.0]  # orange
preview[10:15, :] = [0.0, 0.0, 0.4] # dark blue
preview[15:, :] = [1.0, 0.9, 0.0]   # yellow
mask = np.ones((20, 20), dtype=bool)
result = g._auto_contrast_preview(preview, mask)
for name, (r, c) in [("blue", (2, 5)), ("orange", (7, 5)),
                       ("dark blue", (12, 5)), ("yellow", (17, 5))]:
    p = result[r, c]
    sat = max(p) - min(p)
    test(f"{name} saturation", sat > 0.1,
         f"sat={sat:.3f}, pixel=[{p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}]")

# --- Test 4: Simulated Beer-Lambert output with multiple filaments ---
print("\nTest 4: Beer-Lambert simulation preserves color variety")
# Simulate 5 filaments being layered (dark to light)
filament_colors_srgb = np.array([
    [0.2, 0.1, 0.1],   # dark red
    [0.0, 0.3, 0.0],   # green
    [0.1, 0.1, 0.5],   # blue
    [0.8, 0.6, 0.0],   # yellow/gold
    [0.9, 0.9, 0.9],   # near white
])
filament_colors_lin = srgb_to_linear(filament_colors_srgb)
H, W = 50, 50
light = np.ones((H, W, 3))
rng = np.random.RandomState(42)
for k in range(5):
    thickness = rng.uniform(0.0, 0.5, (H, W))
    td = 2.0
    transmission = np.exp(-thickness / td)[:, :, np.newaxis]
    light = light * transmission + filament_colors_lin[k] * (1.0 - transmission)
preview = linear_to_srgb(np.clip(light, 0, 1))
mask = np.ones((H, W), dtype=bool)
result = g._auto_contrast_preview(preview, mask)

# Check color variance across image (should NOT be greyscale)
r_std = result[:, :, 0].std()
g_std = result[:, :, 1].std()
b_std = result[:, :, 2].std()
test("red channel variance", r_std > 0.02, f"std={r_std:.4f}")
test("green channel variance", g_std > 0.02, f"std={g_std:.4f}")
test("blue channel variance", b_std > 0.02, f"std={b_std:.4f}")

# Check that result is not greyscale (channels differ at most pixels)
diffs = np.abs(result[:, :, 0] - result[:, :, 1]) + \
        np.abs(result[:, :, 1] - result[:, :, 2])
chromatic_frac = (diffs > 0.02).mean()
test("fraction of chromatic pixels", chromatic_frac > 0.3,
     f"{chromatic_frac:.1%} of pixels are chromatic (want > 30%)")

# --- Test 5: Contrast params computed once, applied to multiple layers ---
print("\nTest 5: Consistent params across layers")
# Build layer previews like generate_preview_scene does
light = np.ones((30, 30, 3))
layer_previews = []
for k in range(3):
    thickness = rng.uniform(0.0, 0.3, (30, 30))
    transmission = np.exp(-thickness / 2.0)[:, :, np.newaxis]
    light = light * transmission + filament_colors_lin[k] * (1.0 - transmission)
    layer_previews.append(linear_to_srgb(np.clip(light.copy(), 0, 1)))

mask = np.ones((30, 30), dtype=bool)
params = g._compute_contrast_params(layer_previews[-1], mask)
adjusted = [g._apply_contrast_params(p, mask, params) for p in layer_previews]

# Each adjusted layer should have increasing absorption (darker)
for i in range(1, len(adjusted)):
    prev_mean = adjusted[i-1].mean()
    curr_mean = adjusted[i].mean()
    test(f"layer {i} darker than {i-1}", curr_mean <= prev_mean + 0.01,
         f"mean[{i-1}]={prev_mean:.3f}, mean[{i}]={curr_mean:.3f}")

# --- Test 6: Full L-range image needs no adjustment ---
print("\nTest 6: Full L-range image passes through unchanged")
preview = np.zeros((20, 20, 3))
preview[:10, :] = [0.95, 0.95, 0.90]  # near white (L≈96)
preview[10:, :] = [0.05, 0.05, 0.10]  # near black (L≈5)
mask = np.ones((20, 20), dtype=bool)
params = g._compute_contrast_params(preview, mask)
test("no adjustment needed", params is None,
     f"params={params}")

print(f"\n{'='*50}")
print(f"Results: {passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
