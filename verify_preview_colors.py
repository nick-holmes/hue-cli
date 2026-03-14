"""Verify preview contrast stretch handles both grey and colorful filaments."""
import numpy as np
import sys

# Minimal mock of STLGenerator to test the fixed method
class MockGenerator:
    def _auto_contrast_preview(self, preview_rgb, valid_mask):
        valid_pixels = preview_rgb[valid_mask]
        if len(valid_pixels) > 0:
            lum = (0.299 * preview_rgb[:, :, 0] +
                   0.587 * preview_rgb[:, :, 1] +
                   0.114 * preview_rgb[:, :, 2])
            valid_lum = lum[valid_mask]
            p_lo = np.percentile(valid_lum, 0.5)
            p_hi = np.percentile(valid_lum, 99.5)
            rng = max(p_hi - p_lo, 0.01)
            stretched_lum = np.clip((lum - p_lo) / rng, 0, 1)

            # Multiplicative: preserves channel ratios (safe for near-neutral)
            mul_scale = np.where(lum > 1e-10, stretched_lum / lum, 0.0)
            stretched_mul = np.clip(preview_rgb * mul_scale[:, :, np.newaxis], 0, 1)

            # Subtractive: preserves absolute channel diffs (safe for saturated)
            stretched_sub = np.clip((preview_rgb - p_lo) / rng, 0, 1)

            # Blend by original saturation
            sat = np.max(preview_rgb, axis=2) - np.min(preview_rgb, axis=2)
            blend = np.clip((sat - 0.05) / 0.15, 0, 1)[:, :, np.newaxis]
            preview_rgb = stretched_mul * (1 - blend) + stretched_sub * blend

            # Adaptive gamma
            result_lum = (0.299 * preview_rgb[:, :, 0] +
                          0.587 * preview_rgb[:, :, 1] +
                          0.114 * preview_rgb[:, :, 2])
            valid_result_lum = result_lum[valid_mask]
            median_lum = np.median(valid_result_lum)
            if median_lum > 0.6:
                gamma = np.log(0.5) / np.log(median_lum)
                corrected_lum = np.power(np.clip(result_lum, 1e-10, 1), gamma)
                gamma_scale = np.where(result_lum > 1e-10,
                                       corrected_lum / result_lum, 1.0)
                preview_rgb = preview_rgb * gamma_scale[:, :, np.newaxis]
                preview_rgb = np.clip(preview_rgb, 0, 1)
        return preview_rgb


g = MockGenerator()
passed = 0
failed = 0

# Test 1: Burnt Grey (#45444d) stays near-neutral after contrast stretch
print("Test 1: Burnt Grey hue preservation")
preview = np.full((10, 10, 3), [0.271, 0.267, 0.302])  # #45444d
mask = np.ones((10, 10), dtype=bool)
result = g._auto_contrast_preview(preview, mask)
pixel = result[5, 5]
ratio = max(pixel) / max(min(pixel), 1e-10)
if ratio < 1.15:
    print(f"  PASS — channel ratio {ratio:.3f} (< 1.15)")
    passed += 1
else:
    print(f"  FAIL — channel ratio {ratio:.3f} (>= 1.15), pixel={pixel}")
    failed += 1

# Test 2: Mixed grey palette preserves neutrality
print("Test 2: Mixed grey palette")
greys = [
    [0.271, 0.267, 0.302],  # #45444d
    [0.361, 0.361, 0.361],  # #5c5c5c
    [0.886, 0.886, 0.886],  # #e2e2e2
    [1.0, 1.0, 1.0],        # #ffffff
]
preview = np.zeros((20, 20, 3))
for i, c in enumerate(greys):
    preview[i*5:(i+1)*5, :] = c
mask = np.ones((20, 20), dtype=bool)
result = g._auto_contrast_preview(preview, mask)
max_ratio = 0
for i in range(20):
    for j in range(20):
        p = result[i, j]
        mn = min(p)
        if mn > 0.01:
            r = max(p) / mn
            max_ratio = max(max_ratio, r)
if max_ratio < 1.15:
    print(f"  PASS — max channel ratio {max_ratio:.3f} (< 1.15)")
    passed += 1
else:
    print(f"  FAIL — max channel ratio {max_ratio:.3f} (>= 1.15)")
    failed += 1

# Test 3: Colorful pixels retain saturation (regression test for cyberpunk wash-out)
print("Test 3: Saturated colors retain saturation")
preview = np.zeros((20, 20, 3))
preview[:5, :] = [0.0, 0.0, 0.8]    # blue
preview[5:10, :] = [0.9, 0.5, 0.0]  # orange
preview[10:15, :] = [0.0, 0.0, 0.4] # dark blue
preview[15:, :] = [1.0, 0.9, 0.0]   # yellow
mask = np.ones((20, 20), dtype=bool)
result = g._auto_contrast_preview(preview, mask)
# Check that saturated pixels retain color (saturation > 0.1)
test_pixels = [(2, 5), (7, 5), (12, 5), (17, 5)]
all_saturated = True
for r, c in test_pixels:
    p = result[r, c]
    sat = max(p) - min(p)
    if sat < 0.1:
        print(f"  FAIL — pixel ({r},{c}) lost saturation: {p}, sat={sat:.3f}")
        all_saturated = False
        failed += 1
        break
if all_saturated:
    avg_sat = np.mean([max(result[r, c]) - min(result[r, c]) for r, c in test_pixels])
    print(f"  PASS — avg saturation {avg_sat:.3f} (> 0.1)")
    passed += 1

# Test 4: Contrast stretch still works for non-grey images
print("Test 4: Non-grey image still stretches")
preview = np.zeros((10, 10, 3))
preview[:5, :] = [0.2, 0.05, 0.05]  # dark red
preview[5:, :] = [0.4, 0.1, 0.1]    # slightly brighter red
mask = np.ones((10, 10), dtype=bool)
result = g._auto_contrast_preview(preview, mask)
bright = result[8, 5]
dark = result[2, 5]
if bright[0] > dark[0] + 0.3:
    print(f"  PASS — contrast stretched (dark={dark[0]:.2f}, bright={bright[0]:.2f})")
    passed += 1
else:
    print(f"  FAIL — insufficient stretch (dark={dark[0]:.2f}, bright={bright[0]:.2f})")
    failed += 1

print(f"\n{'='*40}")
print(f"Results: {passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
