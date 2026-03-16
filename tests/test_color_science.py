import numpy as np
import pytest
from color_science import (
    srgb_to_linear, linear_to_srgb, vectorized_beer_lambert,
    sort_filaments_by_luminosity, allocate_layers_td_proportional,
    apply_contrast_enhancement, compute_heightmap, apply_unsharp_mask,
    compute_effective_color, compute_achievable_gamut_sample,
    compute_achievable_gamut,
)
import pandas as pd
from skimage import color as skcolor


def _make_filaments(n=3):
    colors = [
        ('Dark', (0.1, 0.05, 0.05), 1.5),
        ('Mid', (0.5, 0.3, 0.2), 3.0),
        ('Light', (0.9, 0.85, 0.8), 8.0),
    ][:n]
    df = pd.DataFrame({
        'name': [c[0] for c in colors],
        'rgb': [c[1] for c in colors],
        'transmission_distance': [c[2] for c in colors],
    })
    df['lab'] = df['rgb'].apply(lambda rgb: skcolor.rgb2lab([[rgb]])[0][0])
    return df


class TestSrgbRoundTrip:
    def test_identity(self):
        values = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
        result = linear_to_srgb(srgb_to_linear(values))
        np.testing.assert_allclose(result, values, atol=1e-10)

    def test_zero_and_one(self):
        assert srgb_to_linear(0.0) == pytest.approx(0.0)
        assert srgb_to_linear(1.0) == pytest.approx(1.0)
        assert linear_to_srgb(0.0) == pytest.approx(0.0)
        assert linear_to_srgb(1.0) == pytest.approx(1.0)


class TestBeerLambert:
    def test_zero_thickness_is_white(self):
        rgbs = np.array([[0.5, 0.0, 0.0]])
        tds = np.array([2.0])
        combos = np.array([[0.0]])
        result = vectorized_beer_lambert(rgbs, tds, combos)
        np.testing.assert_allclose(result[0], [1.0, 1.0, 1.0], atol=1e-6)

    def test_thick_material_approaches_filament_color(self):
        rgbs = np.array([[0.8, 0.1, 0.1]])
        tds = np.array([1.0])
        combos = np.array([[100.0]])  # Very thick
        result = vectorized_beer_lambert(rgbs, tds, combos)
        # Should be very close to the filament color (in sRGB)
        assert result[0, 0] > 0.7  # Red channel dominant
        assert result[0, 1] < 0.2  # Green low
        assert result[0, 2] < 0.2  # Blue low

    def test_multiple_combos(self):
        rgbs = np.array([[0.5, 0.5, 0.5]])
        tds = np.array([2.0])
        combos = np.array([[0.0], [1.0], [5.0]])
        result = vectorized_beer_lambert(rgbs, tds, combos)
        assert result.shape == (3, 3)
        # Thicker = darker (closer to filament)
        assert result[0, 0] > result[1, 0] > result[2, 0]


class TestSortFilaments:
    def test_dark_to_light_order(self):
        df = _make_filaments(3)
        sorted_df = sort_filaments_by_luminosity(df)
        luminosities = [float(np.asarray(row['lab']).flat[0]) for _, row in sorted_df.iterrows()]
        assert luminosities == sorted(luminosities)


class TestAllocateLayers:
    def test_sum_equals_total(self):
        tds = np.array([1.0, 3.0, 8.0])
        counts, boundaries, z_bounds = allocate_layers_td_proportional(tds, 25, 0.08)
        assert counts.sum() == 25

    def test_min_one_per_color(self):
        tds = np.array([1.0, 1.0, 1.0, 1.0])
        counts, _, _ = allocate_layers_td_proportional(tds, 10, 0.08)
        assert all(c >= 1 for c in counts)

    def test_boundaries_monotonic(self):
        tds = np.array([2.0, 5.0])
        _, boundaries, z_bounds = allocate_layers_td_proportional(tds, 20, 0.08)
        assert all(boundaries[i] <= boundaries[i+1] for i in range(len(boundaries)-1))
        assert all(z_bounds[i] <= z_bounds[i+1] for i in range(len(z_bounds)-1))


class TestContrastEnhancement:
    def test_output_in_range(self):
        brightness = np.random.rand(10, 10)
        alpha = np.ones((10, 10))
        result = apply_contrast_enhancement(brightness, alpha, 2.0)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_no_enhancement_passthrough(self):
        brightness = np.random.rand(10, 10)
        alpha = np.ones((10, 10))
        result = apply_contrast_enhancement(brightness, alpha, 1.0)
        np.testing.assert_array_equal(result, brightness)


class TestHeightmap:
    def test_transparent_pixels_zero(self):
        gray = np.ones((5, 5)) * 0.5
        alpha = np.zeros((5, 5), dtype=bool)
        result = compute_heightmap(gray, alpha, 2.0, 0.08)
        assert result.max() == 0.0

    def test_heights_in_range(self):
        gray = np.random.rand(10, 10)
        alpha = np.ones((10, 10), dtype=bool)
        result = compute_heightmap(gray, alpha, 2.0, 0.08)
        assert result.min() >= 2 * 0.08  # min_height
        assert result.max() <= 2.0


class TestEffectiveColor:
    def test_zero_thickness_is_white(self):
        lab = compute_effective_color((0.5, 0.0, 0.0), 2.0, 0.0)
        # At zero thickness, should be white (L~100, a~0, b~0)
        assert lab[0] > 99.0

    def test_thick_approaches_filament(self):
        rgb = (0.8, 0.1, 0.1)
        lab_thick = compute_effective_color(rgb, 1.0, 100.0)
        filament_lab = skcolor.rgb2lab([[rgb]])[0][0]
        # At extreme thickness, rendered colour should be close to filament
        delta = np.sqrt(np.sum((lab_thick - filament_lab) ** 2))
        assert delta < 5.0

    def test_intermediate_thickness(self):
        lab_thin = compute_effective_color((0.5, 0.5, 0.5), 2.0, 0.5)
        lab_thick = compute_effective_color((0.5, 0.5, 0.5), 2.0, 5.0)
        # Thicker = darker (lower L)
        assert lab_thick[0] < lab_thin[0]


class TestAchievableGamut:
    def test_gamut_sample_shape(self):
        result = compute_achievable_gamut_sample((0.5, 0.2, 0.1), 2.0, 0.0, 2.0, num_samples=7)
        assert result.shape == (7, 3)

    def test_gamut_sample_range(self):
        result = compute_achievable_gamut_sample((0.8, 0.1, 0.1), 1.0, 0.0, 5.0, num_samples=10)
        # L should range from near-white (thin) to near-filament (thick)
        assert result[0, 0] > result[-1, 0]  # First sample is thinner = lighter

    def test_achievable_gamut_cloud(self):
        rgbs = np.array([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8]])
        tds = np.array([1.5, 2.0])
        cloud = compute_achievable_gamut(rgbs, tds, 0.08, 25)
        assert cloud.shape[1] == 3
        assert len(cloud) > 1
        # Should include near-white (all zero thickness)
        white_lab = skcolor.rgb2lab([[[1.0, 1.0, 1.0]]])[0][0]
        min_dist_to_white = np.min(np.sqrt(np.sum((cloud - white_lab) ** 2, axis=1)))
        assert min_dist_to_white < 1.0
