import numpy as np
import pytest
from huecli.color_science import (
    srgb_to_linear, linear_to_srgb, vectorized_beer_lambert,
    sort_filaments_by_luminosity, allocate_layers_td_proportional,
    allocate_layers_standard,
    apply_contrast_enhancement, compute_heightmap, apply_unsharp_mask,
    compute_effective_color, compute_achievable_gamut_sample,
    compute_achievable_gamut,
    limit_height_gradient, score_standard_filament_set,
    recommend_model_height, median_smooth, render_standard_preview,
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


class TestAllocateLayersStandard:
    def test_sum_equals_total(self):
        tds = np.array([1.0, 3.0, 8.0])
        counts, boundaries, z_bounds = allocate_layers_standard(tds, 25, 0.08)
        assert counts.sum() == 25

    def test_base_gets_at_least_two(self):
        tds = np.array([1.5, 3.0, 9.0])
        counts, _, _ = allocate_layers_standard(tds, 10, 0.08)
        assert counts[0] >= 2

    def test_min_one_per_color(self):
        tds = np.array([1.0, 3.0, 5.0, 8.0])
        counts, _, _ = allocate_layers_standard(tds, 10, 0.08)
        assert all(c >= 1 for c in counts)

    def test_near_uniform(self):
        # All bands should be within 1 layer of each other
        tds = np.array([1.0, 3.0, 9.0])
        counts, _, _ = allocate_layers_standard(tds, 25, 0.08)
        assert max(counts) - min(counts) <= 1

    def test_base_gets_two_even_when_tight(self):
        # With very few layers, base still gets 2
        tds = np.array([1.0, 3.0, 5.0])
        counts, _, _ = allocate_layers_standard(tds, 4, 0.08)
        assert counts[0] >= 2
        assert all(c >= 1 for c in counts)

    def test_boundaries_monotonic(self):
        tds = np.array([2.0, 5.0])
        _, boundaries, z_bounds = allocate_layers_standard(tds, 20, 0.08)
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
        assert result.min() >= 0.08  # min_height = 1 layer
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


class TestLimitHeightGradient:
    def test_no_change_on_flat(self):
        h = np.ones((10, 10)) * 0.5
        alpha = np.ones((10, 10), dtype=bool)
        result = limit_height_gradient(h, alpha, 0.5)
        np.testing.assert_array_equal(result, h)

    def test_spike_gets_reduced(self):
        h = np.zeros((10, 10))
        alpha = np.ones((10, 10), dtype=bool)
        h[5, 5] = 10.0  # Tall spike
        result = limit_height_gradient(h, alpha, 0.5)
        # Spike should be pulled down significantly
        assert result[5, 5] < 5.0

    def test_preserves_gradual_slope(self):
        h = np.zeros((10, 10))
        alpha = np.ones((10, 10), dtype=bool)
        for i in range(10):
            h[i, :] = i * 0.1  # Gradual slope of 0.1 per pixel
        result = limit_height_gradient(h, alpha, 0.5)  # Max slope 0.5
        np.testing.assert_allclose(result, h, atol=1e-6)

    def test_respects_alpha_mask(self):
        h = np.ones((10, 10)) * 0.5
        alpha = np.zeros((10, 10), dtype=bool)
        alpha[3:7, 3:7] = True
        h[5, 5] = 10.0
        result = limit_height_gradient(h, alpha, 0.5)
        # Non-alpha pixels should remain 0
        assert result[0, 0] == 0.0

    def test_symmetric_spike_reduction(self):
        h = np.zeros((11, 11))
        alpha = np.ones((11, 11), dtype=bool)
        h[5, 5] = 5.0
        result = limit_height_gradient(h, alpha, 1.0)
        # Peak should be reduced; neighbors should form a cone
        assert result[5, 5] < 5.0
        assert result[5, 5] >= result[4, 5]


class TestScoreStandardFilamentSet:
    def test_better_match_scores_lower(self):
        # Create a simple grayscale gradient image
        gray = np.linspace(0, 1, 100).reshape(10, 10)
        alpha = np.ones((10, 10), dtype=bool)
        # Create image LAB from grayscale
        rgb = np.stack([gray]*3, axis=-1)
        lab = skcolor.rgb2lab(rgb)

        # Good match: dark to light filaments
        good_df = _make_filaments(3)
        score_good = score_standard_filament_set(
            good_df, gray, alpha, lab, 2.0, 0.08, 25)

        # Bad match: use only mid-tone filaments
        bad_df = pd.DataFrame({
            'name': ['A', 'B', 'C'],
            'rgb': [(0.5, 0.5, 0.5), (0.55, 0.55, 0.55), (0.6, 0.6, 0.6)],
            'transmission_distance': [2.0, 3.0, 4.0],
        })
        bad_df['lab'] = bad_df['rgb'].apply(lambda rgb: skcolor.rgb2lab([[rgb]])[0][0])
        score_bad = score_standard_filament_set(
            bad_df, gray, alpha, lab, 2.0, 0.08, 25)

        # Good set should score lower (less error)
        assert score_good < score_bad

    def test_returns_finite(self):
        gray = np.random.rand(10, 10)
        alpha = np.ones((10, 10), dtype=bool)
        rgb = np.stack([gray]*3, axis=-1)
        lab = skcolor.rgb2lab(rgb)
        df = _make_filaments(3)
        score = score_standard_filament_set(df, gray, alpha, lab, 2.0, 0.08, 25)
        assert np.isfinite(score)
        assert score >= 0


class TestRecommendModelHeight:
    def test_min_floor(self):
        tds = np.array([1.0, 2.0, 3.0])
        h = recommend_model_height(tds, 3, 0.08)
        assert h >= 1.5

    def test_max_ceiling(self):
        tds = np.array([10.0, 20.0, 30.0])
        h = recommend_model_height(tds, 10, 0.08)
        assert h <= 5.0

    def test_rounds_to_layer_height(self):
        tds = np.array([2.0, 4.0])
        h = recommend_model_height(tds, 4, 0.08)
        # Should be a multiple of layer_height
        assert abs(h / 0.08 - round(h / 0.08)) < 1e-6

    def test_more_colors_more_height(self):
        tds = np.array([2.0, 4.0, 6.0])
        h3 = recommend_model_height(tds[:2], 2, 0.08)
        h5 = recommend_model_height(np.append(tds, [8.0, 10.0]), 5, 0.08)
        assert h5 >= h3


class TestMedianSmooth:
    def test_preserves_shape(self):
        gray = np.random.rand(20, 20)
        alpha = np.ones((20, 20))
        result = median_smooth(gray, alpha, radius=2)
        assert result.shape == gray.shape

    def test_zeros_outside_alpha(self):
        gray = np.ones((10, 10)) * 0.5
        alpha = np.zeros((10, 10))
        result = median_smooth(gray, alpha, radius=1)
        assert result.max() == 0.0

    def test_uniform_region_unchanged(self):
        gray = np.ones((10, 10)) * 0.7
        alpha = np.ones((10, 10))
        result = median_smooth(gray, alpha, radius=2)
        np.testing.assert_allclose(result, gray, atol=1e-10)
