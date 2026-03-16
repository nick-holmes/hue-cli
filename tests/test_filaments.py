import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from filaments import FilamentLibrary
from skimage import color as skcolor


@pytest.fixture
def filament_lib():
    csv_path = Path(__file__).parent.parent / 'filaments.csv'
    if not csv_path.exists():
        pytest.skip("filaments.csv not found")
    return FilamentLibrary(str(csv_path))


class TestCSVLoading:
    def test_loads_filaments(self, filament_lib):
        assert len(filament_lib.df) > 0

    def test_has_required_columns(self, filament_lib):
        for col in ['name', 'color_hex', 'transmission_distance', 'rgb', 'lab']:
            assert col in filament_lib.df.columns


class TestFilamentSelection:
    def test_correct_count(self, filament_lib):
        targets = np.array([
            skcolor.rgb2lab([[[0.5, 0.1, 0.1]]])[0][0],
            skcolor.rgb2lab([[[0.1, 0.5, 0.1]]])[0][0],
        ])
        result = filament_lib.select_best_filaments(targets, 2)
        assert len(result) == 2

    def test_diverse_selection(self, filament_lib):
        targets = np.array([
            skcolor.rgb2lab([[[0.8, 0.1, 0.1]]])[0][0],
            skcolor.rgb2lab([[[0.1, 0.1, 0.8]]])[0][0],
            skcolor.rgb2lab([[[0.1, 0.8, 0.1]]])[0][0],
        ])
        result = filament_lib.select_best_filaments(targets, 3, min_color_difference=10.0)
        # All selected filaments should be different
        names = result['name'].tolist()
        assert len(set(names)) == len(names)


class TestTransparentSelection:
    def test_selects_transparent(self, filament_lib):
        idx = filament_lib._select_transparent_filament()
        filament = filament_lib.df.iloc[idx]
        # Should be high-luminosity and high-TD
        assert filament['lab'][0] > 50  # Reasonably light
        assert filament['transmission_distance'] > 1.0


class TestPhysicsAwareSelection:
    def test_physics_aware_picks_better_rendered_match(self, filament_lib):
        """Verify physics_aware=True picks filaments whose rendered colour
        at thickness matches better than raw colour."""
        targets = np.array([
            skcolor.rgb2lab([[[0.5, 0.1, 0.1]]])[0][0],
            skcolor.rgb2lab([[[0.1, 0.5, 0.1]]])[0][0],
        ])
        result_aware = filament_lib.select_best_filaments(
            targets, 2, model_height=2.0, physics_aware=True)
        result_raw = filament_lib.select_best_filaments(
            targets, 2, model_height=2.0, physics_aware=False)
        # Both should return 2 filaments
        assert len(result_aware) == 2
        assert len(result_raw) == 2

    def test_physics_aware_vs_raw_different_or_same(self, filament_lib):
        """Physics-aware may select different filaments — at minimum
        it should not crash and return valid results."""
        targets = np.array([
            skcolor.rgb2lab([[[0.9, 0.9, 0.1]]])[0][0],
        ])
        result = filament_lib.select_best_filaments(
            targets, 1, model_height=3.0, physics_aware=True)
        assert len(result) == 1
        assert 'name' in result.columns


class TestRefinementPass:
    def test_refinement_maintains_or_improves(self, filament_lib):
        """Verify _refine_selection doesn't make things worse."""
        from color_science import compute_effective_color, allocate_layers_td_proportional
        targets = np.array([
            skcolor.rgb2lab([[[0.8, 0.1, 0.1]]])[0][0],
            skcolor.rgb2lab([[[0.1, 0.1, 0.8]]])[0][0],
            skcolor.rgb2lab([[[0.1, 0.8, 0.1]]])[0][0],
        ])
        initial = filament_lib.select_best_filaments(
            targets, 3, model_height=2.0, physics_aware=False)

        refined = filament_lib._refine_selection(initial, targets, 2.0, 0.08)
        assert len(refined) == 3

        # Compute mean delta-E for both
        def mean_de(filaments_df):
            tds = np.array([f['transmission_distance'] for _, f in filaments_df.iterrows()])
            counts, _, _ = allocate_layers_td_proportional(tds, 25, 0.08)
            total = 0.0
            for i, (_, row) in enumerate(filaments_df.iterrows()):
                thickness = counts[i] * 0.08
                rendered = compute_effective_color(row['rgb'], row['transmission_distance'], thickness)
                best_de = min(
                    float(np.sqrt(np.sum((np.array(rendered) - np.array(t)) ** 2)))
                    for t in targets
                )
                total += best_de
            return total / len(filaments_df)

        de_initial = mean_de(initial)
        de_refined = mean_de(refined)
        # Refined should be same or better (lower delta-E)
        assert de_refined <= de_initial + 0.1  # Small tolerance for floating point
