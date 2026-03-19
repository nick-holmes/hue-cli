"""FilamentLibrary — loads CSV, selects best filaments, always-on SA optimization."""

import numpy as np
import pandas as pd
import logging
from skimage import color
from .color_science import compute_effective_color, allocate_layers_td_proportional

logger = logging.getLogger(__name__)


class FilamentLibrary:
    """Manages filament library from CSV"""

    def __init__(self, csv_path):
        self.df = self.load_filaments(csv_path)

    def load_filaments(self, csv_path):
        """Load filament library from CSV file"""
        logger.info(f"Loading filament library from {csv_path}")
        try:
            df = pd.read_csv(csv_path)

            # Strip whitespace from column names
            df.columns = df.columns.str.strip()

            # Support both old and new CSV formats
            if 'Color' in df.columns and 'TD' in df.columns and 'Name' in df.columns:
                # New format: Brand, Type, Color, Name, TD, Tags, etc.
                df = df.rename(columns={
                    'Color': 'color_hex',
                    'TD': 'transmission_distance',
                    'Name': 'name'
                })
                # Add height_offset_mm if not present (default to 0)
                if 'height_offset_mm' not in df.columns:
                    df['height_offset_mm'] = 0.0

            else:
                # Old format: name, color_hex, height_offset_mm, transmission_distance
                required_cols = ['name', 'color_hex', 'height_offset_mm', 'transmission_distance']
                if not all(col in df.columns for col in required_cols):
                    raise ValueError(f"CSV must contain columns: {required_cols} (old format) or Brand,Type,Color,Name,TD (new format)")

            # Parse hex colors to RGB
            df['rgb'] = df['color_hex'].apply(self._hex_to_rgb)
            df['lab'] = df['rgb'].apply(lambda rgb: color.rgb2lab([[rgb]])[0][0])

            logger.info(f"Loaded {len(df)} filaments")
            return df

        except Exception as e:
            logger.error(f"Failed to load filaments: {e}")
            raise

    @staticmethod
    def _hex_to_rgb(hex_color):
        """Convert hex color to RGB tuple (0-1 range)"""
        hex_color = hex_color.lstrip('#')
        # Handle alpha channel if present
        if len(hex_color) == 8:
            hex_color = hex_color[:6]  # Ignore alpha for now
        return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))

    def select_best_filaments(self, target_colors_lab, count, layer_height=0.08, min_color_difference=15.0, randomize=False, use_flat_cap=False, model_height=2.0, physics_aware=True):
        """
        Select best matching filaments using Beer-Lambert physics-aware scoring.

        When physics_aware=True (default), estimates each filament's allocated
        thickness as model_height / color_count and scores by delta-E between
        the target LAB and the Beer-Lambert rendered colour at that thickness.

        When physics_aware=False, falls back to raw filament colour comparison.

        Args:
            min_color_difference: Minimum delta-E between selected filaments to ensure diversity
            randomize: Add random perturbation to selection to get variation
            use_flat_cap: Reserve a slot for transparent cap filament
            model_height: Total model height in mm
            physics_aware: Use Beer-Lambert rendered colour for scoring
        """
        logger.info(f"Selecting {count} unique filaments from library (physics_aware: {physics_aware}, flat_cap: {use_flat_cap})")

        if use_flat_cap:
            result = self._select_with_flat_cap(target_colors_lab, count, layer_height, min_color_difference, randomize, model_height=model_height, physics_aware=physics_aware)
            return result

        # Estimate per-filament thickness for physics-aware scoring
        estimated_thickness = model_height / max(count, 1)

        selected_indices = []
        selected_labs = []

        target_colors_sorted = sorted(enumerate(target_colors_lab), key=lambda x: x[1][0])

        for color_idx, target_lab in target_colors_sorted:
            distances = []

            for idx, row in self.df.iterrows():
                if idx in selected_indices:
                    continue

                filament_lab = row['lab']
                too_similar = False
                for selected_lab in selected_labs:
                    delta_e_to_selected = color.deltaE_ciede2000(
                        np.array([[filament_lab]]),
                        np.array([[selected_lab]])
                    )[0][0]
                    if delta_e_to_selected < min_color_difference:
                        too_similar = True
                        break

                if too_similar:
                    continue

                if physics_aware:
                    # Score by rendered colour at estimated thickness
                    rendered_lab = compute_effective_color(
                        row['rgb'], row['transmission_distance'], estimated_thickness)
                    delta_e = color.deltaE_ciede2000(
                        np.array([[target_lab]]),
                        np.array([[rendered_lab]])
                    )[0][0]

                    # Penalize near-invisible filaments: if opacity at print
                    # thickness is very low, the filament contributes almost
                    # nothing to the output regardless of its color match.
                    td = row['transmission_distance']
                    opacity = 1.0 - np.exp(-estimated_thickness / max(td, 0.1))
                    if opacity < 0.15:
                        # Nearly invisible at this thickness — heavy penalty
                        delta_e += 30.0 * (1.0 - opacity / 0.15)
                else:
                    delta_e = color.deltaE_ciede2000(
                        np.array([[target_lab]]),
                        np.array([[filament_lab]])
                    )[0][0]

                    td = row['transmission_distance']
                    if td > 20:
                        td_penalty = np.log(td / 20) * 5.0
                        delta_e += td_penalty

                if randomize:
                    import random
                    delta_e += random.uniform(-5.0, 5.0)

                distances.append((delta_e, idx))

            if distances:
                distances.sort()
                best_idx = distances[0][1]
                selected_indices.append(best_idx)
                selected_labs.append(self.df.iloc[best_idx]['lab'])
            else:
                logger.warning(f"Could not find suitable filament for target color {color_idx} (L={target_lab[0]:.1f})")
                logger.warning("Consider relaxing constraints or expanding filament library")

        if len(selected_indices) < count:
            logger.warning(f"Only selected {len(selected_indices)} filaments out of {count} requested")

        result = self.df.iloc[selected_indices].reset_index(drop=True)

        # Phase 2: Refine selection using actual allocated thicknesses
        if physics_aware and len(result) > 1:
            result = self._refine_selection(result, target_colors_lab, model_height, layer_height)

        return result

    def _select_with_flat_cap(self, target_colors_lab, count, layer_height, min_color_difference, randomize, model_height=2.0, physics_aware=True):
        """Select filaments for flat-cap mode

        Reserves 1 slot for transparent (cap + gap fill).
        All remaining slots are for color filaments.

        Returns filaments in order: [color1, color2, ..., transparent]
        """
        logger.info("Flat-cap mode: selecting transparent + color filaments")

        if count < 2:
            raise ValueError("Flat-cap mode requires at least 2 colors (1 color + transparent)")

        selected_indices = []

        # 1. Select PURE TRANSPARENT filament (cap + negative fill)
        clear_candidates = []
        for idx, row in self.df.iterrows():
            luminosity = row['lab'][0]
            a_value = abs(row['lab'][1])
            b_value = abs(row['lab'][2])
            td = row['transmission_distance']

            if luminosity > 85 and td > 5.0 and (a_value + b_value) < 10.0:
                color_penalty = (a_value + b_value) * 2.0
                score = luminosity + td * 5.0 - color_penalty
                clear_candidates.append((score, idx))

        if not clear_candidates:
            logger.warning("No pure transparent filaments found, trying less strict...")
            for idx, row in self.df.iterrows():
                luminosity = row['lab'][0]
                td = row['transmission_distance']
                if luminosity > 85 and td > 5.0:
                    score = luminosity + td * 5.0
                    clear_candidates.append((score, idx))

        if not clear_candidates:
            logger.warning("No clear filaments found, using most transparent available")
            best_td_idx = self.df['transmission_distance'].idxmax()
            selected_indices.append(best_td_idx)
        else:
            clear_candidates.sort(reverse=True)
            selected_indices.append(clear_candidates[0][1])

        clear_filament = self.df.iloc[selected_indices[0]]
        logger.info(f"Selected transparent: {clear_filament['name']} "
                     f"(L={clear_filament['lab'][0]:.1f}, TD={clear_filament['transmission_distance']}mm)")

        # 2. Select IMAGE filaments (count - 1 remaining slots)
        num_image_colors = count - 1
        logger.info(f"Selecting {num_image_colors} image filaments to match image")

        # Estimate per-filament thickness for physics-aware scoring
        estimated_thickness = model_height / max(num_image_colors, 1)

        selected_labs = [self.df.iloc[idx]['lab'] for idx in selected_indices]

        target_colors_sorted = sorted(enumerate(target_colors_lab), key=lambda x: x[1][0])

        for color_idx, target_lab in target_colors_sorted[:num_image_colors]:
            distances = []

            for idx, row in self.df.iterrows():
                if idx in selected_indices:
                    continue

                filament_lab = row['lab']
                too_similar = False
                for selected_lab in selected_labs:
                    delta_e_to_selected = color.deltaE_ciede2000(
                        np.array([[filament_lab]]),
                        np.array([[selected_lab]])
                    )[0][0]
                    if delta_e_to_selected < min_color_difference:
                        too_similar = True
                        break

                if too_similar:
                    continue

                if physics_aware:
                    rendered_lab = compute_effective_color(
                        row['rgb'], row['transmission_distance'], estimated_thickness)
                    delta_e = color.deltaE_ciede2000(
                        np.array([[target_lab]]),
                        np.array([[rendered_lab]])
                    )[0][0]
                else:
                    delta_e = color.deltaE_ciede2000(
                        np.array([[target_lab]]),
                        np.array([[filament_lab]])
                    )[0][0]

                if randomize:
                    import random
                    delta_e += random.uniform(-5.0, 5.0)

                distances.append((delta_e, idx))

            if distances:
                distances.sort()
                best_idx = distances[0][1]
                selected_indices.append(best_idx)
                selected_labs.append(self.df.iloc[best_idx]['lab'])
                logger.info(f"  Image color {len(selected_indices)-1}: "
                             f"{self.df.iloc[best_idx]['name']} ({self.df.iloc[best_idx]['color_hex']})")

        if len(selected_indices) < count:
            logger.warning(f"Only selected {len(selected_indices)} filaments out of {count} requested")

        # Reorder: [image colors..., transparent] — transparent goes last
        result_indices = selected_indices[1:]  # image colors
        result_indices.append(selected_indices[0])  # transparent at end

        return self.df.iloc[result_indices].reset_index(drop=True)

    def _select_transparent_filament(self, exclude_indices=None):
        """Select the best transparent filament from the library.

        Looks for colorless, high-TD filaments (L>85, TD>5.0, low a/b tint).
        Falls back progressively to less strict criteria.

        Args:
            exclude_indices: Set of DataFrame indices to skip

        Returns:
            DataFrame index of the selected transparent filament
        """
        exclude = exclude_indices or set()

        # Strict: very light, very transparent, minimal color tint
        clear_candidates = []
        for idx, row in self.df.iterrows():
            if idx in exclude:
                continue
            luminosity = row['lab'][0]
            a_value = abs(row['lab'][1])
            b_value = abs(row['lab'][2])
            td = row['transmission_distance']

            if luminosity > 85 and td > 5.0 and (a_value + b_value) < 10.0:
                color_penalty = (a_value + b_value) * 2.0
                score = luminosity + td * 5.0 - color_penalty
                clear_candidates.append((score, idx))

        if not clear_candidates:
            logger.warning("No pure transparent filaments found (L > 85, TD > 5mm, low a/b), trying less strict...")
            for idx, row in self.df.iterrows():
                if idx in exclude:
                    continue
                luminosity = row['lab'][0]
                td = row['transmission_distance']
                if luminosity > 85 and td > 5.0:
                    score = luminosity + td * 5.0
                    clear_candidates.append((score, idx))

        if not clear_candidates:
            logger.warning("No clear filaments found at all, using most transparent available")
            best_td_idx = None
            best_td = 0
            for idx, row in self.df.iterrows():
                if idx not in exclude:
                    if row['transmission_distance'] > best_td:
                        best_td = row['transmission_distance']
                        best_td_idx = idx
            if best_td_idx is not None:
                return best_td_idx
            # Absolute fallback: first non-excluded
            for idx in self.df.index:
                if idx not in exclude:
                    return idx
        else:
            clear_candidates.sort(reverse=True)
            return clear_candidates[0][1]

    def select_for_exploded(self, target_colors_lab, min_color_difference=5.0, layer_height=0.08, model_height=2.0, physics_aware=True):
        """Select filaments for exploded mode: one per target color + transparent.

        When physics_aware=True, scores filaments by their Beer-Lambert rendered
        colour at the estimated sandwich thickness, rather than raw filament colour.

        Args:
            target_colors_lab: Array of target LAB colors from K-means
            min_color_difference: Minimum delta-E between selected filaments
            layer_height: Layer height in mm (for TD penalty calculation)
            model_height: Model height in mm (for max thickness calculation)
            physics_aware: Use Beer-Lambert rendered colour for scoring

        Returns:
            DataFrame: [color1, color2, ..., transparent] (transparent last)
        """
        logger.info(f"Exploded mode: selecting {len(target_colors_lab)} color filaments + transparent (physics_aware: {physics_aware})")

        # Max material thickness any color can achieve in exploded mode
        total_sandwiches = int(model_height / layer_height)
        max_color_thickness = total_sandwiches * layer_height

        # 1. Auto-select transparent filament
        transparent_idx = self._select_transparent_filament()
        transparent_filament = self.df.iloc[transparent_idx]
        logger.info(f"Selected transparent: {transparent_filament['name']} "
                     f"(L={transparent_filament['lab'][0]:.1f}, TD={transparent_filament['transmission_distance']}mm)")

        # Estimate per-color thickness for physics-aware scoring
        num_colors = len(target_colors_lab)
        estimated_thickness = max_color_thickness / max(num_colors, 1)

        # 2. Select color filaments
        selected_labs = [self.df.iloc[transparent_idx]['lab']]
        color_indices = []

        target_colors_sorted = sorted(enumerate(target_colors_lab), key=lambda x: x[1][0])

        for color_idx, target_lab in target_colors_sorted:
            distances = []

            for idx, row in self.df.iterrows():
                if idx == transparent_idx or idx in color_indices:
                    continue

                filament_lab = row['lab']
                too_similar = False
                for selected_lab in selected_labs:
                    delta_e_to_selected = color.deltaE_ciede2000(
                        np.array([[filament_lab]]),
                        np.array([[selected_lab]])
                    )[0][0]
                    if delta_e_to_selected < min_color_difference:
                        too_similar = True
                        break

                if too_similar:
                    continue

                if physics_aware:
                    rendered_lab = compute_effective_color(
                        row['rgb'], row['transmission_distance'], estimated_thickness)
                    delta_e = color.deltaE_ciede2000(
                        np.array([[target_lab]]),
                        np.array([[rendered_lab]])
                    )[0][0]
                else:
                    delta_e = color.deltaE_ciede2000(
                        np.array([[target_lab]]),
                        np.array([[filament_lab]])
                    )[0][0]

                # TD penalty: high-TD filaments are nearly invisible in thin layers
                td = row['transmission_distance']
                opacity_at_max = 1.0 - np.exp(-max_color_thickness / max(td, 0.1))
                td_penalty = max(0, (1.0 - opacity_at_max) * 50.0)

                score = delta_e + td_penalty
                distances.append((score, idx))

            if distances:
                distances.sort()
                best_idx = distances[0][1]
                best_row = self.df.iloc[best_idx]
                color_indices.append(best_idx)
                selected_labs.append(best_row['lab'])
                opacity = 1.0 - np.exp(-max_color_thickness / max(best_row['transmission_distance'], 0.1))
                logger.info(f"  Color {len(color_indices)}: {best_row['name']} "
                             f"({best_row['color_hex']}, TD={best_row['transmission_distance']:.1f}, "
                             f"max opacity={opacity:.0%})")

        if not color_indices:
            logger.warning("No color filaments selected for exploded mode")

        # Order: [colors..., transparent]
        result_indices = color_indices + [transparent_idx]
        return self.df.iloc[result_indices].reset_index(drop=True)

    def select_for_exploded_cmyk(self, layer_height=0.08, model_height=2.0, sandwich_layers=3, max_color_sandwiches=1):
        """Select filaments for exploded CMYK mode: C/M/Y/K primaries + transparent.

        Uses fixed CMYK target colors instead of K-means. Finds the best filament
        match for each ideal primary, with TD penalty for overly transparent filaments.

        Args:
            sandwich_layers: Color layers per sandwich (default 3 for CMYK)
            max_color_sandwiches: Max sandwiches per colour (default 1)

        Returns:
            DataFrame: [cyan, magenta, yellow, black, transparent] (transparent last)
        """
        logger.info("Exploded CMYK: selecting C/M/Y/K filaments + transparent")

        # Max material thickness per color in CMYK mode
        max_sandwiches_per_color = max_color_sandwiches
        max_color_thickness = max_sandwiches_per_color * sandwich_layers * layer_height

        # 1. Auto-select transparent filament
        transparent_idx = self._select_transparent_filament()
        transparent_filament = self.df.iloc[transparent_idx]
        logger.info(f"Selected transparent: {transparent_filament['name']} "
                     f"(L={transparent_filament['lab'][0]:.1f}, TD={transparent_filament['transmission_distance']}mm)")

        # 2. Define ideal CMYK targets in LAB
        cmyk_targets = {
            'Cyan':    color.rgb2lab([[[0.0, 1.0, 1.0]]])[0][0],
            'Magenta': color.rgb2lab([[[1.0, 0.0, 1.0]]])[0][0],
            'Yellow':  color.rgb2lab([[[1.0, 1.0, 0.0]]])[0][0],
            'Black':   color.rgb2lab([[[0.0, 0.0, 0.0]]])[0][0],
        }

        # 3. Select best filament for each CMYK channel
        selected_indices = []
        exclude = {transparent_idx}

        for channel_name, target_lab in cmyk_targets.items():
            distances = []

            for idx, row in self.df.iterrows():
                if idx in exclude:
                    continue

                filament_lab = row['lab']
                delta_e = color.deltaE_ciede2000(
                    np.array([[target_lab]]),
                    np.array([[filament_lab]])
                )[0][0]

                # Light TD penalty: prefer filaments that can show color in thin layers,
                # but prioritize color accuracy (CMYK needs the right primaries)
                td = row['transmission_distance']
                opacity_at_max = 1.0 - np.exp(-max_color_thickness / max(td, 0.1))
                td_penalty = max(0, (1.0 - opacity_at_max) * 10.0)

                score = delta_e + td_penalty
                distances.append((score, idx))

            if distances:
                distances.sort()
                best_idx = distances[0][1]
                best_row = self.df.iloc[best_idx]
                selected_indices.append(best_idx)
                exclude.add(best_idx)
                opacity = 1.0 - np.exp(-max_color_thickness / max(best_row['transmission_distance'], 0.1))
                delta_e = color.deltaE_ciede2000(
                    np.array([[target_lab]]),
                    np.array([[best_row['lab']]])
                )[0][0]
                logger.info(f"  {channel_name}: {best_row['name']} "
                             f"({best_row['color_hex']}, TD={best_row['transmission_distance']:.1f}, "
                             f"deltaE={delta_e:.1f}, max opacity={opacity:.0%})")

        if len(selected_indices) < 4:
            logger.warning(f"Only found {len(selected_indices)} CMYK filaments (expected 4)")

        # Order: [C, M, Y, K, transparent]
        result_indices = selected_indices + [transparent_idx]
        return self.df.iloc[result_indices].reset_index(drop=True)

    def _refine_selection(self, selected_df, target_colors_lab, model_height, layer_height):
        """Refine filament selection using actual allocated thicknesses.

        After greedy selection, runs layer allocation to get real Z-bands,
        then tries swapping each slot to an unused filament if it improves
        delta-E by >1.0. Repeats until stable.

        Args:
            selected_df: DataFrame of initially selected filaments
            target_colors_lab: Array of target LAB colors
            model_height: Total model height in mm
            layer_height: Layer height in mm

        Returns:
            Refined DataFrame of filaments
        """
        num_layers = int(model_height / layer_height)
        result = selected_df.copy()

        for pass_num in range(3):  # Max 3 passes
            improved = False
            filament_tds = np.array([f['transmission_distance'] for _, f in result.iterrows()])
            layer_counts, _, z_boundaries = allocate_layers_td_proportional(
                filament_tds, num_layers, layer_height)

            # Compute effective colour at actual allocated thickness per slot
            current_scores = []
            for i, (_, row) in enumerate(result.iterrows()):
                thickness = layer_counts[i] * layer_height
                rendered_lab = compute_effective_color(
                    row['rgb'], row['transmission_distance'], thickness)
                # Find nearest target
                best_de = float('inf')
                for t_lab in target_colors_lab:
                    de = color.deltaE_ciede2000(
                        np.array([[t_lab]]),
                        np.array([[rendered_lab]])
                    )[0][0]
                    best_de = min(best_de, de)
                current_scores.append(best_de)

            current_indices = set()
            for _, row in result.iterrows():
                matches = self.df[self.df['name'] == row['name']].index
                if len(matches) > 0:
                    current_indices.add(matches[0])

            # Try swapping each slot
            for slot in range(len(result)):
                thickness = layer_counts[slot] * layer_height
                best_score = current_scores[slot]
                best_replacement = None

                for idx, row in self.df.iterrows():
                    if idx in current_indices:
                        continue

                    rendered_lab = compute_effective_color(
                        row['rgb'], row['transmission_distance'], thickness)
                    best_de = float('inf')
                    for t_lab in target_colors_lab:
                        de = color.deltaE_ciede2000(
                            np.array([[t_lab]]),
                            np.array([[rendered_lab]])
                        )[0][0]
                        best_de = min(best_de, de)

                    if best_de < best_score - 1.0:  # Require >1.0 improvement
                        best_score = best_de
                        best_replacement = idx

                if best_replacement is not None:
                    old_name = result.iloc[slot]['name']
                    # Rebuild DataFrame with replacement
                    rows = []
                    for r in range(len(result)):
                        if r == slot:
                            rows.append(self.df.iloc[best_replacement])
                        else:
                            rows.append(result.iloc[r])
                    result = pd.DataFrame(rows).reset_index(drop=True)

                    # Update current_indices
                    old_matches = self.df[self.df['name'] == old_name].index
                    if len(old_matches) > 0:
                        current_indices.discard(old_matches[0])
                    current_indices.add(best_replacement)
                    improved = True
                    logger.debug(f"  Refinement pass {pass_num+1}: slot {slot} swapped "
                                 f"{old_name} -> {self.df.iloc[best_replacement]['name']} "
                                 f"(deltaE {current_scores[slot]:.1f} -> {best_score:.1f})")

            if not improved:
                break

        return result

    def optimize_filament_set(self, selected_filaments, image_rgb, alpha_mask,
                               layer_height, model_height, num_layers,
                               mode='standard',
                               sandwich_layers=1, use_fill=False):
        """Optimize filament selection via simulated annealing.

        Starts from greedy selection, iteratively swaps filaments and scores
        the set by mean delta-E on a downsampled image using Beer-Lambert simulation.

        Always runs 150 iterations with early stopping (50 no-improvement cutoff).

        Args:
            selected_filaments: DataFrame of initially selected filaments
            image_rgb: HxWx3 RGB image (0-1 range)
            alpha_mask: HxW alpha mask
            layer_height: Layer height in mm
            model_height: Total model height in mm
            num_layers: Total number of layers
            mode: Generation mode string
            sandwich_layers: Color layers per sandwich
            use_fill: Whether fill is enabled

        Returns:
            Optimized DataFrame of filaments
        """
        import random
        from scipy.spatial import cKDTree
        from .color_science import vectorized_beer_lambert

        random.seed(42)
        iterations = 150

        is_flat = mode in ('flat', 'flat-cap')
        is_exploded = mode in ('exploded', 'exploded-multi', 'exploded-cmyk')
        has_transparent = is_flat and mode == 'flat-cap' or is_exploded

        # Downsample image for fast scoring (~50x50 = 2500 pixels)
        H, W = image_rgb.shape[:2]
        from math import sqrt
        ds = max(1, int(sqrt(H * W / 2500)))
        ds_rgb = image_rgb[::ds, ::ds]
        ds_alpha = alpha_mask[::ds, ::ds]
        ds_H, ds_W = ds_rgb.shape[:2]
        ds_lab = color.rgb2lab(ds_rgb)
        alpha_valid = ds_alpha >= 0.5

        logger.info(f"Filament optimization: {iterations} SA iterations on {ds_H}x{ds_W} image")

        def score_filaments(filaments_df):
            """Score a filament set by mean delta-E via Beer-Lambert simulation."""
            filament_rgbs = np.array([f['rgb'] for _, f in filaments_df.iterrows()])
            filament_tds = np.array([f['transmission_distance'] for _, f in filaments_df.iterrows()])

            if is_flat or is_exploded:
                # Use Beer-Lambert combo simulation
                if has_transparent:
                    color_fils = filaments_df.iloc[:-1]
                    trans_fil = filaments_df.iloc[-1]
                    c_rgbs = np.array([f['rgb'] for _, f in color_fils.iterrows()])
                    c_tds = np.array([f['transmission_distance'] for _, f in color_fils.iterrows()])
                    all_rgbs = np.vstack([c_rgbs, np.array(trans_fil['rgb']).reshape(1, 3)])
                    all_tds = np.append(c_tds, trans_fil['transmission_distance'])
                else:
                    c_rgbs = filament_rgbs
                    c_tds = filament_tds
                    all_rgbs = filament_rgbs
                    all_tds = filament_tds

                num_c = len(c_tds)
                from itertools import product as _prod
                caps = [min(3, num_layers) for _ in range(num_c)]
                from math import prod as _mprod
                while _mprod(c + 1 for c in caps) > 50000:
                    mi = caps.index(max(caps))
                    caps[mi] -= 1
                ranges = [range(0, c + 1) for c in caps]
                all_combos = np.array(list(_prod(*ranges)))
                valid = all_combos.sum(axis=1) <= num_layers
                combos = all_combos[valid]

                if is_exploded and has_transparent:
                    color_thickness = layer_height * sandwich_layers
                    trans_per_sandwich = color_thickness if use_fill else layer_height
                    trans_layers = num_layers - combos.sum(axis=1)
                    thickness = np.column_stack([
                        combos * color_thickness,
                        (trans_layers * trans_per_sandwich).reshape(-1, 1)
                    ])
                else:
                    thickness = combos * layer_height

                sim_rgb = vectorized_beer_lambert(all_rgbs, all_tds, thickness)
                sim_lab = color.rgb2lab(sim_rgb.reshape(-1, 1, 3)).reshape(-1, 3)
                tree = cKDTree(sim_lab)
                target_lab = ds_lab.reshape(-1, 3)
                dists, _ = tree.query(target_lab, k=1)
                return float(np.mean(dists[alpha_valid.ravel()]))
            else:
                # Standard mode: score by Beer-Lambert rendered color at allocated thickness
                layer_counts, _, _ = allocate_layers_td_proportional(
                    filament_tds, num_layers, layer_height)
                rendered_labs = []
                for i, (_, f) in enumerate(filaments_df.iterrows()):
                    thickness = layer_counts[i] * layer_height
                    rendered_lab = compute_effective_color(
                        f['rgb'], f['transmission_distance'], thickness)
                    rendered_labs.append(rendered_lab)
                fil_lab = np.array(rendered_labs)
                tree = cKDTree(fil_lab)
                target_lab = ds_lab.reshape(-1, 3)
                dists, _ = tree.query(target_lab, k=1)
                return float(np.mean(dists[alpha_valid.ravel()]))

        # Initial score
        best_filaments = selected_filaments.copy()
        best_score = score_filaments(best_filaments)
        current_filaments = best_filaments.copy()
        current_score = best_score
        initial_score = best_score

        # Identify which slots can be swapped (protect transparent in exploded/flat-cap)
        num_total = len(selected_filaments)
        if has_transparent:
            swappable_slots = list(range(num_total - 1))  # Don't swap transparent
        else:
            swappable_slots = list(range(num_total))

        if not swappable_slots:
            logger.info("No swappable filament slots — skipping optimization")
            return selected_filaments

        # Get all available filament indices
        all_indices = set(self.df.index)
        temperature = 10.0
        cooling = 0.97
        no_improve_count = 0

        for i in range(iterations):
            # Pick a random slot and a random replacement
            slot = random.choice(swappable_slots)
            current_indices = set()
            for _, row in current_filaments.iterrows():
                # Find matching index in library
                matches = self.df[self.df['name'] == row['name']].index
                if len(matches) > 0:
                    current_indices.add(matches[0])

            available = list(all_indices - current_indices)
            if not available:
                continue

            new_idx = random.choice(available)
            # Build candidate by replacing one row via concat (avoids iloc assignment issues with array columns)
            rows = []
            for r in range(len(current_filaments)):
                if r == slot:
                    rows.append(self.df.iloc[new_idx])
                else:
                    rows.append(current_filaments.iloc[r])
            candidate = pd.DataFrame(rows).reset_index(drop=True)

            candidate_score = score_filaments(candidate)
            delta = candidate_score - current_score

            # Accept if better, or probabilistically if worse
            if delta < 0 or random.random() < np.exp(-delta / max(temperature, 0.01)):
                current_filaments = candidate
                current_score = candidate_score

                if current_score < best_score:
                    best_filaments = current_filaments.copy()
                    best_score = current_score
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1

            temperature *= cooling

            # Early stopping
            if no_improve_count >= 50:
                logger.debug(f"  Early stopping at iteration {i+1} (no improvement for 50 iterations)")
                break

        improvement = (initial_score - best_score) / max(initial_score, 0.01) * 100
        logger.info(f"Filament optimization: deltaE {initial_score:.2f} -> {best_score:.2f} "
                     f"({improvement:+.1f}% improvement)")

        # Log selected filaments
        for _, row in best_filaments.iterrows():
            logger.info(f"  Optimized: {row['name']} ({row['color_hex']})")

        return best_filaments
