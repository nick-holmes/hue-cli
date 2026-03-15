#!/usr/bin/env python3
"""
HueCLI - Generate multi-color 3D print STLs from images.
Converts images to topographical STL files with Beer-Lambert color stacking.
"""

import argparse
import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
from skimage import color
import trimesh

from generator import STLGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Color scheme palettes — each maps a name to a list of RGB hex colors.
# When --scheme is used, image pixels are remapped to the nearest palette color
# before entering the normal pipeline.
COLOR_SCHEMES = {
    'greyscale': [
        '#000000', '#404040', '#808080', '#B0B0B0', '#FFFFFF',
    ],
    'cyberpunk': [
        '#0D0221', '#FF00FF', '#00FFFF', '#FF6600', '#FFFF00',
        '#8B00FF',
    ],
    'sepia': [
        '#2B1700', '#6B3A1F', '#A0724A', '#C4A47A', '#F5E6C8',
    ],
    'sunset': [
        '#1A0533', '#8B1A4A', '#E94E3D', '#F49D37', '#FFD662',
    ],
    'ocean': [
        '#001B2E', '#014F6B', '#0496A8', '#5CC8D4', '#D1F0F0',
    ],
    'vaporwave': [
        '#2B0A3D', '#FF71CE', '#B967FF', '#01CDFE', '#05FFA1',
    ],
    'autumn': [
        '#2D1B00', '#8B2500', '#CC5500', '#E09540', '#FFD700',
        '#556B2F',
    ],
    'nordic': [
        '#1C2833', '#4A6A7A', '#8EB8C4', '#C8DDE0', '#F0F5F5',
    ],
}


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

    def select_best_filaments(self, target_colors_lab, count, layer_height=0.08, min_color_difference=15.0, randomize=False, use_flat_cap=False, model_height=2.0):
        """
        Select best matching filaments using direct delta-E color matching.

        Each filament's raw color is compared to target colors. The Beer-Lambert
        transmission physics are handled during generation, not selection.

        Penalizes very high-TD filaments (TD > 20) in standard mode since they
        need impractical thicknesses to show color and waste layer budget.

        When use_flat_cap=True, reserves 1 slot for transparent cap,
        and uses remaining slots for color matching.

        Args:
            min_color_difference: Minimum delta-E between selected filaments to ensure diversity
            randomize: Add random perturbation to selection to get variation
            use_flat_cap: Reserve a slot for transparent cap filament
            model_height: Total model height in mm (for TD penalty scaling)
        """
        logger.info(f"Selecting {count} unique filaments from library (randomize: {randomize}, flat_cap: {use_flat_cap})")

        if use_flat_cap:
            result = self._select_with_flat_cap(target_colors_lab, count, layer_height, min_color_difference, randomize)
            return result

        # Standard selection below

        selected_indices = []
        selected_labs = []  # Track LAB colors of selected filaments for diversity check

        # Sort target colors by luminosity (dark to light)
        target_colors_sorted = sorted(enumerate(target_colors_lab), key=lambda x: x[1][0])

        for color_idx, target_lab in target_colors_sorted:
            # Calculate delta-E distance for all filaments
            distances = []

            for idx, row in self.df.iterrows():
                # Skip already selected filaments
                if idx in selected_indices:
                    continue

                # Check if this filament is too similar to already selected ones
                # This ensures color diversity and avoids selecting duplicate colors
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

                # Direct delta-E color comparison
                delta_e = color.deltaE_ciede2000(
                    np.array([[target_lab]]),
                    np.array([[filament_lab]])
                )[0][0]

                # Penalize very high-TD filaments in standard mode.
                # High TD means the filament is nearly transparent — it needs
                # many mm of material to show color. In a model_height tall
                # stack, a filament with TD >> model_height wastes its layer
                # allocation producing almost no visible color.
                td = row['transmission_distance']
                if td > 20:
                    # Logarithmic penalty: TD=20 -> +0, TD=100 -> +8, TD=200 -> +11
                    td_penalty = np.log(td / 20) * 5.0
                    delta_e += td_penalty
                    logger.debug(f"  TD penalty for {row['name']}: +{td_penalty:.1f} (TD={td})")

                # Add random perturbation if randomize mode enabled
                if randomize:
                    import random
                    # Add random noise ±5.0 to delta_e to get variation
                    delta_e += random.uniform(-5.0, 5.0)

                distances.append((delta_e, idx))

            # Select best match that hasn't been selected yet
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
            logger.warning("Try reducing --color-count or using --no-split-colors")

        result = self.df.iloc[selected_indices].reset_index(drop=True)
        return result

    def _select_with_flat_cap(self, target_colors_lab, count, layer_height, min_color_difference, randomize):
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

    def select_for_exploded(self, target_colors_lab, min_color_difference=5.0, layer_height=0.08, model_height=2.0):
        """Select filaments for exploded mode: one per target color + transparent.

        Penalizes high-TD filaments since each sandwich layer is only layer_height
        thick — very transparent filaments can't produce visible color in thin layers.

        Args:
            target_colors_lab: Array of target LAB colors from K-means
            min_color_difference: Minimum delta-E between selected filaments
            layer_height: Layer height in mm (for TD penalty calculation)
            model_height: Model height in mm (for max thickness calculation)

        Returns:
            DataFrame: [color1, color2, ..., transparent] (transparent last)
        """
        logger.info(f"Exploded mode: selecting {len(target_colors_lab)} color filaments + transparent")

        # Max material thickness any color can achieve in exploded mode
        # (all sandwich slots assigned to one color)
        total_sandwiches = int(model_height / layer_height)
        max_color_thickness = total_sandwiches * layer_height

        # 1. Auto-select transparent filament
        transparent_idx = self._select_transparent_filament()
        transparent_filament = self.df.iloc[transparent_idx]
        logger.info(f"Selected transparent: {transparent_filament['name']} "
                     f"(L={transparent_filament['lab'][0]:.1f}, TD={transparent_filament['transmission_distance']}mm)")

        # 2. Select color filaments via delta-E matching with TD penalty
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

                delta_e = color.deltaE_ciede2000(
                    np.array([[target_lab]]),
                    np.array([[filament_lab]])
                )[0][0]

                # TD penalty: high-TD filaments are nearly invisible in thin layers.
                # Beer-Lambert opacity = 1 - exp(-thickness / TD)
                # At max thickness, what fraction of color can this filament show?
                td = row['transmission_distance']
                opacity_at_max = 1.0 - np.exp(-max_color_thickness / max(td, 0.1))
                # Filaments that can only show <50% opacity at max thickness get penalized
                # heavily — they'll waste sandwich slots producing near-identical layers.
                # Scale: opacity=1.0 → penalty=0, opacity=0.1 → penalty=+40
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

class ImageProcessor:
    """Handles image loading and color quantization"""

    def __init__(self, image_path, width_mm, color_count):
        self.image_path = image_path
        self.width_mm = width_mm
        self.color_count = color_count
        self.image = None
        self.image_lab = None
        self.alpha_mask = None  # Mask for transparent/rounded corners

    def load_and_prepare(self, nozzle_diameter=0.2):
        """Load image and prepare for processing

        CRITICAL: Resolution is set to match nozzle diameter for maximum quality
        - Each pixel = one nozzle width
        - This matches the high-quality example STL approach (0.2mm pixels)
        """
        logger.info(f"Loading image from {self.image_path}")

        try:
            img_orig = Image.open(self.image_path)

            # Check if image has alpha channel
            has_alpha = img_orig.mode in ('RGBA', 'LA') or (img_orig.mode == 'P' and 'transparency' in img_orig.info)

            if has_alpha:
                logger.info("Image has alpha channel - will preserve transparency for rounded corners")
                img_rgba = img_orig.convert('RGBA')
            else:
                img_rgba = img_orig.convert('RGB')

            # UPDATED: Calculate target size based on NOZZLE DIAMETER for maximum detail
            # Each pixel = one nozzle width (e.g., 0.2mm nozzle = 5 pixels per mm)
            # This matches the example STL's approach and maximizes print quality
            pixels_per_mm = 1.0 / nozzle_diameter
            target_width_px = int(self.width_mm * pixels_per_mm)

            logger.info(f"Target resolution: {pixels_per_mm:.1f} pixels/mm (pixel size = {nozzle_diameter}mm = nozzle diameter)")

            # Maintain aspect ratio
            aspect_ratio = img_rgba.height / img_rgba.width
            target_height_px = int(target_width_px * aspect_ratio)

            img_rgba = img_rgba.resize((target_width_px, target_height_px), Image.Resampling.LANCZOS)
            logger.info(f"Resized image to {target_width_px}x{target_height_px} pixels")

            # Convert to numpy array
            img_array = np.array(img_rgba) / 255.0  # Normalize to 0-1

            # Extract alpha channel if present
            if has_alpha:
                self.alpha_mask = img_array[:, :, 3]  # Alpha channel
                self.image = img_array[:, :, :3]  # RGB channels
                logger.info(f"Extracted alpha mask - transparency range: {self.alpha_mask.min():.2f} to {self.alpha_mask.max():.2f}")
            else:
                self.image = img_array[:, :, :3] if img_array.shape[2] >= 3 else img_array
                # Create a full mask (all opaque)
                self.alpha_mask = np.ones((target_height_px, target_width_px))

            # Flip to correct orientation for 3D printing
            # Vertical flip: image Y-axis points down, 3D Y-axis points up
            self.image = np.flipud(self.image)
            self.alpha_mask = np.flipud(self.alpha_mask)

            # DISABLED: Smoothing can reduce detail in high-res images
            # The continuous heightmap creates smooth slopes naturally
            # For maximum detail (Pokemon cards), skip smoothing entirely
            # if self.preserve_details:
            #     self.image = self._apply_selective_smoothing(self.image)

            # Convert to LAB color space
            self.image_lab = color.rgb2lab(self.image)

            return self.image

        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            raise

    def quantize_colors(self):
        """Extract dominant colors using K-means clustering

        Ensures extreme colors (very dark, very light) are included for text/highlights.
        """
        logger.info(f"Extracting {self.color_count} dominant colors")

        # Reshape image for K-means
        pixels_lab = self.image_lab.reshape(-1, 3)

        # Filter out transparent pixels so background doesn't waste K-means centroids
        alpha_flat = self.alpha_mask.reshape(-1)
        opaque_mask = alpha_flat >= 0.5
        pixels_lab = pixels_lab[opaque_mask]
        logger.info(f"Clustering {len(pixels_lab)} opaque pixels (filtered {(~opaque_mask).sum()} transparent)")

        # Check for extreme values that should be preserved (text, highlights)
        min_L = pixels_lab[:, 0].min()
        max_L = pixels_lab[:, 0].max()

        # If we have very dark pixels (text), ensure they're represented
        has_very_dark = min_L < 25  # Black/very dark text
        has_very_light = max_L > 90  # White/very light areas

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=self.color_count, random_state=42, n_init=10)
        kmeans.fit(pixels_lab)

        # Get cluster centers (dominant colors in LAB)
        dominant_colors_lab = kmeans.cluster_centers_

        # Force include very dark color if we detected dark pixels but K-means missed them
        if has_very_dark and dominant_colors_lab[:, 0].min() > 25:
            logger.info(f"Forcing very dark color for text (min L={min_L:.1f})")
            # Replace darkest cluster with actual darkest pixels
            very_dark_pixels = pixels_lab[pixels_lab[:, 0] < 25]
            darkest_cluster_idx = np.argmin(dominant_colors_lab[:, 0])
            dominant_colors_lab[darkest_cluster_idx] = np.mean(very_dark_pixels, axis=0)

        # Sort by luminosity (L channel): darkest to lightest
        sorted_indices = np.argsort(dominant_colors_lab[:, 0])
        dominant_colors_lab = dominant_colors_lab[sorted_indices]

        logger.info(f"Extracted colors (L values): {dominant_colors_lab[:, 0]}")

        return dominant_colors_lab, kmeans, sorted_indices

    def auto_determine_color_count(self, min_delta_e=10.0, max_colors=12):
        """Determine optimal number of colors by iterating K-means until new clusters add little.

        Starts at K=2, increments K. For each K+1 result, finds the "new" cluster center
        (highest minimum delta-E to any K center). If that min delta-E < threshold, stop.

        Args:
            min_delta_e: Minimum delta-E a new color must contribute to be worthwhile
            max_colors: Maximum number of colors to try

        Returns:
            Optimal color count (int)
        """
        pixels_lab = self.image_lab.reshape(-1, 3)
        alpha_flat = self.alpha_mask.reshape(-1)
        pixels_lab = pixels_lab[alpha_flat >= 0.5]

        logger.info(f"Auto-determining color count (min delta-E={min_delta_e}, max={max_colors})...")

        prev_centers = None
        optimal_k = 2

        for k in range(2, max_colors + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(pixels_lab)
            centers = kmeans.cluster_centers_

            if prev_centers is not None:
                # Find the "new" center: the one with highest min-distance to previous centers
                best_new_delta_e = 0
                for center in centers:
                    # Min delta-E from this center to any previous center
                    min_de = min(
                        color.deltaE_ciede2000(
                            np.array([[center]]),
                            np.array([[pc]])
                        )[0][0]
                        for pc in prev_centers
                    )
                    best_new_delta_e = max(best_new_delta_e, min_de)

                logger.info(f"  K={k}: new cluster delta-E={best_new_delta_e:.1f}")

                if best_new_delta_e < min_delta_e:
                    logger.info(f"  Stopping at K={k-1} (new color delta-E {best_new_delta_e:.1f} < {min_delta_e})")
                    break

                optimal_k = k
            else:
                optimal_k = k
                logger.info(f"  K={k}: initial clustering")

            prev_centers = centers

        logger.info(f"Optimal color count: {optimal_k}")
        return optimal_k


def prompt_with_default(prompt_text, default_value, value_type=str):
    """Prompt user for input with a default value

    Args:
        prompt_text: The question to ask
        default_value: Default value if user presses Enter
        value_type: Type to convert input to (str, int, float)

    Returns:
        User input or default value
    """
    if value_type == bool:
        default_str = "yes" if default_value else "no"
        user_input = input(f"{prompt_text} [{default_str}]: ").strip().lower()
        if user_input == "":
            return default_value
        return user_input in ['y', 'yes', 'true', '1']
    else:
        user_input = input(f"{prompt_text} [{default_value}]: ").strip()
        if user_input == "":
            return default_value
        try:
            return value_type(user_input)
        except ValueError:
            print(f"Invalid input, using default: {default_value}")
            return default_value



def show_3d_preview(stl_gen):
    """Generate and display an interactive 3D preview in the default browser

    Exports the scene as GLB, embeds it as base64 in a custom HTML viewer
    that uses Three.js (CDN) with GLTFLoader.parse() to avoid data-URL
    size limits that cause blank pages in some browsers.

    Args:
        stl_gen: STLGenerator instance with all parameters configured
    """
    import base64
    import json
    import tempfile
    import webbrowser

    logger.info("Generating 3D preview scene...")
    scene, preview_filaments = stl_gen.generate_preview_scene()

    if len(scene.geometry) == 0:
        logger.warning("No geometry generated for 3D preview")
        return

    logger.info("Exporting scene to GLB...")
    glb_data = scene.export(file_type="glb")
    b64 = base64.b64encode(glb_data).decode("utf-8")
    logger.info(f"GLB size: {len(glb_data)/1024/1024:.1f} MB")

    # Build filament info for sidebar
    total_faces = sum(g.faces.shape[0] for g in scene.geometry.values())
    filament_info = []
    for k, (_, f) in enumerate(preview_filaments.iterrows()):
        prefix = f"S{k+1:02d}_"
        mesh_faces = sum(g.faces.shape[0] for name, g in scene.geometry.items() if name.startswith(prefix))
        coverage = (mesh_faces / total_faces * 100) if total_faces > 0 else 0
        filament_info.append({
            'name': f['name'],
            'hex': '#%02x%02x%02x' % tuple((np.array(f['rgb']) * 255).astype(int)),
            'brand': f.get('Brand', ''),
            'td': round(float(f['transmission_distance']), 2),
            'coverage': round(coverage, 1),
            'layer': k + 1,
            'meshPrefix': f"S{k+1:02d}",
        })
    filament_info_json = json.dumps(filament_info)

    is_exploded = stl_gen.use_exploded or stl_gen.use_exploded_multi or stl_gen.use_exploded_cmyk
    html = _build_viewer_html(b64, use_transparency=is_exploded, use_slider=True,
                               default_gap=2.0 if is_exploded else 0.0,
                               filament_info_json=filament_info_json)

    with tempfile.NamedTemporaryFile('w', suffix='.html', delete=False) as f:
        f.write(html)
        tmp_path = f.name

    logger.info("Opening 3D preview in browser...")
    webbrowser.open('file://' + tmp_path)


def _build_viewer_html(b64_glb, use_transparency=False, use_slider=False, default_gap=2.0,
                       filament_info_json='[]'):
    """Build a self-contained HTML page with an embedded Three.js GLB viewer.

    Uses CDN-loaded Three.js with GLTFLoader.parse() to decode the base64
    GLB data directly into an ArrayBuffer, bypassing data-URL size limits.

    Args:
        b64_glb: base64-encoded GLB binary string
        use_transparency: if True, enable transparent materials with vertex alpha
        use_slider: if True, show layer gap slider and parse S## mesh names
        default_gap: initial gap value for the slider (0.0 for standard/flat)
        filament_info_json: JSON string of filament metadata for sidebar

    Returns:
        Complete HTML string
    """
    show_controls = use_transparency or use_slider
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HueCLI — 3D Preview</title>
<style>
  body {{ margin: 0; overflow: hidden; background: #2a2a2a; }}
  canvas {{ display: block; }}
  #info {{
    position: absolute; bottom: 70px; left: 50%; transform: translateX(-50%);
    color: #ccc; font: 13px/1.4 system-ui, sans-serif; pointer-events: none;
    text-align: center;
  }}
  #error {{
    position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%);
    color: #f44; font: 16px system-ui; text-align: center; display: none;
  }}
  #controls {{
    position: absolute; bottom: 20px; left: 50%; transform: translateX(-50%);
    display: {'flex' if show_controls else 'none'}; align-items: center; gap: 12px;
    background: rgba(0,0,0,0.6); padding: 8px 16px; border-radius: 8px;
  }}
  #controls label {{ color: #ccc; font: 13px system-ui; white-space: nowrap; }}
  #gap-slider {{ width: 200px; cursor: pointer; }}
  #gap-value {{ color: #fff; font: 13px monospace; min-width: 40px; }}
  #color-mode {{
    display: flex; border-radius: 4px; overflow: hidden;
    border: 1px solid rgba(255,255,255,0.3);
  }}
  #color-mode button {{
    background: rgba(255,255,255,0.08); color: rgba(255,255,255,0.5);
    border: none; padding: 4px 14px; cursor: pointer; font: 13px system-ui;
    white-space: nowrap; transition: background 0.15s, color 0.15s;
  }}
  #color-mode button:hover {{ background: rgba(255,255,255,0.15); }}
  #color-mode button.active {{
    background: rgba(255,255,255,0.3); color: #fff;
    cursor: default;
  }}
  #info-btn {{
    width: 22px; height: 22px; border-radius: 50%; border: 1px solid rgba(255,255,255,0.3);
    background: rgba(255,255,255,0.08); color: rgba(255,255,255,0.5);
    font: italic 13px Georgia, serif; cursor: pointer; display: flex;
    align-items: center; justify-content: center; flex-shrink: 0;
    transition: background 0.15s, color 0.15s;
  }}
  #info-btn:hover {{ background: rgba(255,255,255,0.2); color: #fff; }}
  #info-tooltip {{
    display: none; position: absolute; bottom: 60px; right: 10px;
    background: rgba(0,0,0,0.85); color: #ddd; font: 12px/1.5 system-ui;
    padding: 10px 14px; border-radius: 8px; max-width: 300px;
    border: 1px solid rgba(255,255,255,0.15);
  }}
  #info-tooltip strong {{ color: #fff; }}
  #info-tooltip.visible {{ display: block; }}

  /* Sidebar */
  #sidebar {{
    position: absolute; top: 0; left: 0; bottom: 0; width: 180px;
    background: rgba(0,0,0,0.7); backdrop-filter: blur(8px);
    display: flex; flex-direction: column; padding: 12px 10px;
    transition: transform 0.25s ease; z-index: 10;
    overflow-y: auto; border-right: 1px solid rgba(255,255,255,0.1);
    scrollbar-width: thin; scrollbar-color: rgba(255,255,255,0.2) transparent;
  }}
  #sidebar.collapsed {{ transform: translateX(-100%); }}
  #sidebar-header {{
    text-align: center;
    margin-bottom: 10px; flex-shrink: 0;
  }}
  #sidebar-header span {{ color: #fff; font: bold 13px system-ui, sans-serif; }}
  #sidebar-toggle {{
    position: absolute; top: 12px; left: 180px; z-index: 11;
    width: 24px; height: 28px; border: 1px solid rgba(255,255,255,0.2);
    border-left: none; border-radius: 0 6px 6px 0;
    background: rgba(0,0,0,0.6); color: rgba(255,255,255,0.6);
    cursor: pointer; font: 12px system-ui; display: flex;
    align-items: center; justify-content: center;
    transition: left 0.25s ease, background 0.15s;
  }}
  #sidebar-toggle:hover {{ background: rgba(0,0,0,0.8); color: #fff; }}
  #sidebar.collapsed ~ #sidebar-toggle {{ left: 0; }}
  .swatch-item {{
    text-align: center; margin-bottom: 10px; cursor: pointer;
    padding: 6px 4px; border-radius: 6px; transition: opacity 0.15s;
    position: relative;
  }}
  .swatch-item:hover {{ background: rgba(255,255,255,0.08); }}
  .swatch-item.hidden {{ opacity: 0.25; }}
  .swatch-color {{
    width: 44px; height: 44px; border-radius: 8px; margin: 0 auto;
    border: 2px solid rgba(255,255,255,0.2);
  }}
  .swatch-name {{
    color: #ccc; font: 11px system-ui, sans-serif; margin-top: 4px;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  }}
  .swatch-tooltip {{
    position: fixed; background: rgba(0,0,0,0.92);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 8px; padding: 10px 12px; color: #ddd;
    font: 12px/1.6 system-ui, sans-serif;
    white-space: nowrap; pointer-events: none; z-index: 20;
  }}
  .swatch-tooltip strong {{ color: #fff; }}
</style>
</head>
<body>
<div id="info">Drag to rotate &middot; Scroll to zoom &middot; Right-drag to pan</div>
<div id="error"></div>
<div id="controls">
  <label for="gap-slider">Gap:</label>
  <input type="range" id="gap-slider" min="0" max="5" value="{default_gap:.1f}" step="0.1">
  <span id="gap-value">{default_gap:.1f}mm</span>
  <div id="color-mode">
    <button id="btn-realistic" class="active">Realistic</button>
    <button id="btn-filaments">Filaments</button>
  </div>
  <button id="info-btn">i</button>
</div>
<div id="info-tooltip">
  <strong>Realistic</strong> &mdash; Simulates light passing through stacked filament layers (Beer-Lambert), showing how the print will actually look when backlit.<br><br>
  <strong>Filaments</strong> &mdash; Shows the raw filament colour for each layer, so you can see which filament is assigned where. Layers separate slightly for visibility.
</div>

<div id="sidebar">
  <div id="sidebar-header">
    <span>Filaments</span>
  </div>
  <div id="swatch-list"></div>
</div>
<button id="sidebar-toggle">&#9664;</button>

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
import {{ GLTFLoader }} from 'three/addons/loaders/GLTFLoader.js';

const base64 = "{b64_glb}";
const FILAMENT_DATA = {filament_info_json};

// Decode base64 → ArrayBuffer (avoids data-URL size limits)
const bin = atob(base64);
const buf = new Uint8Array(bin.length);
for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);

const scene    = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a1a);

const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.toneMapping = THREE.NoToneMapping;
document.body.appendChild(renderer.domElement);

// Lighting — dim ambient only; mesh uses emissive material for backlit effect
const ambient = new THREE.AmbientLight(0xffffff, 0.15);
scene.add(ambient);

// Camera (will be repositioned after model loads)
const camera = new THREE.PerspectiveCamera(
  45, window.innerWidth / window.innerHeight, 0.01, 10000
);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.12;

// Parse GLB
const loader = new GLTFLoader();
const useTransparency = {'true' if use_transparency else 'false'};
const useSlider = {'true' if use_slider else 'false'};

// Track meshes for slider + color toggle
const layerMeshes = [];  // {{mesh, idx, filamentColor, realisticColors}}
let showFilamentColors = false;
let gapBeforeFilamentMode = null;  // stored gap when entering filament mode

loader.parse(buf.buffer, '', (gltf) => {{
  gltf.scene.traverse((child) => {{
    if (child.isMesh) {{
      const colorAttr = child.geometry.attributes.color;
      const hasAlpha = useTransparency && colorAttr && colorAttr.itemSize === 4;

      let opacity = 1.0;
      if (hasAlpha) {{
        opacity = colorAttr.getW(0);
        if (opacity > 1.0) opacity = opacity / 255.0;
      }}

      // Parse layer index from mesh name for polygon offset
      let layerIdx = 0;
      const idxMatch = child.name.match(/^S(\\d+)/);
      if (idxMatch) layerIdx = parseInt(idxMatch[1], 10);

      child.material = new THREE.MeshBasicMaterial({{
        vertexColors: true,
        transparent: hasAlpha,
        opacity: opacity,
        depthWrite: !hasAlpha,
        side: hasAlpha ? THREE.DoubleSide : THREE.FrontSide,
        polygonOffset: true,
        polygonOffsetFactor: -layerIdx,
        polygonOffsetUnits: -layerIdx,
      }});
      child.renderOrder = layerIdx;

      // Parse mesh name: "S01_Name_Crrggbb" or "S01_Name_L1"
      if (useSlider) {{
        const m = idxMatch;
        if (m) {{
          const entry = {{ mesh: child, idx: parseInt(m[1], 10) - 1,
                          filamentColor: null, realisticColors: null }};

          // Extract filament hex color from name (e.g. "_Cff6f4f")
          const cm = child.name.match(/_C([0-9a-f]{{6}})$/i);
          if (cm && colorAttr) {{
            const hex = cm[1];
            const ri = parseInt(hex.slice(0,2), 16);
            const gi = parseInt(hex.slice(2,4), 16);
            const bi = parseInt(hex.slice(4,6), 16);
            // Store both int (0-255) and float (0-1) for different array types
            entry.filamentColor = [ri, gi, bi];
            entry.filamentColorF = [ri / 255, gi / 255, bi / 255];

            // Save realistic (Beer-Lambert) vertex colors
            const arr = colorAttr.array;
            entry.realisticColors = new Float32Array(arr.length);
            entry.realisticColors.set(arr);
          }}

          layerMeshes.push(entry);
        }}
      }}
    }}
  }});
  scene.add(gltf.scene);

  // Apply initial gap from slider
  updateGap();

  // Auto-fit camera to model bounds (after gap applied)
  const box    = new THREE.Box3().setFromObject(gltf.scene);
  const center = box.getCenter(new THREE.Vector3());
  const size   = box.getSize(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z);
  const dist   = maxDim * 2.0;

  camera.position.set(center.x, center.y - dist * 0.6, center.z + dist * 0.8);
  camera.near = maxDim * 0.001;
  camera.far  = maxDim * 100;
  camera.updateProjectionMatrix();

  controls.target.copy(center);
  controls.update();
}},
(err) => {{
  console.error('GLB parse error:', err);
  const el = document.getElementById('error');
  el.textContent = 'Failed to load 3D model — see console for details.';
  el.style.display = 'block';
}});

// Gap slider: reposition meshes along Z
function updateGap() {{
  const slider = document.getElementById('gap-slider');
  const label = document.getElementById('gap-value');
  if (!slider) return;
  const gap = parseFloat(slider.value);
  label.textContent = gap.toFixed(1) + 'mm';
  for (const entry of layerMeshes) {{
    entry.mesh.position.z = entry.idx * gap;
  }}
}}

// Color toggle: switch between realistic (Beer-Lambert) and solid filament colors
function applyColorMode() {{
  const btnR = document.getElementById('btn-realistic');
  const btnF = document.getElementById('btn-filaments');
  if (btnR && btnF) {{
    btnR.classList.toggle('active', !showFilamentColors);
    btnF.classList.toggle('active', showFilamentColors);
  }}

  for (const entry of layerMeshes) {{
    if (!entry.filamentColor || !entry.realisticColors) continue;
    const colorAttr = entry.mesh.geometry.attributes.color;
    if (!colorAttr) continue;
    const arr = colorAttr.array;
    const itemSize = colorAttr.itemSize;
    const count = colorAttr.count;

    if (showFilamentColors) {{
      // Use int (0-255) for Uint8/Uint16 arrays, float (0-1) for Float arrays
      const useInt = arr instanceof Uint8Array || arr instanceof Uint16Array
                     || arr instanceof Uint8ClampedArray;
      const [r, g, b] = useInt ? entry.filamentColor : entry.filamentColorF;
      for (let i = 0; i < count; i++) {{
        arr[i * itemSize] = r;
        arr[i * itemSize + 1] = g;
        arr[i * itemSize + 2] = b;
      }}
    }} else {{
      arr.set(entry.realisticColors);
    }}
    colorAttr.needsUpdate = true;
  }}
}}

const slider = document.getElementById('gap-slider');
if (slider) {{
  slider.addEventListener('input', () => {{
    updateGap();
    // Re-center camera target on the midpoint of the stack
    if (layerMeshes.length > 0) {{
      const maxIdx = Math.max(...layerMeshes.map(s => s.idx));
      const gap = parseFloat(slider.value);
      const midZ = (maxIdx * gap) / 2;
      controls.target.z = midZ;
    }}
  }});
}}

// Segmented control: Realistic / Filaments
function setColorMode(filaments) {{
  if (showFilamentColors === filaments) return;
  const slider = document.getElementById('gap-slider');

  if (filaments && slider) {{
    // Entering filament mode: save current gap, apply minimum gap so layers are visible
    gapBeforeFilamentMode = parseFloat(slider.value);
    if (gapBeforeFilamentMode < 0.5) {{
      slider.value = '0.5';
      updateGap();
    }}
  }} else if (!filaments && slider && gapBeforeFilamentMode !== null) {{
    // Leaving filament mode: restore previous gap
    slider.value = String(gapBeforeFilamentMode);
    gapBeforeFilamentMode = null;
    updateGap();
  }}

  showFilamentColors = filaments;
  applyColorMode();
}}

const btnR = document.getElementById('btn-realistic');
const btnF = document.getElementById('btn-filaments');
if (btnR) btnR.addEventListener('click', () => setColorMode(false));
if (btnF) btnF.addEventListener('click', () => setColorMode(true));

const infoBtn = document.getElementById('info-btn');
const infoTip = document.getElementById('info-tooltip');
if (infoBtn && infoTip) {{
  infoBtn.addEventListener('click', (e) => {{
    e.stopPropagation();
    infoTip.classList.toggle('visible');
  }});
  document.addEventListener('click', () => infoTip.classList.remove('visible'));
}}

window.addEventListener('resize', () => {{
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}});

// --- Sidebar: build swatches from FILAMENT_DATA ---
const swatchList = document.getElementById('swatch-list');
const tooltip = document.createElement('div');
tooltip.className = 'swatch-tooltip';
tooltip.style.display = 'none';
document.body.appendChild(tooltip);

FILAMENT_DATA.forEach((f, i) => {{
  const item = document.createElement('div');
  item.className = 'swatch-item';
  item.dataset.prefix = f.meshPrefix;
  item.innerHTML = `
    <div class="swatch-color" style="background:${{f.hex}}"></div>
    <div class="swatch-name">${{f.name}}</div>
  `;

  // Hover tooltip
  item.addEventListener('mouseenter', (e) => {{
    const r = item.getBoundingClientRect();
    tooltip.innerHTML = `<strong>${{f.name}}</strong><br>` +
      (f.brand ? `Brand: ${{f.brand}}<br>` : '') +
      `TD: ${{f.td}}mm<br>Hex: ${{f.hex}}<br>Coverage: ${{f.coverage}}%<br>Layer: ${{f.layer}}`;
    tooltip.style.display = 'block';
    tooltip.style.left = (r.right + 8) + 'px';
    tooltip.style.top = r.top + 'px';
  }});
  item.addEventListener('mouseleave', () => {{ tooltip.style.display = 'none'; }});

  // Click to toggle visibility
  item.addEventListener('click', () => {{
    const isHidden = item.classList.toggle('hidden');
    const targetIdx = f.layer - 1;
    for (const entry of layerMeshes) {{
      if (entry.idx === targetIdx) {{
        entry.mesh.visible = !isHidden;
      }}
    }}
  }});

  swatchList.appendChild(item);
}});

// Sidebar toggle
document.getElementById('sidebar-toggle').addEventListener('click', () => {{
  const sidebar = document.getElementById('sidebar');
  const btn = document.getElementById('sidebar-toggle');
  sidebar.classList.toggle('collapsed');
  btn.innerHTML = sidebar.classList.contains('collapsed') ? '&#9654;' : '&#9664;';
}});


(function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}})();
</script>
</body>
</html>'''


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='HueCLI - Generate multi-color 3D print STLs from images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Example: python3 huecli.py image.png --colors 5 --size 120x160'
    )

    parser.add_argument('image', help='Input image path (PNG/JPG/WEBP)')
    parser.add_argument('-f', '--filaments', type=str, default=None,
                        help='Filament library CSV path (default: filaments.csv)')
    parser.add_argument('-c', '--colors', type=int, default=None,
                        help='Number of colors/filaments (default: 4)')
    parser.add_argument('-n', '--nozzle', type=float, default=None,
                        help='Nozzle diameter in mm (default: 0.2)')
    parser.add_argument('-l', '--layer-height', type=float, default=None,
                        help='Layer height in mm (default: 0.08)')
    parser.add_argument('-m', '--model-height', type=float, default=None,
                        help='Total model height in mm (default: 2.0)')
    parser.add_argument('-s', '--size', type=str, default=None,
                        help='Print size as WIDTHxHEIGHT in mm, e.g. 100x140 (default: 100x140)')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output filename (default: <input_stem>.stl)')
    parser.add_argument('-d', '--min-delta-e', type=float, default=None,
                        help='Min color difference delta-E between filaments (default: 5.0)')
    parser.add_argument('--mode', type=str, default=None,
                        choices=['standard', 'flat', 'flat-cap',
                                 'exploded', 'exploded-multi', 'exploded-cmyk'],
                        help='Generation mode (default: standard)')
    parser.add_argument('--sandwich-layers', type=int, default=None,
                        help='Color layers per sandwich in exploded modes (default: 1, CMYK default: 3)')
    parser.add_argument('--fill', type=str, default=None, metavar='BOOL',
                        help='Fill sandwiches with transparent (inverse middle + top layer) [yes/no] (default: no)')
    parser.add_argument('--base-layers', type=int, default=None,
                        help='Transparent base layers per sandwich in exploded modes (default: 3)')
    parser.add_argument('--max-color-sandwiches', type=int, default=None,
                        help='Max sandwiches per colour in exploded modes (default: 3, multi: 5, CMYK: 1)')
    parser.add_argument('--scheme', type=str, default=None,
                        choices=list(COLOR_SCHEMES.keys()),
                        help='Remap image to a colour scheme palette before processing')
    parser.add_argument('--flip', type=str, default=None,
                        choices=['horizontal', 'vertical', 'both'],
                        help='Flip the image before processing (horizontal, vertical, or both)')

    args = parser.parse_args()

    try:
        # Validate image exists
        if not Path(args.image).exists():
            logger.error(f"Image file not found: {args.image}")
            return 1

        print("\nHueCLI - Multi-Color 3D Print STL Generator")
        print(f"Input: {args.image}\n")

        filaments_csv = args.filaments if args.filaments is not None else prompt_with_default(
            "Filament library CSV path",
            "filaments.csv",
            str
        )

        # Validate filaments CSV exists
        if not Path(filaments_csv).exists():
            logger.error(f"Filaments CSV not found: {filaments_csv}")
            return 1

        color_count = args.colors if args.colors is not None else None  # Resolved after mode flags

        nozzle_diameter = args.nozzle if args.nozzle is not None else prompt_with_default(
            "Nozzle diameter (mm)",
            0.2,
            float
        )

        layer_height = args.layer_height if args.layer_height is not None else prompt_with_default(
            "Layer height (mm)",
            0.08,
            float
        )

        model_height = args.model_height if args.model_height is not None else prompt_with_default(
            "Total model height (mm)",
            2.0,
            float
        )

        size_input = args.size if args.size is not None else prompt_with_default(
            "Print size - width x height (mm)",
            "100x140",
            str
        )

        # Parse size input
        try:
            if 'x' in size_input.lower():
                parts = size_input.lower().split('x')
                width = float(parts[0].strip())
                height = float(parts[1].strip()) if len(parts) > 1 else None
            else:
                width = float(size_input)
                height = None
        except ValueError:
            logger.error(f"Invalid size format: {size_input}. Use format like '100x140' or just '100'")
            return 1

        # Generate default output filename from input filename
        input_path = Path(args.image)
        default_output = input_path.stem + ".stl"

        output_name = args.output if args.output is not None else prompt_with_default(
            "Output filename",
            default_output,
            str
        )

        min_color_difference = args.min_delta_e if args.min_delta_e is not None else prompt_with_default(
            "Min color difference (delta-E)",
            5.0,
            float
        )

        if args.mode is not None:
            mode = args.mode
        else:
            print("\nAvailable modes:")
            print("  1. standard        - Multi-color stacked topographical STLs")
            print("  2. flat            - Flat color slabs, multi-color per pixel")
            print("  3. flat-cap        - Flat + transparent cap layer")
            print("  4. exploded        - Each color as standalone transparent sandwich")
            print("  5. exploded-multi  - Up to 3 colors per sandwich, higher fidelity")
            print("  6. exploded-cmyk   - 4 CMYK primaries, max 8 sandwiches")
            mode_choice = input("Select mode [1]: ").strip() or "1"
            mode_map = {
                '1': 'standard', '2': 'flat', '3': 'flat-cap',
                '4': 'exploded', '5': 'exploded-multi',
                '6': 'exploded-cmyk',
            }
            mode = mode_map.get(mode_choice, mode_choice)  # Accept number or name

        # Derive boolean flags from mode
        use_flat = mode == 'flat'
        use_flat_cap = mode == 'flat-cap'
        use_exploded = mode == 'exploded'
        use_exploded_multi = mode == 'exploded-multi'
        use_exploded_cmyk = mode == 'exploded-cmyk'
        exploded_any = use_exploded or use_exploded_multi or use_exploded_cmyk

        # Resolve color_count now that mode flags are known
        if use_exploded_cmyk:
            color_count = 4  # CMYK always uses exactly 4 colors
        elif color_count is None and not exploded_any:
            color_count = prompt_with_default("Number of colors/filaments", 4, int)

        if not exploded_any and use_flat_cap and color_count < 2:
            logger.warning("Flat-cap requires at least 2 colors. Setting to 2.")
            color_count = 2

        # Resolve sandwich_layers (color layers per sandwich in exploded modes)
        if args.sandwich_layers is not None:
            sandwich_layers = args.sandwich_layers
        elif use_exploded_cmyk:
            sandwich_layers = 3  # CMYK needs thicker color layers for high-TD filaments
        else:
            sandwich_layers = 1  # Standard exploded default

        # Resolve fill (transparent fill in exploded sandwiches)
        use_fill = (args.fill.lower() in ('y', 'yes', 'true', '1')) if args.fill is not None else False

        # Resolve base_layers (transparent base layers per sandwich / cap layers in flat-cap mode)
        if args.base_layers is not None:
            base_layers = args.base_layers
        elif use_flat_cap:
            base_layers = 1
        elif exploded_any:
            base_layers = 3
        else:
            base_layers = 2

        # Resolve max_color_sandwiches (max sandwiches per colour in exploded modes)
        if args.max_color_sandwiches is not None:
            max_color_sandwiches = args.max_color_sandwiches
        elif use_exploded_cmyk:
            max_color_sandwiches = 1
        elif use_exploded_multi:
            max_color_sandwiches = 5
        else:
            max_color_sandwiches = 3

        print("\n" + "=" * 60)

        # Pipeline execution
        logger.info("Starting STL generation...")

        # 1. Load filament library
        filament_lib = FilamentLibrary(filaments_csv)

        # 2. Load and prepare image
        # For exploded mode, use placeholder color_count; auto-determine after loading
        img_processor = ImageProcessor(args.image, width, color_count or 4)
        img_processor.load_and_prepare(nozzle_diameter=nozzle_diameter)

        # Apply user-requested flip
        if args.flip in ('horizontal', 'both'):
            img_processor.image = np.fliplr(img_processor.image)
            img_processor.image_lab = np.fliplr(img_processor.image_lab)
            if img_processor.alpha_mask is not None:
                img_processor.alpha_mask = np.fliplr(img_processor.alpha_mask)
            logger.info("Applied horizontal flip")
        if args.flip in ('vertical', 'both'):
            img_processor.image = np.flipud(img_processor.image)
            img_processor.image_lab = np.flipud(img_processor.image_lab)
            if img_processor.alpha_mask is not None:
                img_processor.alpha_mask = np.flipud(img_processor.alpha_mask)
            logger.info("Applied vertical flip")

        # Apply colour scheme remapping
        if args.scheme is not None:
            scheme_name = args.scheme
            palette_hex = COLOR_SCHEMES[scheme_name]
            palette_rgb = np.array([[int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)]
                                    for h in palette_hex]) / 255.0
            palette_lab = color.rgb2lab(palette_rgb.reshape(-1, 1, 3)).reshape(-1, 3)

            # Remap every pixel to the nearest palette colour (in LAB space)
            pixels_lab = img_processor.image_lab.reshape(-1, 3)
            from scipy.spatial import cKDTree
            tree = cKDTree(palette_lab)
            _, indices = tree.query(pixels_lab, k=1)

            # Rebuild the image from palette colours
            remapped_rgb = palette_rgb[indices].reshape(img_processor.image.shape)
            img_processor.image = remapped_rgb
            img_processor.image_lab = color.rgb2lab(remapped_rgb)

            logger.info(f"Remapped image to '{scheme_name}' scheme ({len(palette_hex)} colours)")

            # Default -c to scheme size if not specified
            if color_count is None and not use_exploded_cmyk:
                color_count = len(palette_hex)
                img_processor.color_count = color_count
                logger.info(f"Defaulting colour count to scheme size: {color_count}")

        if exploded_any and not use_exploded_cmyk and color_count is None:
            # Iterative color count optimization: try K=3..max_k, score each by
            # Beer-Lambert mean delta-E, pick lowest K within 2.0 of best score.
            from generator import STLGenerator as _STLGen

            grayscale_tmp = (0.2126 * img_processor.image[:, :, 0] +
                             0.7152 * img_processor.image[:, :, 1] +
                             0.0722 * img_processor.image[:, :, 2])
            if grayscale_tmp.max() > grayscale_tmp.min():
                grayscale_tmp = (grayscale_tmp - grayscale_tmp.min()) / (grayscale_tmp.max() - grayscale_tmp.min())

            total_sandwiches = int(model_height / layer_height)
            MAX_S_PER_COLOR = max_color_sandwiches
            max_try_k = min(total_sandwiches, 12)  # cap search at 12 colors
            mode_name = "exploded-multi" if use_exploded_multi else "exploded"
            logger.info(f"Auto-determining optimal color count for {mode_name} (K=3..{max_try_k}, cap={MAX_S_PER_COLOR})...")
            k_scores = []

            for try_k in range(3, max_try_k + 1):
                img_processor.color_count = try_k
                try:
                    dom_colors, _, _ = img_processor.quantize_colors()
                except Exception:
                    continue

                try_filaments = filament_lib.select_for_exploded(
                    dom_colors,
                    min_color_difference=min_color_difference,
                    layer_height=layer_height,
                    model_height=model_height,
                )

                # Score via Beer-Lambert optimization (no STL generation)
                try_gen = _STLGen(
                    grayscale_tmp, width, layer_height, model_height,
                    try_filaments, alpha_mask=img_processor.alpha_mask,
                    image_rgb=img_processor.image, use_exploded=True,
                )

                num_color_fils = len(try_filaments) - 1
                color_fils = try_filaments.iloc[:num_color_fils]
                trans_fil = try_filaments.iloc[-1]

                caps = [min(MAX_S_PER_COLOR, total_sandwiches) for _ in range(num_color_fils)]

                # Reduce caps if combinatorial space is too large
                combo_size = 1
                for c in caps:
                    combo_size *= (c + 1)
                while combo_size > 500_000:
                    max_i = caps.index(max(caps))
                    caps[max_i] -= 1
                    combo_size = 1
                    for c in caps:
                        combo_size *= (c + 1)

                layer_counts = try_gen._compute_exploded_layer_counts(
                    color_fils, trans_fil, total_sandwiches, caps
                )

                # Compute mean delta-E score
                from itertools import product as _prod
                from scipy.spatial import cKDTree

                fil_tds = np.array([f['transmission_distance'] for _, f in color_fils.iterrows()])
                all_rgbs = np.vstack([
                    np.array([f['rgb'] for _, f in color_fils.iterrows()]),
                    np.array(trans_fil['rgb']).reshape(1, 3)
                ])
                all_tds = np.append(fil_tds, trans_fil['transmission_distance'])

                ranges = [range(0, c + 1) for c in caps]
                all_combos = np.array(list(_prod(*ranges)))
                valid = all_combos.sum(axis=1) <= total_sandwiches
                combos = all_combos[valid]
                trans_l = total_sandwiches - combos.sum(axis=1)
                color_thickness = layer_height * sandwich_layers
                trans_per_sandwich = color_thickness if use_fill else layer_height
                thickness = np.column_stack([combos * color_thickness, (trans_l * trans_per_sandwich).reshape(-1, 1)])
                sim_rgb = try_gen._vectorized_beer_lambert(all_rgbs, all_tds, thickness)
                sim_lab = color.rgb2lab(sim_rgb.reshape(-1, 1, 3)).reshape(-1, 3)
                tree = cKDTree(sim_lab)
                target_lab = color.rgb2lab(img_processor.image).reshape(-1, 3)
                dists, _ = tree.query(target_lab, k=1)
                valid_mask = (img_processor.alpha_mask >= 0.5).ravel()
                mean_de = float(np.mean(dists[valid_mask]))

                fil_names = [f['name'] for _, f in color_fils.iterrows()]
                k_scores.append((try_k, mean_de, fil_names))
                logger.info(f"  K={try_k}: mean deltaE={mean_de:.2f} ({', '.join(fil_names)})")

            # Pick lowest K within 2.0 delta-E of best score
            if k_scores:
                best_score = min(s for _, s, _ in k_scores)
                for try_k, score, names in k_scores:
                    if score <= best_score + 2.0:
                        color_count = try_k
                        logger.info(f"Selected K={color_count} (deltaE={score:.2f}, best={best_score:.2f})")
                        break
            else:
                color_count = 6

            img_processor.color_count = color_count
        elif exploded_any:
            img_processor.color_count = color_count

        # 3. Quantize colors
        dominant_colors_lab, kmeans, sorted_indices = img_processor.quantize_colors()

        # 4. Select best filaments (with transmission-aware color simulation)
        if use_exploded_cmyk:
            selected_filaments = filament_lib.select_for_exploded_cmyk(
                layer_height=layer_height,
                model_height=model_height,
                sandwich_layers=sandwich_layers,
                max_color_sandwiches=max_color_sandwiches,
            )
        elif exploded_any:
            selected_filaments = filament_lib.select_for_exploded(
                dominant_colors_lab,
                min_color_difference=min_color_difference,
                layer_height=layer_height,
                model_height=model_height,
            )
        else:
            selected_filaments = filament_lib.select_best_filaments(
                dominant_colors_lab, color_count,
                layer_height=layer_height,
                min_color_difference=min_color_difference,
                use_flat_cap=use_flat_cap,
                model_height=model_height,
            )

        logger.info("Selected filaments:")
        for i, row in selected_filaments.iterrows():
            logger.info(f"  {i+1}. {row['name']} ({row['color_hex']}) TD={row['transmission_distance']:.1f}mm")

        # 5. Convert to grayscale for brightness-based thickness mapping
        logger.info("Converting to grayscale...")

        grayscale = (0.2126 * img_processor.image[:, :, 0] +
                     0.7152 * img_processor.image[:, :, 1] +
                     0.0722 * img_processor.image[:, :, 2])

        # Normalize to 0-1 range
        if grayscale.max() > grayscale.min():
            grayscale = (grayscale - grayscale.min()) / (grayscale.max() - grayscale.min())
        else:
            logger.warning("WARNING: Image has no brightness variation!")
            grayscale = np.ones_like(grayscale) * 0.5

        logger.info(f"Grayscale range: {grayscale.min():.3f} to {grayscale.max():.3f}")

        # 6. Preview loop — 3D browser preview + simplified menu
        def _make_preview_gen():
            return STLGenerator(
                grayscale, width, layer_height, model_height,
                selected_filaments,
                alpha_mask=img_processor.alpha_mask,
                use_flat=use_flat,
                image_rgb=img_processor.image,
                use_flat_cap=use_flat_cap,
                use_exploded=use_exploded,
                use_exploded_multi=use_exploded_multi,
                use_exploded_cmyk=use_exploded_cmyk,
                sandwich_layers=sandwich_layers,
                use_fill=use_fill,
                base_layers=base_layers,
                max_color_sandwiches=max_color_sandwiches,
            )

        show_3d_preview(_make_preview_gen())

        while True:
            print("\n" + "=" * 60)
            print("3D preview opened in browser. What would you like to do?")
            print("1. Generate STLs")
            print("2. Adjust colors (reduce delta-E, re-preview)")
            print("3. Cancel")
            print("=" * 60)
            choice = input("Enter choice [1]: ").strip() or "1"

            if choice == "3":
                logger.info("Cancelled by user")
                return 0
            elif choice == "2":
                min_color_difference = min_color_difference * 0.7
                logger.info(f"Reducing color delta to {min_color_difference:.1f} and re-selecting filaments...")
                if use_exploded_cmyk:
                    selected_filaments = filament_lib.select_for_exploded_cmyk(
                        layer_height=layer_height,
                        model_height=model_height,
                        sandwich_layers=sandwich_layers,
                        max_color_sandwiches=max_color_sandwiches,
                    )
                elif exploded_any:
                    selected_filaments = filament_lib.select_for_exploded(
                        dominant_colors_lab,
                        min_color_difference=min_color_difference,
                        layer_height=layer_height,
                        model_height=model_height,
                    )
                else:
                    selected_filaments = filament_lib.select_best_filaments(
                        dominant_colors_lab, color_count,
                        layer_height=layer_height,
                        min_color_difference=min_color_difference,
                        use_flat_cap=use_flat_cap,
                        randomize=False,
                        model_height=model_height,
                    )
                logger.info("New filaments selected:")
                for i, row in selected_filaments.iterrows():
                    logger.info(f"  {i+1}. {row['name']} ({row['color_hex']}) - L={row['lab'][0]:.1f}")
                show_3d_preview(_make_preview_gen())
            else:
                break

        logger.info("Generating STLs...")

        # 7. Create output directory
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)

        # Ensure output file is in the output directory
        output_filename = Path(output_name).name
        output_path = output_dir / output_filename

        # 7a. Generate STLs
        logger.info("\nGenerating STLs...")

        stl_gen = STLGenerator(
            grayscale,
            width,
            layer_height,
            model_height,
            selected_filaments,
            alpha_mask=img_processor.alpha_mask,
            use_flat=use_flat,
            image_rgb=img_processor.image,
            use_flat_cap=use_flat_cap,
            use_exploded=use_exploded,
            use_exploded_multi=use_exploded_multi,
            use_exploded_cmyk=use_exploded_cmyk,
            sandwich_layers=sandwich_layers,
            use_fill=use_fill,
            base_layers=base_layers,
            max_color_sandwiches=max_color_sandwiches,
        )

        generated_files = stl_gen.generate_all(output_path)
        logger.info(f"Generated {len(generated_files)} STL file(s)")

        # 8. Generate description file
        desc_path = output_path.with_suffix('.txt')
        with open(desc_path, 'w') as f:
            f.write("HueCLI Multi-Color STL Generation\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Model: {width}mm wide x {model_height}mm tall\n")
            f.write(f"Layer height: {layer_height}mm\n")
            f.write(f"Total layers: {stl_gen.num_layers}\n")
            f.write(f"Colors: {stl_gen.num_colors}\n\n")

            f.write("Generated Files:\n")
            f.write("-" * 60 + "\n")
            for file_path, color_name, layer_start, layer_end in generated_files:
                layer_range_mm = f"{layer_start * layer_height:.2f}-{layer_end * layer_height:.2f}mm"
                f.write(f"{file_path.name}\n")
                f.write(f"  {color_name}: layers {layer_start}-{layer_end} ({layer_range_mm})\n\n")

            # Write filament-change schedule if available
            f.write("\nPrinting Instructions:\n")
            f.write("-" * 60 + "\n")
            if use_exploded_cmyk:
                f.write("EXPLODED CMYK MODE\n")
                f.write("Uses 4 subtractive primaries (Cyan, Magenta, Yellow, Black) + transparent.\n")
                f.write(f"Each colour has up to {max_color_sandwiches} intensity level(s) = max {4 * max_color_sandwiches} sandwiches.\n")
                f.write(f"Each sandwich has {base_layers} base layer(s) + {sandwich_layers} colour layer(s)")
                if use_fill:
                    f.write(f" + 1 top = {base_layers + sandwich_layers + 1} layers.\n\n")
                else:
                    f.write(f" = {base_layers + sandwich_layers} layers.\n\n")
                f.write("For each sandwich:\n")
                f.write("1. Load both STLs (_color.stl and _transparent.stl) into slicer\n")
                f.write("2. Assign the colour filament to the _color.stl part\n")
                f.write("3. Assign transparent filament to the _transparent.stl part\n")
                layers_total = base_layers + sandwich_layers + (1 if use_fill else 0)
                f.write(f"4. Print as a single {layers_total}-layer print\n\n")
                f.write("After printing all sandwiches:\n")
                f.write("5. Stack all sandwiches in order and backlight for effect\n")
            elif use_exploded_multi:
                layers_total = base_layers + sandwich_layers + (1 if use_fill else 0)
                f.write("EXPLODED-MULTI MODE\n")
                f.write(f"Each sandwich has {base_layers} base + {sandwich_layers} colour layer(s) with up to 3 colors")
                if use_fill:
                    f.write(f" + 1 top = {layers_total} layers.\n")
                else:
                    f.write(f" = {layers_total} layers.\n")
                f.write("Requires a multi-material printer (or filament swaps) per sandwich.\n\n")
                f.write("For each sandwich (S01, S02, ...):\n")
                f.write("1. Load the _transparent.stl and all _color.stl files for that sandwich\n")
                f.write("2. Assign each color filament to its _color.stl part\n")
                f.write("3. Assign transparent filament to the _transparent.stl part\n")
                f.write(f"4. Print as a single {layers_total}-layer print\n\n")
                f.write("After printing all sandwiches:\n")
                f.write("5. Stack all sandwiches in order and backlight for effect\n")
            elif use_exploded:
                layers_total = base_layers + sandwich_layers + (1 if use_fill else 0)
                f.write("EXPLODED MODE\n")
                f.write(f"Each color is a standalone sandwich: {base_layers} base + {sandwich_layers} colour")
                if use_fill:
                    f.write(f" + 1 top = {layers_total} layers.\n")
                else:
                    f.write(f" = {layers_total} layers.\n")
                f.write("Print each sandwich separately on any single-material printer.\n\n")
                f.write("For each color:\n")
                f.write("1. Load both STLs (_color.stl and _transparent.stl) into slicer\n")
                f.write("2. Assign the color filament to the _color.stl part\n")
                f.write("3. Assign transparent filament to the _transparent.stl part\n")
                f.write(f"4. Print as a single {layers_total}-layer print\n\n")
                f.write("After printing all sandwiches:\n")
                f.write("5. Stack all sandwiches in order and backlight for effect\n")
            elif use_flat_cap:
                f.write("FLAT-CAP MODE\n")
                f.write("Flat color slabs with multi-color per pixel + transparent cap.\n\n")
                f.write("1. Load all STL files into slicer at once\n")
                f.write("2. Right-click each part and assign the corresponding filament\n")
                f.write("3. Print flat (geometry is already optimized)\n")
                f.write(f"4. Transparent cap is {base_layers} layer(s) on top\n")
                f.write("5. Backlight for best effect\n")
            elif use_flat:
                f.write("FLAT MODE\n")
                f.write("Flat color slabs with multi-color per pixel.\n\n")
                f.write("1. Load all STL files into slicer at once\n")
                f.write("2. Right-click each part and assign the corresponding filament\n")
                f.write("3. Print flat (geometry is already optimized)\n")
                f.write("4. Backlight for best effect\n")
            else:
                f.write("1. Load all STL files into slicer at once\n")
                f.write("2. Right-click each part and assign the corresponding filament\n")
                f.write("3. Print flat (geometry is already optimized)\n")
                f.write("4. Backlight for best effect\n")

        logger.info(f"\nDone! Generated {len(generated_files)} STL files in {output_dir}/")
        for file_path, color_name, layer_start, layer_end in generated_files:
            logger.info(f"  {file_path.name} ({color_name})")
        logger.info(f"  {desc_path.name} (printing instructions)")

        return 0

    except Exception as e:
        logger.error(f"Error during STL generation: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
