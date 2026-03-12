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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

from generator import STLGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
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

    def select_best_filaments(self, target_colors_lab, count, layer_height=0.08, min_color_difference=15.0, randomize=False, use_cap_layers=False, use_face_down=False):
        """
        Select best matching filaments using direct delta-E color matching.

        Each filament's raw color is compared to target colors. The Beer-Lambert
        transmission physics are handled during generation, not selection.

        When use_cap_layers=True, reserves 2 slots for dark base + clear top,
        and uses remaining slots for color matching.

        Args:
            min_color_difference: Minimum delta-E between selected filaments to ensure diversity
            randomize: Add random perturbation to selection to get variation
            use_cap_layers: Reserve slots for dark base layer and clear cap layer
        """
        logger.info(f"Selecting {count} unique filaments from library (randomize: {randomize}, cap_layers: {use_cap_layers}, face_down: {use_face_down})")

        if use_cap_layers and use_face_down:
            # Face-down + cap layers: only reserve transparent, all others for image
            result = self._select_with_face_down_cap(target_colors_lab, count, layer_height, min_color_difference, randomize)
            return result

        if use_cap_layers:
            # Cap layers mode: select dark base + clear top + colors
            result = self._select_with_cap_layers(target_colors_lab, count, layer_height, min_color_difference, randomize)
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

    def _select_with_face_down_cap(self, target_colors_lab, count, layer_height, min_color_difference, randomize):
        """Select filaments for face-down + cap layers mode

        Only reserves 1 slot for transparent (cap + negative fill).
        All remaining slots are for image-carrying colors.

        Returns filaments in order: [color1, color2, ..., transparent]
        """
        logger.info("Face-down + cap layers: selecting transparent + image colors")

        if count < 2:
            raise ValueError("Face-down cap layers requires at least 2 colors (1 image + transparent)")

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

    def select_for_exploded_cmyk(self, layer_height=0.08, model_height=2.0, sandwich_layers=3):
        """Select filaments for exploded CMYK mode: C/M/Y/K primaries + transparent.

        Uses fixed CMYK target colors instead of K-means. Finds the best filament
        match for each ideal primary, with TD penalty for overly transparent filaments.

        Args:
            sandwich_layers: Color layers per sandwich (default 3 for CMYK)

        Returns:
            DataFrame: [cyan, magenta, yellow, black, transparent] (transparent last)
        """
        logger.info("Exploded CMYK: selecting C/M/Y/K filaments + transparent")

        # Max material thickness per color in CMYK mode
        # (2 sandwiches × sandwich_layers color layers × layer_height)
        max_sandwiches_per_color = 2
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

    def _select_with_cap_layers(self, target_colors_lab, count, layer_height, min_color_difference, randomize):
        """Select filaments for cap layers mode: dark base + clear top + colors

        Returns filaments in order: [dark_base, color1, color2, ..., clear_top]
        """
        logger.info("Cap layers mode: selecting dark base + clear top + colors")

        if count < 3:
            logger.error("Cap layers mode requires at least 3 colors (dark + 1 color + clear)")
            raise ValueError("use_cap_layers requires --color-count >= 3")

        selected_indices = []

        # 1. Select DARK base layer (darkest, most opaque filament for text)
        dark_candidates = []
        for idx, row in self.df.iterrows():
            luminosity = row['lab'][0]
            if luminosity < 30:  # Very dark
                dark_candidates.append((luminosity, idx))

        if not dark_candidates:
            logger.warning("No very dark filaments found (L < 30), using darkest available")
            darkest_idx = self.df['lab'].apply(lambda x: x[0]).idxmin()
            selected_indices.append(darkest_idx)
        else:
            dark_candidates.sort()
            selected_indices.append(dark_candidates[0][1])

        dark_filament = self.df.iloc[selected_indices[0]]
        logger.info(f"Selected dark base: {dark_filament['name']} (L={dark_filament['lab'][0]:.1f}, TD={dark_filament['transmission_distance']}mm)")

        # 2. Select PURE TRANSPARENT layer
        transparent_idx = self._select_transparent_filament(exclude_indices=set(selected_indices))
        selected_indices.append(transparent_idx)

        clear_filament = self.df.iloc[selected_indices[-1]]
        logger.info(f"Selected PURE transparent: {clear_filament['name']} (L={clear_filament['lab'][0]:.1f}, TD={clear_filament['transmission_distance']}mm, a={clear_filament['lab'][1]:.1f}, b={clear_filament['lab'][2]:.1f})")

        # 3. Select COLOR filaments (count - 2 remaining slots)
        num_colors = count - 2
        logger.info(f"Selecting {num_colors} color filaments to match image")

        # Track already selected LABs for diversity
        selected_labs = [self.df.iloc[idx]['lab'] for idx in selected_indices]

        # Sort target colors by luminosity
        target_colors_sorted = sorted(enumerate(target_colors_lab), key=lambda x: x[1][0])

        for color_idx, target_lab in target_colors_sorted[:num_colors]:
            distances = []

            for idx, row in self.df.iterrows():
                if idx in selected_indices:
                    continue

                # Check diversity
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

                # Calculate delta-E to target color
                delta_e = color.deltaE_ciede2000(
                    np.array([[target_lab]]),
                    np.array([[filament_lab]])
                )[0][0]

                # Add random perturbation if requested
                if randomize:
                    import random
                    delta_e += random.uniform(-5.0, 5.0)

                distances.append((delta_e, idx))

            if distances:
                distances.sort()
                best_idx = distances[0][1]
                selected_indices.append(best_idx)
                selected_labs.append(self.df.iloc[best_idx]['lab'])
                logger.info(f"  Color {len(selected_indices)-2}: {self.df.iloc[best_idx]['name']} ({self.df.iloc[best_idx]['color_hex']})")

        if len(selected_indices) < count:
            logger.warning(f"Only selected {len(selected_indices)} filaments out of {count} requested")

        # Reorder: [dark, colors..., clear] but we want dark at index 0 and clear at the end
        # Currently we have: [dark, clear, colors...]
        # Need to move clear to end
        result_indices = [selected_indices[0]]  # dark
        result_indices.extend(selected_indices[2:])  # colors
        result_indices.append(selected_indices[1])  # clear

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

    def load_and_prepare(self, nozzle_diameter=0.2, face_down=False):
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

            # Face-down mode: horizontal flip so text reads correctly when print is flipped over
            if face_down:
                self.image = np.fliplr(self.image)
                self.alpha_mask = np.fliplr(self.alpha_mask)
                logger.info("Applied horizontal flip for face-down printing")

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

    def _apply_selective_smoothing(self, image):
        """Apply smoothing to gradients while preserving text and sharp details

        Uses edge detection to identify text/details and applies smoothing only to
        smooth gradient areas. This reduces color banding while keeping text crisp.
        """
        logger.info("Applying selective smoothing (preserving text and sharp details)...")

        try:
            from scipy import ndimage

            # Convert to grayscale for edge detection
            grayscale = np.mean(image, axis=2)

            # Detect edges using Sobel operator
            # High gradient = text/edges, low gradient = smooth areas
            gradient_x = ndimage.sobel(grayscale, axis=1)
            gradient_y = ndimage.sobel(grayscale, axis=0)
            gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

            # Normalize gradient to 0-1
            gradient_magnitude = (gradient_magnitude - gradient_magnitude.min()) / (gradient_magnitude.max() - gradient_magnitude.min() + 1e-8)

            # Create edge mask: high gradient = edge (preserve), low gradient = smooth area
            edge_threshold = 0.08  # VERY sensitive to preserve text and sharp features
            edge_mask = gradient_magnitude > edge_threshold

            # Dilate edge mask to protect areas around edges (prevents halos around text)
            edge_mask_dilated = ndimage.binary_dilation(edge_mask, iterations=4)

            logger.info(f"Preserving edges: {edge_mask_dilated.sum() / edge_mask_dilated.size * 100:.1f}% of image")

            # Apply LIGHT Gaussian smoothing only to pure gradients (not near any edges)
            smoothed = np.zeros_like(image)
            for channel in range(3):
                # Light blur - just enough to reduce noise, not enough to blur detail
                smoothed[:, :, channel] = ndimage.gaussian_filter(image[:, :, channel], sigma=0.8)

            # Blend: use original at edges, smoothed in gradient areas
            result = np.zeros_like(image)
            for channel in range(3):
                result[:, :, channel] = np.where(
                    edge_mask_dilated,
                    image[:, :, channel],  # Keep original at edges (100% sharp)
                    smoothed[:, :, channel]  # Use smoothed in gradients
                )

            logger.info("Selective smoothing complete - text and details preserved")
            return result

        except ImportError:
            logger.warning("scipy not available - skipping selective smoothing")
            return image
        except Exception as e:
            logger.warning(f"Selective smoothing failed: {e} - using original image")
            return image

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


class TransmissionColorSimulator:
    """Simulates color appearance based on light transmission through layered filaments"""

    def __init__(self, selected_filaments, layer_height):
        self.selected_filaments = selected_filaments
        self.layer_height = layer_height

    def calculate_transmission_factor(self, td_value, thickness_mm):
        """
        Calculate how much light passes through a layer using Beer-Lambert law approximation
        TD = transmission distance (mm of material that transmits ~37% of light)
        """
        if td_value < 0.1:  # Completely opaque
            return 0.0
        # Exponential decay: transmission = e^(-thickness / TD)
        return np.exp(-thickness_mm / max(td_value, 0.1))

    def simulate_layer_stack_color(self, layer_indices, layer_heights, subtractive=False):
        """
        Simulate the perceived color of a stack of layers
        layer_indices: list of filament indices from bottom to top
        layer_heights: thickness of each layer in mm
        subtractive: If True, use multiplicative filter model (light only decreases).
                     Required for cap-layer mode where opaque base + additive model
                     causes inversion.
        Returns: RGB color (0-1 range)
        """
        # Start with white light from below (backlit assumption)
        light_rgb = np.array([1.0, 1.0, 1.0])

        # Process each layer from bottom to top
        for layer_idx, thickness in zip(layer_indices, layer_heights):
            filament = self.selected_filaments.iloc[layer_idx]
            filament_rgb = np.array(filament['rgb'])
            td = filament['transmission_distance']

            # Calculate transmission factor
            transmission = self.calculate_transmission_factor(td, thickness)

            if subtractive:
                # Subtractive filter: filament color acts as per-channel multiplier
                # on the absorbed fraction. Light can only decrease.
                # Clear (RGB≈1): filter≈1, no absorption. Black (RGB≈0): filter=t, pure absorption.
                # Colored: selective wavelength absorption (e.g. red filter passes red, blocks blue)
                light_rgb = light_rgb * (transmission + filament_rgb * (1 - transmission))
            else:
                # Additive model: absorbed fraction replaced with filament color
                light_rgb = light_rgb * transmission + filament_rgb * (1 - transmission)

            # Clamp to [0, 1]
            light_rgb = np.clip(light_rgb, 0, 1)

        return light_rgb

    def calculate_perceived_color_at_height(self, heightmap_layers, filament_assignments, y, x):
        """
        Calculate the perceived color at a specific pixel considering all layers below it
        """
        pixel_layer = heightmap_layers[y, x]

        # Build layer stack from bottom (layer 0) to current height
        layer_stack = []
        layer_thicknesses = []

        for layer_num in range(pixel_layer + 1):
            # Find which filament is assigned to this layer
            filament_idx = filament_assignments[layer_num]
            layer_stack.append(filament_idx)
            layer_thicknesses.append(self.layer_height)

        if not layer_stack:
            return np.array([1.0, 1.0, 1.0])  # White if no layers

        return self.simulate_layer_stack_color(layer_stack, layer_thicknesses)


class PreviewGenerator:
    """Generates preview images of layer assignments"""

    def __init__(self, heightmap_layers, selected_filaments, layer_height, layer_assignments=None, use_cap_layers=False, flat=False):
        self.heightmap_layers = heightmap_layers
        self.selected_filaments = selected_filaments
        self.layer_height = layer_height
        self.num_layers = heightmap_layers.max() + 1
        self.layer_assignments = layer_assignments  # 3D array: [height, width, layer]
        self.use_cap_layers = use_cap_layers
        self.flat = flat

    def generate(self, output_path=None, use_transmission_preview=True, interactive=False):
        """Create preview PNG showing layer bands with transmission simulation

        Args:
            output_path: Path to save PNG (optional if interactive=True)
            use_transmission_preview: Whether to simulate transmission colors
            interactive: If True, display interactive window instead of saving
        """
        if output_path:
            logger.info(f"Generating preview image to {output_path}")
        else:
            logger.info("Generating interactive preview...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Left: layer visualization on image
        colored_preview = np.zeros((*self.heightmap_layers.shape, 3))

        if self.layer_assignments is not None and use_transmission_preview:
            # TD-based transmission mode: use actual layer-by-layer assignments
            logger.info("Rendering TD-based transmission preview (VECTORIZED)...")
            simulator = TransmissionColorSimulator(self.selected_filaments, self.layer_height)

            height_px, width_px = self.heightmap_layers.shape

            # OPTIMIZATION: Build unique layer stacks and compute colors only once
            # Then map back to pixels - this is MUCH faster than per-pixel computation
            logger.info("Building unique layer stack catalog...")

            # Create a hashable representation of each pixel's layer stack
            stack_to_color = {}  # Maps stack tuple -> RGB color
            pixel_stack_map = {}  # Maps (y, x) -> stack tuple

            for y in tqdm(range(height_px), desc="Cataloging stacks", leave=False):
                for x in range(width_px):
                    # Determine number of layers for this pixel
                    if self.flat:
                        num_pixel_layers = self.num_layers
                    else:
                        num_pixel_layers = self.heightmap_layers[y, x] + 1

                    # Build layer stack tuple (hashable)
                    layer_stack = tuple(
                        self.layer_assignments[y, x, layer_num]
                        for layer_num in range(num_pixel_layers)
                    )

                    pixel_stack_map[(y, x)] = layer_stack

                    # Track unique stacks
                    if layer_stack not in stack_to_color:
                        stack_to_color[layer_stack] = None  # Placeholder

            # Compute transmission color for each UNIQUE stack
            logger.info(f"Computing colors for {len(stack_to_color)} unique layer stacks (was {height_px * width_px} pixels)...")
            for layer_stack in tqdm(stack_to_color.keys(), desc="Computing colors", leave=False):
                layer_heights = [self.layer_height] * len(layer_stack)
                color_rgb = simulator.simulate_layer_stack_color(list(layer_stack), layer_heights)
                stack_to_color[layer_stack] = color_rgb

            # Map computed colors back to pixels (FAST)
            logger.info("Mapping colors to pixels...")
            for y in range(height_px):
                for x in range(width_px):
                    layer_stack = pixel_stack_map[(y, x)]
                    colored_preview[y, x] = stack_to_color[layer_stack]

            if self.flat:
                ax1.set_title('Flat Transmission Preview (All Layers)')
            else:
                ax1.set_title('TD-Based Transmission Preview (Heightmap)')
        else:
            # Standard mode: VECTORIZED transmission simulation
            simulator = TransmissionColorSimulator(self.selected_filaments, self.layer_height)

            logger.info("Simulating transmission colors for preview (vectorized)...")

            if self.use_cap_layers:
                # CAP LAYERS MODE: dark base + colors + clear top
                num_filaments = len(self.selected_filaments)
                num_color_filaments = num_filaments - 2  # Exclude dark base and clear cap

                logger.info(f"Cap layers mode: dark base + {num_color_filaments} colors + clear top")

                # Create layer-to-filament assignment
                # Layer 0-1: dark base (filament 0)
                # Layer 2 to num_layers-2: colors (filaments 1 to n-2)
                # Layer num_layers-1: clear cap (filament n-1)
                layer_assignments = {}

                # Dark base layers (first 2 layers)
                layer_assignments[0] = 0
                if self.num_layers > 1:
                    layer_assignments[1] = 0

                # Clear cap layer (top layer)
                if self.num_layers > 2:
                    layer_assignments[self.num_layers - 1] = num_filaments - 1

                # Color layers in between
                if self.num_layers > 3 and num_color_filaments > 0:
                    for i in range(2, self.num_layers - 1):
                        # Map to color filaments (indices 1 to num_filaments-2)
                        color_layer_num = i - 2
                        total_color_layers = self.num_layers - 3
                        if total_color_layers > 0:
                            filament_idx = int((color_layer_num / total_color_layers) * num_color_filaments) + 1
                            filament_idx = min(filament_idx, num_filaments - 2)
                        else:
                            filament_idx = 1
                        layer_assignments[i] = filament_idx

            else:
                # STANDARD MODE: distribute filaments evenly
                colors_per_layer = len(self.selected_filaments)
                layer_assignments = {}
                for i in range(self.num_layers):
                    filament_idx = int((i / self.num_layers) * colors_per_layer)
                    filament_idx = min(filament_idx, colors_per_layer - 1)
                    layer_assignments[i] = filament_idx

            if use_transmission_preview:
                # VECTORIZED: Pre-compute color for each unique layer height
                unique_layers = np.unique(self.heightmap_layers)
                layer_colors = {}

                logger.info(f"Computing transmission colors for {len(unique_layers)} unique layer heights...")

                for current_layer in tqdm(unique_layers, desc="Computing layer colors", leave=False):
                    # Build layer stack from bottom to this height
                    layer_stack = []
                    for layer_num in range(int(current_layer) + 1):
                        filament_idx = layer_assignments[layer_num]
                        layer_stack.append(filament_idx)

                    layer_heights = [self.layer_height] * len(layer_stack)
                    color_rgb = simulator.simulate_layer_stack_color(layer_stack, layer_heights)
                    layer_colors[int(current_layer)] = color_rgb

                # VECTORIZED: Map pre-computed colors to all pixels based on their layer height
                logger.info("Mapping colors to pixels (vectorized)...")
                for layer_val, color in layer_colors.items():
                    mask = self.heightmap_layers == layer_val
                    colored_preview[mask] = color

                logger.info("Vectorized transmission preview complete")
                if self.use_cap_layers:
                    ax1.set_title('Cap Layers Preview (Dark Base + Colors + Clear Top)')
                else:
                    ax1.set_title('Transmission-Simulated Color Preview')
            else:
                # Simple flat color assignment (old behavior)
                colors_per_layer = len(self.selected_filaments)
                for i, filament in self.selected_filaments.iterrows():
                    layer_start = int((i / colors_per_layer) * self.num_layers)
                    layer_end = int(((i + 1) / colors_per_layer) * self.num_layers)

                    mask = (self.heightmap_layers >= layer_start) & (self.heightmap_layers < layer_end)
                    colored_preview[mask] = filament['rgb']
                ax1.set_title('Layer Color Preview')

        # Flip preview vertically only to match image orientation
        colored_preview_display = np.flipud(colored_preview)
        ax1.imshow(colored_preview_display)
        ax1.axis('off')

        # Right: filament summary
        filament_names = []
        filament_colors = []

        for i, filament in self.selected_filaments.iterrows():
            name = filament['name']
            td = filament['transmission_distance']
            filament_names.append(f"{name}\n(TD: {td}mm)")
            filament_colors.append(filament['rgb'])

        # Create bar chart showing filament order
        ax2.barh(range(len(filament_names)), [1] * len(filament_names),
                color=filament_colors, edgecolor='black', linewidth=2)
        ax2.set_yticks(range(len(filament_names)))
        ax2.set_yticklabels(filament_names, fontsize=9)
        ax2.set_xlabel('Filament')
        ax2.set_xlim(0, 1)
        ax2.set_xticks([])

        if self.use_cap_layers:
            ax2.set_title('Filaments (Bottom → Top)\nDark Base → Colors → Clear Cap')
        else:
            ax2.set_title('Filaments (Bottom → Top)')

        plt.tight_layout()

        if interactive:
            # Interactive mode: display non-blocking window
            logger.info("Displaying preview window...")
            plt.show(block=False)
            # Return figure handle so caller can close it later
            return fig
        elif output_path:
            # Save mode: export to file
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info("Preview generated successfully")
        else:
            plt.close()


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



def _enhance_contrast(brightness, alpha_mask, strength=2.0):
    """Apply contrast enhancement matching STLGenerator._apply_contrast_enhancement"""
    mean_brightness = np.mean(brightness[alpha_mask >= 0.5]) if np.any(alpha_mask >= 0.5) else 0.5

    if mean_brightness > 0.65:
        gamma = 1.5
        enhanced = np.power(brightness, gamma)
        center = np.mean(enhanced)
        s = 4.0
        enhanced = 1.0 / (1.0 + np.exp(-s * (enhanced - center)))
        enhanced = (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min() + 1e-8)
    elif mean_brightness < 0.35:
        enhanced = np.power(brightness, 0.7)
    else:
        enhanced = 1.0 / (1.0 + np.exp(-strength * (brightness - 0.5)))
        enhanced = (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min() + 1e-8)

    return enhanced


def generate_preview(grayscale, selected_filaments, layer_height, model_height, use_cap_layers, alpha_mask,
                     image_rgb=None, use_exploded=False, exploded_max_cap=3, sandwich_layers=1, use_fill=False):
    """Generate preview image showing simulated color stacking appearance."""
    height_px, width_px = grayscale.shape
    preview = np.ones((height_px, width_px, 3))

    num_colors = len(selected_filaments)
    num_layers = int(model_height / layer_height)

    if use_exploded and image_rgb is not None:
        # Exploded mode preview: Beer-Lambert simulation of optimal layer assignments
        from itertools import product as iter_product

        num_color_filaments = num_colors - 1
        if num_color_filaments <= 0:
            return preview

        total_sandwiches = num_layers

        # Filament properties (colors + transparent)
        color_rgbs = np.array([selected_filaments.iloc[i]['rgb'] for i in range(num_color_filaments)])
        color_tds = np.array([selected_filaments.iloc[i]['transmission_distance'] for i in range(num_color_filaments)])
        trans_rgb = np.array(selected_filaments.iloc[-1]['rgb'])
        trans_td = selected_filaments.iloc[-1]['transmission_distance']

        all_rgbs = np.vstack([color_rgbs, trans_rgb.reshape(1, 3)])
        all_tds = np.append(color_tds, trans_td)

        # Cap each color (matches _generate_exploded / _generate_exploded_multi)
        per_color_caps = [min(exploded_max_cap, total_sandwiches) for _ in range(num_color_filaments)]

        # Reduce caps if combo space too large
        combo_size = 1
        for cap in per_color_caps:
            combo_size *= (cap + 1)
        while combo_size > 500_000:
            max_idx = per_color_caps.index(max(per_color_caps))
            per_color_caps[max_idx] -= 1
            combo_size = 1
            for cap in per_color_caps:
                combo_size *= (cap + 1)

        # Generate valid combos with per-color caps
        ranges = [range(0, cap + 1) for cap in per_color_caps]
        all_combos = np.array(list(iter_product(*ranges)))
        valid = all_combos.sum(axis=1) <= total_sandwiches
        combos = all_combos[valid]

        # Thicknesses including transparent fill
        # Each sandwich has sandwich_layers color layers in the middle
        color_thickness = layer_height * sandwich_layers
        trans_layers = total_sandwiches - combos.sum(axis=1)
        # With fill: unused sandwiches are fully transparent (same thickness as color)
        # Without fill: unused sandwiches only have bottom transparent layer
        trans_per_sandwich = color_thickness if use_fill else layer_height
        thickness_combos = np.column_stack([
            combos * color_thickness,
            (trans_layers * trans_per_sandwich).reshape(-1, 1)
        ])

        # Beer-Lambert simulation
        n_combos = len(thickness_combos)
        light = np.ones((n_combos, 3))
        for k in range(len(all_tds)):
            thickness = thickness_combos[:, k]
            td = max(all_tds[k], 0.1)
            rgb = all_rgbs[k]
            transmission = np.exp(-thickness / td)[:, None]
            light = light * transmission + rgb[None, :] * (1.0 - transmission)
            light = np.clip(light, 0, 1)

        # Convert to LAB and build KDTree
        from scipy.spatial import cKDTree
        combo_lab = color.rgb2lab(light.reshape(-1, 1, 3)).reshape(-1, 3)
        tree = cKDTree(combo_lab)

        # Match each pixel
        target_lab = color.rgb2lab(image_rgb).reshape(-1, 3)
        _, indices = tree.query(target_lab, k=1)

        # Preview shows the simulated Beer-Lambert color
        preview = light[indices].reshape(height_px, width_px, 3)
        preview[alpha_mask < 0.5] = [1, 1, 1]

        return preview

    if use_cap_layers:
        # Cap layers mode: Beer-Lambert simulation matching STL generation
        num_middle_colors = num_colors - 2

        # Apply contrast enhancement to match STL generation
        enhanced = _enhance_contrast(grayscale, alpha_mask)

        # Sort middle filaments by luminosity (darkest first) to match _generate_with_cap_layers
        sorted_fils = selected_filaments.copy()
        sorted_fils['luminosity'] = sorted_fils['lab'].apply(
            lambda x: float(np.asarray(x).flat[0]))
        middle = sorted_fils.iloc[1:-1].sort_values('luminosity').reset_index(drop=True)
        sorted_fils = pd.concat([
            sorted_fils.iloc[:1],
            middle,
            sorted_fils.iloc[-1:]
        ]).reset_index(drop=True)

        simulator = TransmissionColorSimulator(sorted_fils, layer_height)

        # Layer allocation matching _generate_with_cap_layers
        base_layers = 2
        top_layers = 2
        middle_layers = num_layers - base_layers - top_layers
        layers_per_color = middle_layers // num_middle_colors if num_middle_colors > 0 else 0

        # Precompute Beer-Lambert color for 256 brightness levels
        level_colors = np.zeros((256, 3))
        for bi in range(256):
            brightness = bi / 255.0
            layer_stack = [0] * base_layers  # dark base
            for i in range(num_middle_colors):
                # Lightest middle → highest threshold (widest coverage)
                # Darkest middle → lowest threshold (only darkest pixels)
                threshold = (i + 1) / (num_middle_colors + 1)
                filament_idx = i + 1 if brightness <= threshold else num_colors - 1
                layer_stack.extend([filament_idx] * layers_per_color)
            layer_stack.extend([num_colors - 1] * top_layers)  # clear top
            layer_heights_list = [layer_height] * len(layer_stack)
            level_colors[bi] = simulator.simulate_layer_stack_color(layer_stack, layer_heights_list, subtractive=True)

        # Normalize range — Beer-Lambert through many layers compresses dynamic range
        # into a narrow band (e.g. 0.6-0.9), making everything look grey.
        # Stretch to full [0,1] to restore visible contrast.
        min_val = level_colors.min()
        max_val = level_colors.max()
        if max_val > min_val:
            level_colors = (level_colors - min_val) / (max_val - min_val)

        # Map pixels to precomputed colors by contrast-enhanced brightness
        for y in range(height_px):
            for x in range(width_px):
                if alpha_mask[y, x] < 0.5:
                    preview[y, x] = [1, 1, 1]
                    continue
                bi = int(np.clip(enhanced[y, x] * 255, 0, 255))
                preview[y, x] = level_colors[bi]

    else:
        # Standard mode: shaped color layers
        for y in range(height_px):
            for x in range(width_px):
                if alpha_mask[y, x] < 0.5:
                    preview[y, x] = [1, 1, 1]  # White for transparent areas
                    continue

                brightness = grayscale[y, x]

                # Determine which layers this pixel gets
                color_stack = []
                for i in range(num_colors):
                    threshold = i / num_colors
                    if brightness >= threshold:
                        color_stack.append(selected_filaments.iloc[i]['color_hex'])

                # Simulate color mixing
                final_color = simulate_color_stack(color_stack)
                preview[y, x] = final_color

    return preview


def simulate_color_stack(color_hex_list):
    """Simulate appearance of stacked colored layers

    Simple simulation: average the RGB values of all layers
    More sophisticated would use transmission distance, but this is good enough for preview
    """
    if not color_hex_list:
        return [1, 1, 1]  # White if no colors

    rgb_values = []
    for hex_color in color_hex_list:
        # Convert hex to RGB (0-1 range)
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        rgb_values.append([r, g, b])

    # Simple average (could be weighted by transmission distance for accuracy)
    return np.mean(rgb_values, axis=0)


def show_3d_preview(stl_gen):
    """Generate and display an interactive 3D preview in the default browser

    Exports the scene as GLB, embeds it as base64 in a custom HTML viewer
    that uses Three.js (CDN) with GLTFLoader.parse() to avoid data-URL
    size limits that cause blank pages in some browsers.

    Args:
        stl_gen: STLGenerator instance with all parameters configured
    """
    import base64
    import tempfile
    import webbrowser

    logger.info("Generating 3D preview scene...")
    scene = stl_gen.generate_preview_scene()

    if len(scene.geometry) == 0:
        logger.warning("No geometry generated for 3D preview")
        return

    logger.info("Exporting scene to GLB...")
    glb_data = scene.export(file_type="glb")
    b64 = base64.b64encode(glb_data).decode("utf-8")
    logger.info(f"GLB size: {len(glb_data)/1024/1024:.1f} MB")

    use_transparency = stl_gen.use_exploded or stl_gen.use_exploded_multi or stl_gen.use_exploded_cmyk
    html = _build_viewer_html(b64, use_transparency=use_transparency)

    with tempfile.NamedTemporaryFile('w', suffix='.html', delete=False) as f:
        f.write(html)
        tmp_path = f.name

    logger.info("Opening 3D preview in browser...")
    webbrowser.open('file://' + tmp_path)


def _build_viewer_html(b64_glb, use_transparency=False):
    """Build a self-contained HTML page with an embedded Three.js GLB viewer.

    Uses CDN-loaded Three.js with GLTFLoader.parse() to decode the base64
    GLB data directly into an ArrayBuffer, bypassing data-URL size limits.

    Args:
        b64_glb: base64-encoded GLB binary string
        use_transparency: if True, enable transparent materials with vertex alpha

    Returns:
        Complete HTML string
    """
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
    position: absolute; top: 10px; left: 10px; color: #ccc;
    font: 13px/1.4 system-ui, sans-serif; pointer-events: none;
  }}
  #error {{
    position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%);
    color: #f44; font: 16px system-ui; text-align: center; display: none;
  }}
  #controls {{
    position: absolute; bottom: 20px; left: 50%; transform: translateX(-50%);
    display: {'flex' if use_transparency else 'none'}; align-items: center; gap: 12px;
    background: rgba(0,0,0,0.6); padding: 8px 16px; border-radius: 8px;
  }}
  #controls label {{ color: #ccc; font: 13px system-ui; white-space: nowrap; }}
  #gap-slider {{ width: 200px; cursor: pointer; }}
  #gap-value {{ color: #fff; font: 13px monospace; min-width: 40px; }}
</style>
</head>
<body>
<div id="info">Drag to rotate &middot; Scroll to zoom &middot; Right-drag to pan</div>
<div id="error"></div>
<div id="controls">
  <label for="gap-slider">Gap:</label>
  <input type="range" id="gap-slider" min="0" max="5" value="2" step="0.1">
  <span id="gap-value">2.0mm</span>
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
import {{ GLTFLoader }} from 'three/addons/loaders/GLTFLoader.js';

const base64 = "{b64_glb}";

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

// Track sandwich meshes for gap slider repositioning
const sandwichMeshes = [];  // {{mesh, sandwichIndex}}

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

      child.material = new THREE.MeshBasicMaterial({{
        vertexColors: true,
        transparent: hasAlpha,
        opacity: opacity,
        depthWrite: !hasAlpha,
        side: hasAlpha ? THREE.DoubleSide : THREE.FrontSide,
      }});

      // Parse sandwich index from mesh name (e.g. "S01_Black_L1" → 0)
      if (useTransparency) {{
        const m = child.name.match(/^S(\\d+)/);
        if (m) {{
          sandwichMeshes.push({{ mesh: child, idx: parseInt(m[1], 10) - 1 }});
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

// Gap slider: reposition sandwich meshes along Z
function updateGap() {{
  const slider = document.getElementById('gap-slider');
  const label = document.getElementById('gap-value');
  if (!slider) return;
  const gap = parseFloat(slider.value);
  label.textContent = gap.toFixed(1) + 'mm';
  for (const {{ mesh, idx }} of sandwichMeshes) {{
    mesh.position.z = idx * gap;
  }}
}}

const slider = document.getElementById('gap-slider');
if (slider) {{
  slider.addEventListener('input', () => {{
    updateGap();
    // Re-center camera target on the midpoint of the stack
    if (sandwichMeshes.length > 0) {{
      const maxIdx = Math.max(...sandwichMeshes.map(s => s.idx));
      const gap = parseFloat(slider.value);
      const midZ = (maxIdx * gap) / 2;
      controls.target.z = midZ;
    }}
  }});
}}

window.addEventListener('resize', () => {{
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
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
                        choices=['standard', 'cap-layers', 'face-down', 'face-down-cap',
                                 'exploded', 'exploded-multi', 'exploded-cmyk'],
                        help='Generation mode (default: standard)')
    parser.add_argument('--sandwich-layers', type=int, default=None,
                        help='Color layers per sandwich in exploded modes (default: 1, CMYK default: 3)')
    parser.add_argument('--fill', type=str, default=None, metavar='BOOL',
                        help='Fill sandwiches with transparent (inverse middle + top layer) [yes/no] (default: no)')
    parser.add_argument('--base-layers', type=int, default=None,
                        help='Transparent base layers per sandwich in exploded modes (default: 3)')

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
            print("  2. cap-layers      - Black base + auto colors + clear top")
            print("  3. face-down       - Flat voxel grid, flip after printing")
            print("  4. face-down-cap   - Face-down with cap layers")
            print("  5. exploded        - Each color as standalone transparent sandwich")
            print("  6. exploded-multi  - Up to 3 colors per sandwich, higher fidelity")
            print("  7. exploded-cmyk   - 4 CMYK primaries, max 8 sandwiches")
            mode_choice = input("Select mode [1]: ").strip() or "1"
            mode_map = {
                '1': 'standard', '2': 'cap-layers', '3': 'face-down',
                '4': 'face-down-cap', '5': 'exploded', '6': 'exploded-multi',
                '7': 'exploded-cmyk',
            }
            mode = mode_map.get(mode_choice, mode_choice)  # Accept number or name

        # Derive boolean flags from mode
        use_cap_layers = mode in ('cap-layers', 'face-down-cap')
        use_face_down = mode in ('face-down', 'face-down-cap')
        use_exploded = mode == 'exploded'
        use_exploded_multi = mode == 'exploded-multi'
        use_exploded_cmyk = mode == 'exploded-cmyk'
        exploded_any = use_exploded or use_exploded_multi or use_exploded_cmyk

        # Resolve color_count now that mode flags are known
        if use_exploded_cmyk:
            color_count = 4  # CMYK always uses exactly 4 colors
        elif color_count is None and not exploded_any:
            color_count = prompt_with_default("Number of colors/filaments", 4, int)

        if not exploded_any and use_cap_layers and color_count < 3:
            logger.warning("Cap layers requires at least 3 colors. Setting to 3.")
            color_count = 3

        # Resolve sandwich_layers (color layers per sandwich in exploded modes)
        if args.sandwich_layers is not None:
            sandwich_layers = args.sandwich_layers
        elif use_exploded_cmyk:
            sandwich_layers = 3  # CMYK needs thicker color layers for high-TD filaments
        else:
            sandwich_layers = 1  # Standard exploded default

        # Resolve fill (transparent fill in exploded sandwiches)
        use_fill = (args.fill.lower() in ('y', 'yes', 'true', '1')) if args.fill is not None else False

        # Resolve base_layers (transparent base layers per sandwich)
        base_layers = args.base_layers if args.base_layers is not None else 3

        print("\n" + "=" * 60)

        # Pipeline execution
        logger.info("Starting STL generation...")

        # 1. Load filament library
        filament_lib = FilamentLibrary(filaments_csv)

        # 2. Load and prepare image
        # For exploded mode, use placeholder color_count; auto-determine after loading
        img_processor = ImageProcessor(args.image, width, color_count or 4)
        img_processor.load_and_prepare(nozzle_diameter=nozzle_diameter, face_down=use_face_down)

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
            MAX_S_PER_COLOR = 5 if use_exploded_multi else 3
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
            )
        elif exploded_any:
            selected_filaments = filament_lib.select_for_exploded(
                dominant_colors_lab,
                min_color_difference=min_color_difference,
                layer_height=layer_height,
                model_height=model_height,
            )
        else:
            # Cap layers: black base + auto colors + clear top
            selected_filaments = filament_lib.select_best_filaments(
                dominant_colors_lab, color_count,
                layer_height=layer_height,
                min_color_difference=min_color_difference,
                use_cap_layers=use_cap_layers,
                use_face_down=use_face_down
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

        # 6. Preview loop - show preview and get user approval
        logger.info("\nGenerating preview...")

        preview_approved = False
        preview_img = generate_preview(
            grayscale,
            selected_filaments,
            layer_height,
            model_height,
            use_cap_layers,
            img_processor.alpha_mask,
            image_rgb=img_processor.image if exploded_any else None,
            use_exploded=exploded_any,
            exploded_max_cap=2 if use_exploded_cmyk else (5 if use_exploded_multi else 3),
            sandwich_layers=sandwich_layers,
            use_fill=use_fill,
        )
        while not preview_approved:

            # Show preview in non-blocking mode
            import matplotlib.pyplot as plt
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

            # Left: Original image
            # During load_and_prepare, image was flipud for STL orientation
            # Flip it back for display to show original orientation
            original_display = np.flipud(img_processor.image)
            if use_face_down:
                original_display = np.fliplr(original_display)
            ax1.imshow(original_display)
            ax1.set_title('Original Image', fontsize=14)
            ax1.axis('off')

            # Right: Stacking preview
            preview_display = np.flipud(preview_img)
            if use_face_down:
                preview_display = np.fliplr(preview_display)
            ax2.imshow(preview_display)
            ax2.set_title('Color Stacking Preview', fontsize=14)
            ax2.axis('off')

            # Add filament legend
            filament_text = "Filaments:\n"
            for i, row in selected_filaments.iterrows():
                filament_text += f"{i+1}. {row['name']}\n"
            fig.text(0.98, 0.02, filament_text, fontsize=10, ha='right', va='bottom',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            plt.tight_layout()
            plt.show(block=False)  # Non-blocking so CLI prompt appears
            plt.pause(0.1)  # Brief pause to ensure window renders

            # Ask user to proceed (prompt appears in CLI while window is open)
            print("\n" + "=" * 60)
            print("Preview window displayed. What would you like to do?")
            print("1. Generate STLs")
            print("2. Regenerate with lower color delta")
            print("3. View 3D preview (opens in browser)")
            print("4. Cancel")
            print("=" * 60)
            choice = input("Enter choice [1]: ").strip() or "1"

            plt.close(fig)

            if choice == "4":
                logger.info("Cancelled by user")
                return 0
            elif choice == "2":
                # Reduce color delta to allow more similar colors
                min_color_difference = min_color_difference * 0.7
                logger.info(f"Reducing color delta to {min_color_difference:.1f} and re-selecting filaments...")
                if use_exploded_cmyk:
                    selected_filaments = filament_lib.select_for_exploded_cmyk(
                        layer_height=layer_height,
                        model_height=model_height,
                        sandwich_layers=sandwich_layers,
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
                        use_cap_layers=use_cap_layers,
                        use_face_down=use_face_down,
                        randomize=False
                    )
                logger.info("New filaments selected (dark to light):")
                for i, row in selected_filaments.iterrows():
                    logger.info(f"  {i+1}. {row['name']} ({row['color_hex']}) - L={row['lab'][0]:.1f}")
                preview_img = generate_preview(
                    grayscale,
                    selected_filaments,
                    layer_height,
                    model_height,
                    use_cap_layers,
                    img_processor.alpha_mask,
                    image_rgb=img_processor.image if exploded_any else None,
                    use_exploded=exploded_any,
                    exploded_max_cap=2 if use_exploded_cmyk else (5 if use_exploded_multi else 3),
                    sandwich_layers=sandwich_layers,
                    use_fill=use_fill,
                    base_layers=base_layers,
                )
            elif choice == "3":
                # 3D preview in browser
                preview_stl_gen = STLGenerator(
                    grayscale,
                    width,
                    layer_height,
                    model_height,
                    selected_filaments,
                    alpha_mask=img_processor.alpha_mask,
                    use_cap_layers=use_cap_layers,
                    image_rgb=img_processor.image,
                    use_face_down=use_face_down,
                    use_exploded=use_exploded,
                    use_exploded_multi=use_exploded_multi,
                    use_exploded_cmyk=use_exploded_cmyk,
                    sandwich_layers=sandwich_layers,
                    use_fill=use_fill,
                    base_layers=base_layers,
                )
                show_3d_preview(preview_stl_gen)
                # Loop continues to show 2D preview + menu again
            else:
                # Choice 1 or default - proceed with STL generation
                preview_approved = True

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
            use_cap_layers=use_cap_layers,
            image_rgb=img_processor.image,
            use_face_down=use_face_down,
            use_exploded=use_exploded,
            use_exploded_multi=use_exploded_multi,
            use_exploded_cmyk=use_exploded_cmyk,
            sandwich_layers=sandwich_layers,
            use_fill=use_fill,
            base_layers=base_layers,
        )

        generated_files = stl_gen.generate_all(output_path, single_stl=False)
        logger.info(f"Generated {len(generated_files)} STL file(s)")

        # Show Beer-Lambert preview for face-down mode
        if use_face_down and hasattr(stl_gen, 'face_down_pixel_height'):
            logger.info("Rendering face-down Beer-Lambert preview...")
            if hasattr(stl_gen, 'face_down_cap_height'):
                # Face-down + cap layers: use combined preview renderer
                preview_rgb = stl_gen._render_face_down_cap_preview(
                    stl_gen.face_down_pixel_height, stl_gen.face_down_z_boundaries,
                    stl_gen.face_down_image_filaments, stl_gen.face_down_transparent_filament,
                    stl_gen.face_down_cap_height
                )
            else:
                preview_rgb = stl_gen._render_face_down_preview(
                    stl_gen.face_down_pixel_height, stl_gen.face_down_z_boundaries,
                    stl_gen.face_down_filaments
                )
            preview_path = output_path.with_name(output_path.stem + '_preview.png')

            import matplotlib.pyplot as plt
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

            original_display = np.flipud(img_processor.image)
            if use_face_down:
                original_display = np.fliplr(original_display)
            ax1.imshow(original_display)
            ax1.set_title('Original Image', fontsize=14)
            ax1.axis('off')

            preview_display = np.flipud(preview_rgb)
            if use_face_down:
                preview_display = np.fliplr(preview_display)
            ax2.imshow(preview_display)
            ax2.set_title('Face-Down Preview (viewed from bed side)', fontsize=14)
            ax2.axis('off')

            plt.tight_layout()
            plt.savefig(str(preview_path), dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Preview saved to {preview_path}")

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
                f.write(f"Each colour has up to 2 intensity levels = max 8 sandwiches.\n")
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
            elif use_face_down and use_cap_layers:
                f.write("FACE-DOWN + CAP LAYERS MODE\n")
                f.write("1. Load all STL files into slicer at once\n")
                f.write("2. Right-click each part and assign the corresponding filament\n")
                f.write("3. The transparent cap prints first (on the bed, 100% solid)\n")
                f.write("4. Image colors and negative fill print on top of the cap\n")
                f.write("5. The model is completely flat — no topographical surface\n")
                f.write("6. After printing, flip the model over for display\n")
                f.write("7. The bed-side surface (smooth transparent cap) becomes the viewing side\n")
                f.write("8. Backlight for best effect\n")
            elif use_face_down:
                f.write("FACE-DOWN MODE\n")
                f.write("1. Load all STL files into slicer at once\n")
                f.write("2. Right-click each part and assign the corresponding filament\n")
                f.write("3. Print flat on bed (clear/light filament goes first)\n")
                f.write("4. After printing, flip the model over for display\n")
                f.write("5. The bed-side surface (smooth) becomes the viewing side\n")
                f.write("6. Backlight for best effect\n")
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
