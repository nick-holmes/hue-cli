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

    def select_best_filaments(self, target_colors_lab, count, layer_height=0.08, min_color_difference=15.0, randomize=False, use_cap_layers=False):
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
        logger.info(f"Selecting {count} unique filaments from library (randomize: {randomize}, cap_layers: {use_cap_layers})")

        if use_cap_layers:
            # Cap layers mode: select dark base + clear top + colors
            return self._select_with_cap_layers(target_colors_lab, count, layer_height, min_color_difference, randomize)

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

        return self.df.iloc[selected_indices].reset_index(drop=True)

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
        # Look for very dark colors (L < 30) with good opacity
        dark_candidates = []
        for idx, row in self.df.iterrows():
            luminosity = row['lab'][0]
            if luminosity < 30:  # Very dark
                dark_candidates.append((luminosity, idx))

        if not dark_candidates:
            logger.warning("No very dark filaments found (L < 30), using darkest available")
            # Just use darkest
            darkest_idx = self.df['lab'].apply(lambda x: x[0]).idxmin()
            selected_indices.append(darkest_idx)
        else:
            # Sort by darkness and pick darkest
            dark_candidates.sort()
            selected_indices.append(dark_candidates[0][1])

        dark_filament = self.df.iloc[selected_indices[0]]
        logger.info(f"Selected dark base: {dark_filament['name']} (L={dark_filament['lab'][0]:.1f}, TD={dark_filament['transmission_distance']}mm)")

        # 2. Select PURE TRANSPARENT layer (colorless, highest TD)
        # Look for truly colorless transparent filaments (low a,b values in LAB)
        clear_candidates = []
        for idx, row in self.df.iterrows():
            if idx in selected_indices:
                continue
            luminosity = row['lab'][0]
            a_value = abs(row['lab'][1])  # Color: red-green axis
            b_value = abs(row['lab'][2])  # Color: blue-yellow axis
            td = row['transmission_distance']

            # PURE transparent: very light, very transparent, AND minimal color tint
            if luminosity > 85 and td > 5.0 and (a_value + b_value) < 10.0:
                # Score: prefer higher TD, higher L, lower color (a,b close to 0)
                color_penalty = (a_value + b_value) * 2.0
                score = luminosity + td * 5.0 - color_penalty
                clear_candidates.append((score, idx))

        if not clear_candidates:
            logger.warning("No pure transparent filaments found (L > 85, TD > 5mm, low a/b), trying less strict...")
            # Try with relaxed color constraint
            for idx, row in self.df.iterrows():
                if idx in selected_indices:
                    continue
                luminosity = row['lab'][0]
                td = row['transmission_distance']
                if luminosity > 85 and td > 5.0:
                    score = luminosity + td * 5.0
                    clear_candidates.append((score, idx))

        if not clear_candidates:
            logger.warning("No clear filaments found at all, using most transparent available")
            # Find most transparent filament
            best_td_idx = None
            best_td = 0
            for idx, row in self.df.iterrows():
                if idx not in selected_indices:
                    if row['transmission_distance'] > best_td:
                        best_td = row['transmission_distance']
                        best_td_idx = idx
            if best_td_idx is not None:
                selected_indices.append(best_td_idx)
        else:
            # Sort by score (descending) and pick best
            clear_candidates.sort(reverse=True)
            selected_indices.append(clear_candidates[0][1])

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

            # NO horizontal flip needed - coordinate system matches STL export

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

    def simulate_layer_stack_color(self, layer_indices, layer_heights):
        """
        Simulate the perceived color of a stack of layers
        layer_indices: list of filament indices from bottom to top
        layer_heights: thickness of each layer in mm
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

            # Beer-Lambert transmissive filter model:
            # - Transmitted fraction passes through unchanged
            # - Absorbed fraction takes on the filament's color
            # At transmission=1 (transparent): light passes through unchanged
            # At transmission=0 (opaque): output = filament color
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



def generate_preview(grayscale, selected_filaments, layer_height, model_height, use_cap_layers, alpha_mask):
    """Generate preview image showing simulated color stacking appearance."""
    height_px, width_px = grayscale.shape
    preview = np.ones((height_px, width_px, 3))

    num_colors = len(selected_filaments)
    num_layers = int(model_height / layer_height)

    if use_cap_layers:
        # Cap layers mode: flat base + shaped colors + clear
        num_middle_colors = num_colors - 2

        # For each pixel, determine which layers it gets based on brightness
        for y in range(height_px):
            for x in range(width_px):
                if alpha_mask[y, x] < 0.5:
                    preview[y, x] = [1, 1, 1]  # White for transparent areas
                    continue

                brightness = grayscale[y, x]

                # Start with black base
                color_stack = [selected_filaments.iloc[0]['color_hex']]

                # Add middle colors based on brightness thresholds
                for i in range(num_middle_colors):
                    threshold = (i + 1) / (num_middle_colors + 1)
                    if brightness >= threshold:
                        color_stack.append(selected_filaments.iloc[i + 1]['color_hex'])

                # Add clear top (always present)
                color_stack.append(selected_filaments.iloc[-1]['color_hex'])

                # Simulate color mixing (simple average)
                final_color = simulate_color_stack(color_stack)
                preview[y, x] = final_color

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
    parser.add_argument('--cap-layers', type=str, default=None, metavar='BOOL',
                        help='Use cap layers (black base + auto colors + clear top) [yes/no]')

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

        color_count = args.colors if args.colors is not None else prompt_with_default(
            "Number of colors/filaments",
            4,
            int
        )

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

        use_cap_layers = (args.cap_layers.lower() in ('y', 'yes', 'true', '1')) if args.cap_layers is not None else prompt_with_default(
            "Use cap layers (black base + auto colors + clear top)",
            False,
            bool
        )

        if use_cap_layers and color_count < 3:
            logger.warning("Cap layers requires at least 3 colors. Setting to 3.")
            color_count = 3

        print("\n" + "=" * 60)

        # Pipeline execution
        logger.info("Starting STL generation...")

        # 1. Load filament library
        filament_lib = FilamentLibrary(filaments_csv)

        # 2. Load and prepare image
        img_processor = ImageProcessor(args.image, width, color_count)
        img_processor.load_and_prepare(nozzle_diameter=nozzle_diameter)

        # 3. Quantize colors
        dominant_colors_lab, kmeans, sorted_indices = img_processor.quantize_colors()

        # 4. Select best filaments (with transmission-aware color simulation)
        # Cap layers: black base + auto colors + clear top
        selected_filaments = filament_lib.select_best_filaments(
            dominant_colors_lab, color_count,
            layer_height=layer_height,
            min_color_difference=min_color_difference,
            use_cap_layers=use_cap_layers
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
        while not preview_approved:
            preview_img = generate_preview(
                grayscale,
                selected_filaments,
                layer_height,
                model_height,
                use_cap_layers,
                img_processor.alpha_mask
            )

            # Show preview in non-blocking mode
            import matplotlib.pyplot as plt
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

            # Left: Original image
            # During load_and_prepare, image was flipud for STL orientation
            # Flip it back for display to show original orientation
            original_display = np.flipud(img_processor.image)
            ax1.imshow(original_display)
            ax1.set_title('Original Image', fontsize=14)
            ax1.axis('off')

            # Right: Stacking preview
            preview_display = np.flipud(preview_img)
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
            print("3. Cancel")
            print("=" * 60)
            choice = input("Enter choice [1]: ").strip() or "1"

            plt.close(fig)

            if choice == "3":
                logger.info("Cancelled by user")
                return 0
            elif choice == "2":
                # Reduce color delta to allow more similar colors
                min_color_difference = min_color_difference * 0.7
                logger.info(f"Reducing color delta to {min_color_difference:.1f} and re-selecting filaments...")
                selected_filaments = filament_lib.select_best_filaments(
                    dominant_colors_lab, color_count,
                    layer_height=layer_height,
                    min_color_difference=min_color_difference,
                    use_cap_layers=use_cap_layers,
                    randomize=False
                )
                logger.info("New filaments selected (dark to light):")
                for i, row in selected_filaments.iterrows():
                    logger.info(f"  {i+1}. {row['name']} ({row['color_hex']}) - L={row['lab'][0]:.1f}")
                # Loop continues to show new preview
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
        )

        generated_files = stl_gen.generate_all(output_path, single_stl=False)
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
