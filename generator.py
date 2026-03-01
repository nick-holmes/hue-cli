#!/usr/bin/env python3
"""
HueCLI STL Generator

Generates stacked topographical STLs where each color sits directly on top
of the previous one. Per-pixel varying z_bottom = previous color's z_top.
Smooth shared-vertex grid surfaces with Beer-Lambert thickness mapping.
"""

import numpy as np
import pandas as pd
import trimesh
import logging
from pathlib import Path
from skimage import color

logger = logging.getLogger(__name__)


class STLGenerator:
    """Generates stacked topographical STLs with cumulative Z-heights per color

    Each color sits directly on top of the previous one (per-pixel varying bottom).
    Thickness within each color varies per pixel based on brightness + Beer-Lambert.
    Smooth shared-vertex grid surfaces for manifold, watertight meshes.
    """

    def __init__(self, image_grayscale, width_mm, layer_height, model_height,
                 selected_filaments, alpha_mask=None, use_cap_layers=False, image_rgb=None,
                 contrast_strength=2.0, use_face_down=False):
        """
        Args:
            image_grayscale: 2D array of brightness values (0=dark, 1=bright)
            width_mm: Model width in mm
            layer_height: Layer height in mm (e.g., 0.08)
            model_height: Total model height in mm (e.g., 2.0)
            selected_filaments: DataFrame of selected filaments (sorted dark to light)
            alpha_mask: Optional transparency mask
            use_cap_layers: If True, add flat base + negative + flat top
            image_rgb: Full RGB image for color matching (HxWx3 array, 0-1 range)
            contrast_strength: S-curve contrast boost (1.0=none, 2.0=moderate, 3.0=strong)
            use_face_down: If True, generate flat voxel grid for face-down printing
        """
        self.image_grayscale = image_grayscale
        self.image_rgb = image_rgb
        self.width_mm = width_mm
        self.layer_height = layer_height
        self.model_height = model_height
        self.selected_filaments = selected_filaments
        self.alpha_mask = alpha_mask if alpha_mask is not None else np.ones_like(image_grayscale)
        self.use_cap_layers = use_cap_layers
        self.use_face_down = use_face_down
        self.contrast_strength = contrast_strength

        self.num_layers = int(model_height / layer_height)
        self.num_colors = len(selected_filaments)

        # Calculate pixel size
        height_px, width_px = image_grayscale.shape
        self.pixel_size = width_mm / width_px

    def _calculate_color_thicknesses(self):
        """Calculate thickness for each color layer based on image color matching

        Returns thickness (not absolute heights) - these will be stacked cumulatively.
        CRITICAL: Normalizes thicknesses to use FULL model_height.

        Returns:
            List of 2D arrays (one per color) with thickness in mm for each pixel
        """
        height_px, width_px = self.image_grayscale.shape
        thickness_maps = []

        # Convert image RGB to LAB for color matching
        if self.image_rgb is not None:
            image_lab = color.rgb2lab(self.image_rgb)
        else:
            logger.warning("No RGB image provided - using grayscale approximation")
            image_lab = np.stack([self.image_grayscale * 100,
                                   np.zeros_like(self.image_grayscale),
                                   np.zeros_like(self.image_grayscale)], axis=-1)

        # Calculate available height per color (equal distribution as starting point)
        height_per_color = self.model_height / self.num_colors

        # For each filament, calculate contribution based on color matching
        for i, filament in self.selected_filaments.iterrows():
            td = filament['transmission_distance']
            filament_lab = filament['lab']
            filament_name = filament['name']

            # Calculate deltaE distance for each pixel
            filament_lab_broadcast = np.array([[filament_lab]])
            delta_e = color.deltaE_ciede2000(image_lab, filament_lab_broadcast)

            # Convert deltaE to contribution (0-1)
            # deltaE=0 (perfect match) → contribution=1.0
            # deltaE=50+ (very different) → contribution=0.0
            contribution = np.exp(-delta_e / 20.0)

            # Special handling for very dark and very light colors
            luminosity = filament_lab[0]

            if luminosity < 20:  # Very dark (e.g., black)
                # Boost contribution where image is dark
                brightness_factor = (1.0 - self.image_grayscale)
                contribution = np.maximum(contribution, brightness_factor * 0.7)

            elif luminosity > 90:  # Very light/clear
                # Boost contribution where image is bright
                brightness_factor = self.image_grayscale
                contribution = np.maximum(contribution, brightness_factor * 0.7)

            # Calculate thickness for this color
            # Base thickness from equal distribution, modulated by contribution
            thickness = contribution * height_per_color

            # Apply alpha mask
            thickness = thickness * self.alpha_mask

            thickness_maps.append(thickness)

        # CRITICAL FIX: Normalize thicknesses to use FULL model_height
        # Current issue: contributions don't sum to 1.0, so total height < model_height
        # Solution: Scale all thicknesses so max cumulative height = model_height

        # Calculate cumulative height before normalization
        total_thickness = np.zeros((height_px, width_px))
        for thickness in thickness_maps:
            total_thickness += thickness

        max_total = np.max(total_thickness[self.alpha_mask >= 0.5])

        if max_total > 0:
            # Scale factor to reach model_height
            scale_factor = self.model_height / max_total
            logger.info(f"Normalizing heights: scaling by {scale_factor:.3f}x to reach {self.model_height}mm")

            # Apply scaling to all thickness maps
            thickness_maps = [t * scale_factor for t in thickness_maps]

        # Log statistics after normalization
        for i, (filament, thickness) in enumerate(zip(self.selected_filaments.iterrows(), thickness_maps)):
            idx, filament_row = filament
            if np.any(thickness > 0):
                avg_thickness = np.mean(thickness[thickness > 0])
                logger.info(f"  {filament_row['name']}: avg thickness {avg_thickness:.3f}mm (TD={filament_row['transmission_distance']:.1f}mm)")

        return thickness_maps

    def _convert_to_stacked_heights(self, thickness_maps):
        """Convert thickness maps to cumulative Z-heights (stacking on top of each other)

        Args:
            thickness_maps: List of 2D thickness arrays (one per color)

        Returns:
            List of (z_bottom, z_top) tuples for each color
        """
        height_px, width_px = thickness_maps[0].shape
        z_heights = []

        # Start at z=0
        cumulative_height = np.zeros((height_px, width_px))

        for i, thickness in enumerate(thickness_maps):
            # This color's bottom is the current cumulative height
            z_bottom = cumulative_height.copy()

            # Add this color's thickness
            cumulative_height += thickness

            # This color's top is the new cumulative height
            z_top = cumulative_height.copy()

            z_heights.append((z_bottom, z_top))

        return z_heights

    def _calculate_overlapping_tops(self):
        """Calculate varying top height for each color STL using proper COLOR matching

        Each STL has flat bottom at z=0 and varying top.
        Height based on color deltaE distance - closer color match = thicker layer.

        Returns:
            List of 2D arrays (one per color) with top z-height for each pixel
        """
        height_px, width_px = self.image_grayscale.shape
        top_heights = []

        # Convert image RGB to LAB for color matching
        if self.image_rgb is not None:
            image_lab = color.rgb2lab(self.image_rgb)
        else:
            # Fallback to grayscale if no RGB provided
            logger.warning("No RGB image provided - using grayscale approximation")
            image_lab = np.stack([self.image_grayscale * 100,
                                   np.zeros_like(self.image_grayscale),
                                   np.zeros_like(self.image_grayscale)], axis=-1)

        # For each filament, calculate contribution based on color matching
        for i, filament in self.selected_filaments.iterrows():
            td = filament['transmission_distance']
            filament_lab = filament['lab']
            filament_name = filament['name']

            # Calculate deltaE distance for each pixel
            # Reshape for broadcasting
            filament_lab_broadcast = np.array([[filament_lab]])  # Shape: (1, 1, 3)

            # Calculate color difference using deltaE
            delta_e = color.deltaE_ciede2000(image_lab, filament_lab_broadcast)

            # Convert deltaE to contribution (0-1)
            # deltaE=0 (perfect match) → contribution=1.0
            # deltaE=50+ (very different) → contribution=0.0
            # Use exponential decay for smooth transition
            contribution = np.exp(-delta_e / 20.0)  # Adjust 20.0 to control selectivity

            # Special handling for very dark and very light colors
            luminosity = filament_lab[0]

            if luminosity < 20:  # Very dark (e.g., black)
                # Also consider brightness - boost contribution where image is dark
                brightness_factor = (1.0 - self.image_grayscale)
                contribution = np.maximum(contribution, brightness_factor * 0.8)

            elif luminosity > 90:  # Very light/clear
                # Also consider brightness - boost contribution where image is bright
                brightness_factor = self.image_grayscale
                contribution = np.maximum(contribution, brightness_factor * 0.8)

            # Scale by TD: Higher TD needs more thickness to show the same opacity
            # Use Beer-Lambert law approximation
            # For same visual effect: thickness ∝ TD
            td_reference = 10.0  # Reference TD
            td_scale = td / td_reference

            # Calculate top height
            # Base height from contribution, scaled by TD
            top_height = contribution * self.model_height * td_scale

            # Clamp to valid range
            top_height = np.clip(top_height, 0, self.model_height)

            # Apply alpha mask
            top_height = top_height * self.alpha_mask

            top_heights.append(top_height)

            # Log statistics
            if np.any(top_height > 0):
                pixels_used = (top_height > 0.001).sum()
                pixels_total = (self.alpha_mask > 0.5).sum()
                coverage_pct = pixels_used / pixels_total * 100 if pixels_total > 0 else 0
                logger.info(f"  {filament_name}: height 0-{np.max(top_height):.2f}mm, covers {pixels_used}/{pixels_total} pixels ({coverage_pct:.1f}%), TD={td:.1f}mm")

        return top_heights

    def _generate_overlapping_stl(self, z_top, pixel_mask):
        """Generate STL with flat bottom (z=0) and varying top

        This creates overlapping geometry that slicers can handle as multi-part.

        Args:
            z_top: 2D array of top z-heights for each pixel
            pixel_mask: 2D boolean array of which pixels to include

        Returns:
            trimesh.Trimesh object
        """
        height_px, width_px = pixel_mask.shape

        # Create vertices
        vertices = []
        vertex_map_bottom = {}
        vertex_map_top = {}

        # Generate vertices for active pixels
        for y in range(height_px + 1):
            for x in range(width_px + 1):
                # Check if any adjacent pixel is active
                is_needed = False
                z_top_avg = 0
                count = 0

                for dy in [-1, 0]:
                    for dx in [-1, 0]:
                        py, px = y + dy, x + dx
                        if 0 <= py < height_px and 0 <= px < width_px:
                            if pixel_mask[py, px]:
                                is_needed = True
                                z_top_avg += z_top[py, px]
                                count += 1

                if is_needed and count > 0:
                    z_top_avg /= count

                    # Bottom vertex (FLAT at z=0)
                    idx_bottom = len(vertices)
                    vertices.append([x * self.pixel_size, y * self.pixel_size, 0.0])
                    vertex_map_bottom[(y, x)] = idx_bottom

                    # Top vertex (VARYING height)
                    idx_top = len(vertices)
                    vertices.append([x * self.pixel_size, y * self.pixel_size, z_top_avg])
                    vertex_map_top[(y, x)] = idx_top

        if len(vertices) == 0:
            return trimesh.Trimesh()

        vertices = np.array(vertices)
        faces = []

        # Generate faces for active pixels
        for y in range(height_px):
            for x in range(width_px):
                if not pixel_mask[y, x]:
                    continue

                corners = [(y, x), (y, x+1), (y+1, x), (y+1, x+1)]
                if all(c in vertex_map_bottom for c in corners):
                    # Bottom face (flat)
                    v0 = vertex_map_bottom[(y, x)]
                    v1 = vertex_map_bottom[(y, x+1)]
                    v2 = vertex_map_bottom[(y+1, x)]
                    v3 = vertex_map_bottom[(y+1, x+1)]
                    faces.append([v0, v2, v1])
                    faces.append([v1, v2, v3])

                    # Top face (varying)
                    v0 = vertex_map_top[(y, x)]
                    v1 = vertex_map_top[(y, x+1)]
                    v2 = vertex_map_top[(y+1, x)]
                    v3 = vertex_map_top[(y+1, x+1)]
                    faces.append([v0, v1, v2])
                    faces.append([v1, v3, v2])

        # Side walls
        def is_active(y, x):
            return 0 <= y < height_px and 0 <= x < width_px and pixel_mask[y, x]

        for y in range(height_px):
            for x in range(width_px):
                if not pixel_mask[y, x]:
                    continue

                # Left edge
                if not is_active(y, x - 1):
                    if (y, x) in vertex_map_bottom and (y+1, x) in vertex_map_bottom:
                        v0 = vertex_map_bottom[(y, x)]
                        v1 = vertex_map_bottom[(y+1, x)]
                        v2 = vertex_map_top[(y, x)]
                        v3 = vertex_map_top[(y+1, x)]
                        faces.append([v0, v1, v2])
                        faces.append([v1, v3, v2])

                # Right edge
                if not is_active(y, x + 1):
                    if (y, x+1) in vertex_map_bottom and (y+1, x+1) in vertex_map_bottom:
                        v0 = vertex_map_bottom[(y, x+1)]
                        v1 = vertex_map_bottom[(y+1, x+1)]
                        v2 = vertex_map_top[(y, x+1)]
                        v3 = vertex_map_top[(y+1, x+1)]
                        faces.append([v0, v2, v1])
                        faces.append([v1, v2, v3])

                # Front edge
                if not is_active(y - 1, x):
                    if (y, x) in vertex_map_bottom and (y, x+1) in vertex_map_bottom:
                        v0 = vertex_map_bottom[(y, x)]
                        v1 = vertex_map_bottom[(y, x+1)]
                        v2 = vertex_map_top[(y, x)]
                        v3 = vertex_map_top[(y, x+1)]
                        faces.append([v0, v2, v1])
                        faces.append([v1, v2, v3])

                # Back edge
                if not is_active(y + 1, x):
                    if (y+1, x) in vertex_map_bottom and (y+1, x+1) in vertex_map_bottom:
                        v0 = vertex_map_bottom[(y+1, x)]
                        v1 = vertex_map_bottom[(y+1, x+1)]
                        v2 = vertex_map_top[(y+1, x)]
                        v3 = vertex_map_top[(y+1, x+1)]
                        faces.append([v0, v1, v2])
                        faces.append([v1, v3, v2])

        if len(faces) == 0:
            return trimesh.Trimesh()

        faces = np.array(faces)

        # Create mesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
        mesh.fix_normals()

        return mesh

    def _calculate_thickness_maps(self):
        """Calculate thickness maps for each color layer based on image and TD values

        Returns:
            List of 2D arrays, one per color, with thickness in mm for each pixel
        """
        height_px, width_px = self.image_grayscale.shape
        thickness_maps = []

        # Reference TD for scaling (use middle value)
        reference_td = 10.0

        if self.use_cap_layers:
            # Cap layers: base + colors + clear
            num_middle_colors = self.num_colors - 2

            # Available height for variable layers (exclude thin flat base and top)
            base_height = 0.16  # Thin flat base for adhesion
            top_height = 0.16   # Thin flat top
            available_height = self.model_height - base_height - top_height

            # 1. FLAT BASE (just return constant thickness)
            base_thickness = np.full((height_px, width_px), base_height)
            thickness_maps.append(base_thickness)

            # 2. MIDDLE COLORS (topographical)
            height_per_color = available_height / (num_middle_colors + 1)  # +1 for clear fill

            for i in range(num_middle_colors):
                filament = self.selected_filaments.iloc[i + 1]
                td = filament['transmission_distance']

                # Dark colors: thick where image is dark
                # Light colors: thick where image is bright
                # Use luminosity to determine if color is dark or light
                luminosity = filament['lab'][0]

                if luminosity < 50:  # Dark color
                    # Thick where dark (inverse brightness)
                    intensity = 1.0 - self.image_grayscale
                else:  # Light color
                    # Thick where bright
                    intensity = self.image_grayscale

                # Scale by TD: high TD needs more thickness to show color
                td_scale = np.clip(td / reference_td, 0.1, 3.0)

                # Calculate thickness (0 to height_per_color)
                thickness = intensity * height_per_color * td_scale

                # Apply alpha mask
                thickness = thickness * self.alpha_mask

                thickness_maps.append(thickness)

            # 3. CLEAR FILL (topographical - fills to constant total height)
            # This will be calculated during z-height conversion

        else:
            # Standard mode: all colors are topographical
            height_per_color = self.model_height / self.num_colors

            for i, filament in self.selected_filaments.iterrows():
                td = filament['transmission_distance']
                luminosity = filament['lab'][0]

                # Sort filaments light to dark (bottom to top)
                # Lightest at bottom, darkest at top
                if i == 0:  # Bottom layer (lightest)
                    # Should cover all pixels
                    intensity = np.ones_like(self.image_grayscale)
                else:
                    # Upper layers: thickness based on brightness and color
                    if luminosity < 50:  # Dark color
                        intensity = 1.0 - self.image_grayscale
                    else:  # Light color
                        intensity = self.image_grayscale

                # Scale by TD
                td_scale = np.clip(td / reference_td, 0.1, 3.0)

                # Calculate thickness
                thickness = intensity * height_per_color * td_scale

                # Apply alpha mask
                thickness = thickness * self.alpha_mask

                thickness_maps.append(thickness)

        return thickness_maps

    def _convert_to_z_heights(self, thickness_maps):
        """Convert thickness maps to absolute z-height maps (cumulative stacking)

        Args:
            thickness_maps: List of 2D thickness arrays (one per color)

        Returns:
            List of (z_bottom, z_top) tuples for each color
        """
        z_heights = []
        height_px, width_px = self.image_grayscale.shape

        if self.use_cap_layers:
            # Cap layers mode
            cumulative_height = np.zeros((height_px, width_px))

            # 1. Flat base
            z_bottom = cumulative_height.copy()
            cumulative_height += thickness_maps[0]
            z_top = cumulative_height.copy()
            z_heights.append((z_bottom, z_top))

            # 2. Middle colors (topographical)
            num_middle_colors = self.num_colors - 2

            for i in range(num_middle_colors):
                z_bottom = cumulative_height.copy()
                cumulative_height += thickness_maps[i + 1]
                z_top = cumulative_height.copy()
                z_heights.append((z_bottom, z_top))

            # 3. Clear fill (topographical - fills to constant height)
            # Bottom: current cumulative height (varies)
            # Top: model_height - top_flat_height (constant)
            target_height = self.model_height - 0.16  # Leave room for flat top

            z_bottom = cumulative_height.copy()
            z_top = np.full((height_px, width_px), target_height)
            z_heights.append((z_bottom, z_top))

            # 4. Flat top (added as post-process, not in thickness_maps)
            # Will be handled separately

        else:
            # Standard mode: stack all layers cumulatively
            cumulative_height = np.zeros((height_px, width_px))

            for i in range(len(thickness_maps)):
                z_bottom = cumulative_height.copy()
                cumulative_height += thickness_maps[i]
                z_top = cumulative_height.copy()
                z_heights.append((z_bottom, z_top))

        return z_heights

    def _apply_contrast_enhancement(self, brightness):
        """Apply contrast enhancement optimized for midtone detail

        Uses adaptive approach based on image brightness:
        - For bright images (mean > 0.65): Darken to give more material to dark colors
        - For dark images (mean < 0.35): Brighten to preserve highlights
        - For mid-tone images: Apply S-curve for contrast boost

        Args:
            brightness: 2D array of brightness values (0-1)

        Returns:
            Enhanced brightness values (0-1)
        """
        if self.contrast_strength == 1.0:
            return brightness  # No enhancement

        mean_brightness = np.mean(brightness[self.alpha_mask >= 0.5])

        # Adaptive enhancement based on image characteristics
        if mean_brightness > 0.65:
            # BRIGHT IMAGE: Use power curve to darken and increase contrast
            # Higher gamma (>1.0) = darker overall = more material for dark colors
            # This prevents washout in bright images
            gamma = 1.5  # Darken midtones significantly (0.5^1.5 = 0.35)
            enhanced = np.power(brightness, gamma)

            # Then apply S-curve for extra contrast
            center = np.mean(enhanced)  # Center at enhanced image mean
            strength = 4.0  # Strong S-curve
            enhanced = 1.0 / (1.0 + np.exp(-strength * (enhanced - center)))
            enhanced = (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min())

            logger.info(f"Contrast: bright image (mean={mean_brightness:.3f}), gamma={gamma:.2f} + S-curve")

        elif mean_brightness < 0.35:
            # DARK IMAGE: Use inverse power curve to brighten
            gamma = 0.7
            enhanced = np.power(brightness, gamma)

            logger.info(f"Contrast: dark image (mean={mean_brightness:.3f}), gamma={gamma:.2f}")

        else:
            # MID-TONE IMAGE: Standard S-curve
            center = 0.5
            enhanced = 1.0 / (1.0 + np.exp(-self.contrast_strength * (brightness - center)))
            enhanced = (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min())

            logger.info(f"Contrast: S-curve (mean={mean_brightness:.3f}, strength={self.contrast_strength:.1f})")

        return enhanced

    def _vectorized_beer_lambert(self, filament_rgbs, filament_tds, thickness_combos, reverse=False):
        """Compute Beer-Lambert transmitted colors for all thickness combinations

        Uses corrected transmissive filter model:
        light_out = light_in * transmission + filament_rgb * (1 - transmission)

        Args:
            filament_rgbs: (num_colors, 3) array of filament RGB values (0-1)
            filament_tds: (num_colors,) array of transmission distances (mm)
            thickness_combos: (n_combos, num_colors) array of thicknesses (mm)
            reverse: If True, iterate filaments in reverse order (for face-down mode
                     where light enters from the dark/top side first)

        Returns:
            (n_combos, 3) array of resulting RGB colors (0-1)
        """
        n_combos = len(thickness_combos)
        light = np.ones((n_combos, 3))  # White backlight

        indices = range(len(filament_tds) - 1, -1, -1) if reverse else range(len(filament_tds))
        for k in indices:
            thickness = thickness_combos[:, k]  # (n_combos,)
            td = max(filament_tds[k], 0.1)
            rgb = filament_rgbs[k]  # (3,)

            # Beer-Lambert transmission factor
            transmission = np.exp(-thickness / td)  # (n_combos,)
            transmission = transmission[:, None]  # (n_combos, 1)

            # Transmissive filter: transparent → pass through, opaque → filament color
            light = light * transmission + rgb[None, :] * (1.0 - transmission)
            light = np.clip(light, 0, 1)

        return light

    def _optimize_thicknesses_beer_lambert(self, sorted_filaments, layers_per_color, layers_per_color_list=None, reverse_light=False):
        """Find optimal per-pixel layer thicknesses using Beer-Lambert physics

        Uses lookup table + KNN interpolation for smooth continuous thicknesses:
        1. Enumerate all possible quantized thickness combinations
        2. Simulate Beer-Lambert color for each combination
        3. Build KDTree in LAB space for fast matching
        4. For each pixel, find K nearest combos and interpolate thicknesses
           using inverse-distance weighting for smooth gradients

        Args:
            sorted_filaments: DataFrame of filaments sorted dark to light
            layers_per_color: Number of layers allocated to each color (uniform fallback)
            layers_per_color_list: Optional list of ints, per-color layer budgets.
                                   When provided, overrides uniform layers_per_color.

        Returns:
            List of 2D thickness arrays (one per color), continuous values in mm
        """
        from itertools import product
        from scipy.spatial import cKDTree

        # Extract filament properties
        filament_rgbs = np.array([f['rgb'] for _, f in sorted_filaments.iterrows()])
        filament_tds = np.array([f['transmission_distance'] for _, f in sorted_filaments.iterrows()])

        # Compute quantized thickness levels for each color
        thickness_levels = []
        max_thicknesses = []
        for i in range(self.num_colors):
            if layers_per_color_list is not None:
                n_layers = layers_per_color_list[i]
            else:
                layer_start = i * layers_per_color
                layer_end = self.num_layers if i == self.num_colors - 1 else (i + 1) * layers_per_color
                n_layers = layer_end - layer_start
            levels = np.arange(0, n_layers + 1) * self.layer_height
            thickness_levels.append(levels)
            max_thicknesses.append(levels[-1])

        # Generate all combinations
        combos = np.array(list(product(*thickness_levels)))
        logger.info(f"Beer-Lambert optimization: {len(combos)} thickness combinations "
                     f"({' x '.join(str(len(l)) for l in thickness_levels)} levels)")

        # Simulate Beer-Lambert color for each combination
        combo_colors_rgb = self._vectorized_beer_lambert(filament_rgbs, filament_tds, combos, reverse=reverse_light)

        # Convert to LAB for perceptual matching
        combo_colors_lab = color.rgb2lab(combo_colors_rgb.reshape(-1, 1, 3)).reshape(-1, 3)

        # Build KDTree for fast nearest-neighbor lookup
        tree = cKDTree(combo_colors_lab)

        # Get target pixel colors in LAB
        if self.image_rgb is not None:
            target_lab = color.rgb2lab(self.image_rgb).reshape(-1, 3)
        else:
            gray_rgb = np.stack([self.image_grayscale] * 3, axis=-1)
            target_lab = color.rgb2lab(gray_rgb).reshape(-1, 3)

        # KNN interpolation: find K nearest combos and blend thicknesses
        # This produces CONTINUOUS thickness values instead of discrete steps
        K = min(8, len(combos))
        distances, indices = tree.query(target_lab, k=K)
        h, w = self.image_grayscale.shape

        # Log matching quality (using nearest neighbor distance)
        valid_mask = (self.alpha_mask >= 0.5).ravel()
        valid_distances = distances[:, 0][valid_mask]
        logger.info(f"Beer-Lambert matching: mean deltaE={np.mean(valid_distances):.2f}, "
                     f"median={np.median(valid_distances):.2f}, "
                     f"95th percentile={np.percentile(valid_distances, 95):.2f}")

        # Inverse-distance weighted interpolation
        # Add small epsilon to avoid division by zero for exact matches
        eps = 1e-6
        weights = 1.0 / (distances + eps)  # (n_pixels, K)
        weights_sum = weights.sum(axis=1, keepdims=True)
        weights = weights / weights_sum  # Normalize to sum to 1

        # Interpolate thicknesses for each color
        thickness_maps = []
        for k in range(self.num_colors):
            # Get thickness values for K nearest combos
            combo_thicknesses_k = combos[indices, k]  # (n_pixels, K)
            # Weighted average
            interpolated = (combo_thicknesses_k * weights).sum(axis=1)  # (n_pixels,)
            # Clamp to valid range
            interpolated = np.clip(interpolated, 0, max_thicknesses[k])
            t_map = interpolated.reshape(h, w)
            # Zero out transparent pixels
            t_map = np.where(self.alpha_mask >= 0.5, t_map, 0.0)
            thickness_maps.append(t_map)

            # Log statistics
            active = t_map[self.alpha_mask >= 0.5]
            if len(active) > 0:
                filament_name = sorted_filaments.iloc[k]['name']
                nonzero_pct = (active > 0).sum() / len(active) * 100
                unique_vals = len(np.unique(np.round(active / self.layer_height) * self.layer_height))
                logger.info(f"  {filament_name}: avg thickness {np.mean(active):.3f}mm, "
                             f"coverage {nonzero_pct:.1f}%, ~{unique_vals} height levels after quantization")

        return thickness_maps

    def generate_all(self, output_base_path, single_stl=True):
        """Generate STL files with global heightmap and fixed color boundaries

        One global heightmap maps brightness to height. Color boundaries are flat
        planes at fixed z-values, ensuring no color intermingling at any print layer.

        Args:
            output_base_path: Path for output files
            single_stl: If True (default), generate one combined STL with filament-change
                        schedule. If False, generate separate STL per color.

        Returns:
            List of (path, name, layer_start, layer_end) tuples
        """
        logger.info(f"Generating {self.num_colors}-color model: {self.model_height}mm tall, {self.num_layers} layers")

        if self.use_face_down:
            return self._generate_face_down(output_base_path)

        if self.use_cap_layers:
            return self._generate_with_cap_layers(output_base_path)

        generated_files = []

        # Sort dark-to-light by LAB luminosity
        sorted_filaments = self.selected_filaments.copy()
        sorted_filaments['luminosity'] = sorted_filaments['lab'].apply(
            lambda x: float(np.asarray(x).flat[0]))
        sorted_filaments = sorted_filaments.sort_values('luminosity').reset_index(drop=True)

        enhanced_grayscale = self._apply_contrast_enhancement(self.image_grayscale.copy())

        # Allocate layers proportionally to TD (transparent filaments need more thickness)
        filament_rgbs = np.array([f['rgb'] for _, f in sorted_filaments.iterrows()])
        filament_tds = np.array([f['transmission_distance'] for _, f in sorted_filaments.iterrows()])

        # Weight by TD: higher TD → more layers
        # Use log scale to dampen extreme TD ratios (e.g. TD=98.6 vs TD=4.3)
        # so mid-range colors get enough layers for topographical detail
        td_weights = np.log1p(np.maximum(filament_tds, 1.0))
        td_weights = td_weights / td_weights.sum()

        # Convert to layer counts (minimum 1 layer per color to exist in print)
        # Opaque filaments (low TD) need just 1 thin layer to show full color,
        # while transparent filaments (high TD) get proportionally more layers
        min_layers = 1
        available_layers = self.num_layers - min_layers * self.num_colors
        if available_layers < 0:
            available_layers = 0
            min_layers = max(1, self.num_layers // self.num_colors)

        layer_counts = np.round(td_weights * available_layers).astype(int) + min_layers

        # Fix rounding: adjust last color to use remaining layers
        layer_counts[-1] = self.num_layers - layer_counts[:-1].sum()

        # Compute cumulative layer boundaries
        layer_boundaries = np.concatenate([[0], np.cumsum(layer_counts)]).astype(int)

        for idx in range(self.num_colors):
            name = sorted_filaments.iloc[idx]['name']
            td = filament_tds[idx]
            lc = layer_counts[idx]
            height = lc * self.layer_height
            logger.info(f"  {name}: TD={td:.1f}mm → {lc} layers ({height:.2f}mm)")

        # Generate STL for each color with fixed z-boundaries
        from scipy import ndimage

        # Compute global max brightness for full-range normalization
        alpha_pixels = self.alpha_mask >= 0.5
        global_brightness_max = np.max(enhanced_grayscale[alpha_pixels]) if np.any(alpha_pixels) else 1.0

        height_px, width_px = enhanced_grayscale.shape

        # Compute single global heightmap (brightness → height)
        min_height = 2 * self.layer_height  # bed adhesion minimum
        pixel_height = min_height + enhanced_grayscale / max(global_brightness_max, 1e-6) * (self.model_height - min_height)
        pixel_height = np.clip(pixel_height, min_height, self.model_height)
        pixel_height = np.where(alpha_pixels, pixel_height, 0)

        # Fixed color boundary z-values (flat planes — no intermingling)
        z_boundaries = layer_boundaries * self.layer_height

        filament_schedule = []
        min_thickness = self.layer_height * 0.5

        for i, filament in sorted_filaments.iterrows():
            color_name = filament['name'].replace(' ', '_').replace('/', '_')

            # Layer range from TD-proportional allocation
            layer_start = layer_boundaries[i]
            layer_end = layer_boundaries[i + 1]

            # Fixed z boundaries for this color band
            z_bottom_flat = float(z_boundaries[i])
            z_top_boundary = float(z_boundaries[i + 1])

            # Pixels that reach into this color's z-band
            pixel_mask = (pixel_height > z_bottom_flat + min_thickness) & alpha_pixels

            pixel_count = np.sum(pixel_mask)
            total_pixels = np.sum(alpha_pixels)

            # Clean up small disconnected regions
            labeled, num_regions = ndimage.label(pixel_mask)
            min_region_size = 8

            if num_regions > 1 and i > 0:
                region_sizes = np.bincount(labeled.ravel())
                small_regions = region_sizes < min_region_size
                small_regions[0] = False
                pixel_mask[small_regions[labeled]] = False

                pixel_count_filtered = np.sum(pixel_mask)
                removed_pixels = pixel_count - pixel_count_filtered
                if removed_pixels > 0:
                    logger.info(f"  Filtered {removed_pixels} pixels in {np.sum(small_regions) - 1} small regions (< {min_region_size} pixels)")
                pixel_count = pixel_count_filtered

            logger.info(f"  {color_name}: layers {layer_start}-{layer_end} ({layer_end - layer_start} layers, {pixel_count} px)")

            if pixel_count > 0:
                # z_top: pixel height clamped to this color's band
                z_top_color = np.clip(pixel_height, z_bottom_flat, z_top_boundary)
                z_bottom_color = np.full_like(pixel_height, z_bottom_flat)

                # Effective mask: pixels with enough thickness in this band
                thickness = z_top_color - z_bottom_flat
                color_effective_mask = pixel_mask & (thickness >= min_thickness)

                pixels_after_filter = np.sum(color_effective_mask)
                if pixels_after_filter < pixel_count:
                    logger.debug(f"    Filtered {pixel_count - pixels_after_filter} thin pixels")

                if color_effective_mask.any():
                    z_top_max = np.max(z_top_color[color_effective_mask])
                    logger.info(f"    z: {z_bottom_flat:.2f} - {z_top_max:.2f}mm")

                filament_schedule.append((filament['name'], layer_start, layer_end))

                if not single_stl:
                    # Per-color STL mode: generate individual file
                    output_path = output_base_path.parent / f"{output_base_path.stem}_{color_name}.stl"
                    mesh = self._generate_topographical_stl(z_bottom_color, z_top_color, color_effective_mask)

                    if len(mesh.vertices) > 0:
                        mesh.export(str(output_path))
                        max_z_top = np.max(z_top_color[color_effective_mask])
                        layer_start_actual = int(np.floor(z_bottom_flat / self.layer_height))
                        layer_end_actual = int(np.ceil(max_z_top / self.layer_height))
                        generated_files.append((output_path, filament['name'], layer_start_actual, layer_end_actual))
                        logger.info(f"  Generated: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
                    else:
                        logger.warning(f"  Empty mesh for {color_name} - skipping")
            else:
                logger.warning(f"  No pixels for {color_name} - skipping")

        if single_stl:
            # Generate single combined STL (z=0 to global heightmap)
            logger.info("Generating combined STL...")
            output_path = output_base_path.parent / f"{output_base_path.stem}.stl"

            z_bottom_combined = np.zeros_like(pixel_height)
            combined_mask = alpha_pixels & (pixel_height > min_thickness)
            mesh = self._generate_topographical_stl(z_bottom_combined, pixel_height, combined_mask)

            if len(mesh.vertices) > 0:
                mesh.export(str(output_path))
                max_z_top = np.max(pixel_height[combined_mask])
                layer_end_actual = int(np.ceil(max_z_top / self.layer_height))
                generated_files.append((output_path, "Combined", 0, layer_end_actual))
                logger.info(f"  Generated: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
                logger.info(f"  Watertight: {mesh.is_watertight}")
            else:
                logger.warning("  Empty combined mesh - skipping")

            self.filament_schedule = filament_schedule

        return generated_files

    def generate_preview_scene(self):
        """Generate a trimesh.Scene with colored meshes for interactive 3D preview

        Mirrors the heightmap pipeline from generate_all() / _generate_face_down()
        and returns a Scene instead of exporting STLs. Downsamples to ~100K pixels
        for browser-renderable mesh size while preserving good detail.

        Returns:
            trimesh.Scene with one colored mesh per filament
        """
        from math import sqrt

        H, W = self.image_grayscale.shape

        # Downsample for browser rendering (~100K pixels target)
        ds = max(1, int(sqrt(H * W / 100000)))
        ds_grayscale = self.image_grayscale[::ds, ::ds]
        ds_alpha = self.alpha_mask[::ds, ::ds]
        ds_H, ds_W = ds_grayscale.shape
        ds_pixel_size = self.width_mm / ds_W

        logger.info(f"3D preview: {H}x{W} -> {ds_H}x{ds_W} (ds={ds}), "
                     f"pixel_size={ds_pixel_size:.3f}mm")

        # Sort dark-to-light (same as generate_all)
        sorted_filaments = self.selected_filaments.copy()
        sorted_filaments['luminosity'] = sorted_filaments['lab'].apply(
            lambda x: float(np.asarray(x).flat[0]))

        if self.use_cap_layers:
            # Cap layers: keep base (index 0) and clear (index -1) in place,
            # sort only middle colors by luminosity for optimal coverage mapping
            middle = sorted_filaments.iloc[1:-1].sort_values('luminosity').reset_index(drop=True)
            sorted_filaments = pd.concat([
                sorted_filaments.iloc[:1],
                middle,
                sorted_filaments.iloc[-1:]
            ]).reset_index(drop=True)
        else:
            sorted_filaments = sorted_filaments.sort_values('luminosity').reset_index(drop=True)

        # Swap only pixel_size and alpha_mask (used by called methods)
        orig_pixel_size = self.pixel_size
        orig_alpha = self.alpha_mask
        try:
            self.pixel_size = ds_pixel_size
            self.alpha_mask = ds_alpha

            # Contrast enhancement (uses self.alpha_mask internally)
            enhanced = self._apply_contrast_enhancement(ds_grayscale.copy())

            # TD-proportional layer allocation (same logic as generate_all)
            filament_tds = np.array([f['transmission_distance'] for _, f in sorted_filaments.iterrows()])
            td_weights = np.log1p(np.maximum(filament_tds, 1.0))
            td_weights = td_weights / td_weights.sum()

            min_layers = 1
            available_layers = self.num_layers - min_layers * self.num_colors
            if available_layers < 0:
                available_layers = 0
                min_layers = max(1, self.num_layers // self.num_colors)

            layer_counts = np.round(td_weights * available_layers).astype(int) + min_layers
            layer_counts[-1] = self.num_layers - layer_counts[:-1].sum()
            layer_boundaries = np.concatenate([[0], np.cumsum(layer_counts)]).astype(int)
            z_boundaries = layer_boundaries * self.layer_height

            # Compute heightmap
            alpha_pixels = ds_alpha >= 0.5
            global_brightness_max = np.max(enhanced[alpha_pixels]) if np.any(alpha_pixels) else 1.0
            min_height = 2 * self.layer_height

            if self.use_face_down:
                normalized = enhanced / max(global_brightness_max, 1e-6)
                pixel_height = min_height + (1.0 - normalized) * (self.model_height - min_height)
            else:
                pixel_height = min_height + enhanced / max(global_brightness_max, 1e-6) * (self.model_height - min_height)

            pixel_height = np.clip(pixel_height, min_height, self.model_height)
            pixel_height = np.where(alpha_pixels, pixel_height, 0)

            scene = trimesh.Scene()
            min_thickness = self.layer_height * 0.5

            combined_mask = alpha_pixels & (pixel_height > min_thickness)
            z_bottom_all = np.zeros_like(pixel_height)

            if self.use_cap_layers:
                # Flat top for cap-layer mode
                z_top = np.full_like(pixel_height, self.model_height)
            else:
                # Topographical heightmap for standard and face-down modes
                z_top = pixel_height

            z_top = np.where(combined_mask, z_top, 0)
            mesh = self._generate_topographical_stl(z_bottom_all, z_top, combined_mask)

            if len(mesh.vertices) > 0:
                if self.use_cap_layers:
                    preview_rgb = self._render_cap_layer_preview(
                        enhanced, sorted_filaments)
                else:
                    preview_rgb = self._render_face_down_preview(
                        pixel_height, z_boundaries, sorted_filaments)

                # Auto-contrast: Beer-Lambert output is physically correct but
                # visually compressed (often near-white or near-black depending
                # on filament TDs). Stretch to use full display range.
                valid_pixels = preview_rgb[combined_mask]
                if len(valid_pixels) > 0:
                    p_lo = np.percentile(valid_pixels, 0.5)
                    p_hi = np.percentile(valid_pixels, 99.5)
                    rng = max(p_hi - p_lo, 0.01)
                    preview_rgb = (preview_rgb - p_lo) / rng
                    preview_rgb = np.clip(preview_rgb, 0, 1)

                centroids = mesh.triangles_center
                px = np.clip((centroids[:, 0] / self.pixel_size).astype(int),
                             0, ds_W - 1)
                py = np.clip((centroids[:, 1] / self.pixel_size).astype(int),
                             0, ds_H - 1)

                face_colors = np.zeros((len(mesh.faces), 4), dtype=np.uint8)
                face_colors[:, :3] = (preview_rgb[py, px] * 255).astype(np.uint8)
                face_colors[:, 3] = 255
                mesh.visual.face_colors = face_colors

                if self.use_face_down:
                    # Flip to show as-printed view: bed surface up + undo fliplr
                    max_x = float(np.max(mesh.vertices[:, 0]))
                    max_z = float(np.max(mesh.vertices[:, 2]))
                    mesh.vertices[:, 0] = max_x - mesh.vertices[:, 0]
                    mesh.vertices[:, 2] = max_z - mesh.vertices[:, 2]
                    mesh.fix_normals()

                scene.add_geometry(mesh, geom_name='preview')

            logger.info(f"3D preview scene: {len(scene.geometry)} meshes")
            return scene

        finally:
            self.pixel_size = orig_pixel_size
            self.alpha_mask = orig_alpha

    def _generate_standard(self, output_base_path):
        """Generate standard topographical STLs.

        Each color has varying thickness based on image brightness and TD values.
        """
        logger.info("Generating standard topographical STLs")

        generated_files = []

        # Calculate thickness maps for each color
        logger.info("Calculating thickness maps using TD values...")
        thickness_maps = self._calculate_thickness_maps()

        # Convert to absolute z-heights (cumulative stacking)
        logger.info("Converting to cumulative z-heights...")
        z_heights = self._convert_to_z_heights(thickness_maps)

        # Generate STL for each color
        for i, filament in self.selected_filaments.iterrows():
            color_name = filament['name'].replace(' ', '_').replace('/', '_')
            output_path = output_base_path.parent / f"{output_base_path.stem}_{color_name}.stl"

            z_bottom, z_top = z_heights[i]

            # Create pixel mask (include pixels where this layer has non-zero thickness)
            thickness = z_top - z_bottom
            pixel_mask = (thickness > 0.001) & (self.alpha_mask >= 0.5)  # >0.001mm threshold

            # Calculate approximate layer range for logging
            avg_z_bottom = np.mean(z_bottom[pixel_mask]) if pixel_mask.any() else 0
            avg_z_top = np.mean(z_top[pixel_mask]) if pixel_mask.any() else 0
            layer_start = int(avg_z_bottom / self.layer_height)
            layer_end = int(avg_z_top / self.layer_height)

            logger.info(f"Generating {color_name}: avg layers {layer_start}-{layer_end}, varying thickness")

            # Generate topographical STL
            if pixel_mask.any():
                mesh = self._generate_topographical_stl(z_bottom, z_top, pixel_mask)
                if len(mesh.vertices) > 0:
                    mesh.export(str(output_path))
                    generated_files.append((output_path, filament['name'], layer_start, layer_end))
                    logger.info(f"  Generated: {pixel_mask.sum()} pixels, {len(mesh.vertices)} vertices (topographical)")
                else:
                    logger.warning(f"  Generated empty mesh for {color_name} - skipping")
            else:
                logger.warning(f"  No pixels for {color_name} - skipping")

        return generated_files

    def _generate_with_cap_layers(self, output_base_path):
        """Generate cap layers mode: flat base + colors + negative + flat top

        Structure:
        1. Flat base (darkest): Full layer at bottom (layers 0-1)
        2. Color 1: Shaped areas (layers 2-8)
        3. Color 2: Shaped areas (layers 9-15)
        4. Color 3: Shaped areas (layers 16-22)
        5. Clear negative: Fills gaps NOT covered by colors (layers 2-22)
        6. Flat top (clear): Full layer at top (layers 23-25)

        The negative ensures perfectly flat model throughout.
        """
        logger.info("Generating cap layers mode: flat base + shaped colors + negative + flat top")

        enhanced_grayscale = self._apply_contrast_enhancement(self.image_grayscale.copy())

        generated_files = []

        # Sort middle filaments by luminosity (darkest first = most coverage)
        # Keep base (index 0) and clear (index -1) in their fixed positions
        sorted_filaments = self.selected_filaments.copy()
        sorted_filaments['luminosity'] = sorted_filaments['lab'].apply(
            lambda x: float(np.asarray(x).flat[0]))
        middle = sorted_filaments.iloc[1:-1].sort_values('luminosity').reset_index(drop=True)
        sorted_filaments = pd.concat([
            sorted_filaments.iloc[:1],
            middle,
            sorted_filaments.iloc[-1:]
        ]).reset_index(drop=True)

        # Number of middle color layers (exclude dark base and clear top)
        num_middle_colors = self.num_colors - 2

        # Layer allocation
        base_layers = 2  # Flat base
        top_layers = 2   # Flat top
        middle_layers = self.num_layers - base_layers - top_layers
        layers_per_color = middle_layers // num_middle_colors if num_middle_colors > 0 else 0

        # 1. FLAT BASE (darkest filament)
        darkest_filament = sorted_filaments.iloc[0]
        color_name = darkest_filament['name'].replace(' ', '_').replace('/', '_')
        output_path = output_base_path.parent / f"{output_base_path.stem}_{color_name}_base.stl"

        logger.info(f"Generating flat base: {color_name} (layers 0-{base_layers})")

        # Full flat layer covering all valid pixels
        all_pixels_mask = self.alpha_mask >= 0.5
        mesh = self._generate_flat_layer_stl(all_pixels_mask, 0, base_layers)
        mesh.export(str(output_path))
        generated_files.append((output_path, f"{darkest_filament['name']} (base)", 0, base_layers))

        # Track which pixels are covered by colored layers (for negative calculation)
        combined_color_mask = np.zeros_like(self.image_grayscale, dtype=bool)

        # 2. MIDDLE COLOR LAYERS (shaped)
        for i in range(num_middle_colors):
            filament = sorted_filaments.iloc[i + 1]  # Skip darkest (used for base)
            color_name = filament['name'].replace(' ', '_').replace('/', '_')
            output_path = output_base_path.parent / f"{output_base_path.stem}_{color_name}.stl"

            layer_start = base_layers + (i * layers_per_color)
            layer_end = base_layers + ((i + 1) * layers_per_color)

            # Brightness threshold: colored layers appear on DARK pixels
            # (more colored material = more absorption = darker in transmission)
            # Darkest middle color covers most area, lightest covers least
            # Lightest middle → highest threshold (widest coverage, base tone)
            # Darkest middle → lowest threshold (only darkest pixels, detail)
            brightness_threshold = (i + 1) / (num_middle_colors + 1)

            logger.info(f"Generating {color_name}: layers {layer_start}-{layer_end}, brightness <= {brightness_threshold:.2f}")

            # Mask of pixels that get this color
            # Cover DARK pixels where more absorption is needed
            pixel_mask = enhanced_grayscale <= brightness_threshold
            pixel_mask = pixel_mask & (self.alpha_mask >= 0.5)

            if pixel_mask.any():
                mesh = self._generate_flat_layer_stl(pixel_mask, layer_start, layer_end)
                mesh.export(str(output_path))
                generated_files.append((output_path, filament['name'], layer_start, layer_end))
                logger.info(f"  Generated: {pixel_mask.sum()} pixels")

                # Track for negative calculation
                combined_color_mask = combined_color_mask | pixel_mask
            else:
                logger.warning(f"  No pixels for {color_name} - skipping")

        # 3. CLEAR LAYER (combines negative fill + flat top into single STL)
        clear_filament = sorted_filaments.iloc[-1]  # Lightest/clearest
        color_name = clear_filament['name'].replace(' ', '_').replace('/', '_')
        output_path = output_base_path.parent / f"{output_base_path.stem}_{color_name}.stl"

        middle_layer_start = base_layers
        middle_layer_end = self.num_layers

        logger.info(f"Generating clear layer: {color_name} (layers {middle_layer_start}-{middle_layer_end})")

        # Combine negative (gaps) with flat top
        # Negative = pixels NOT covered by any colored layer (in middle section)
        negative_mask = ~combined_color_mask & (self.alpha_mask >= 0.5)

        # For middle section: use negative mask (fills gaps)
        # For top section: use all pixels (flat top)
        # Generate as single combined mesh
        vertices_list = []
        faces_list = []
        vertex_offset = 0

        # Part 1: Negative fill (middle layers)
        if negative_mask.any():
            mesh_negative = self._generate_flat_layer_stl(negative_mask, middle_layer_start, self.num_layers - top_layers)
            vertices_list.append(mesh_negative.vertices)
            faces_list.append(mesh_negative.faces + vertex_offset)
            vertex_offset += len(mesh_negative.vertices)
            logger.info(f"  Negative fill: {negative_mask.sum()} pixels (fills gaps in layers {middle_layer_start}-{self.num_layers - top_layers})")

        # Part 2: Flat top (top layers)
        mesh_top = self._generate_flat_layer_stl(all_pixels_mask, self.num_layers - top_layers, middle_layer_end)
        vertices_list.append(mesh_top.vertices)
        faces_list.append(mesh_top.faces + vertex_offset)
        logger.info(f"  Flat top: all pixels (layers {self.num_layers - top_layers}-{middle_layer_end})")

        # Combine into single mesh
        if len(vertices_list) > 0:
            combined_vertices = np.vstack(vertices_list)
            combined_faces = np.vstack(faces_list)
            combined_mesh = trimesh.Trimesh(vertices=combined_vertices, faces=combined_faces, process=True)
            combined_mesh.fix_normals()
            combined_mesh.export(str(output_path))
            generated_files.append((output_path, clear_filament['name'], middle_layer_start, middle_layer_end))
            logger.info(f"  Generated combined clear STL")

        return generated_files

    @staticmethod
    def _greedy_mesh_rects(mask):
        """Find maximal rectangles covering all True pixels using greedy meshing

        Scans row by row, merging adjacent same-height pixels into rectangles.
        This dramatically reduces face count compared to per-pixel geometry.

        Args:
            mask: 2D boolean array

        Returns:
            List of (y_start, x_start, y_end, x_end) rectangles (exclusive end)
        """
        if not mask.any():
            return []

        height, width = mask.shape
        # Track which pixels have been consumed
        consumed = np.zeros_like(mask, dtype=bool)
        rects = []

        for y in range(height):
            for x in range(width):
                if not mask[y, x] or consumed[y, x]:
                    continue

                # Extend rectangle rightward as far as possible
                x_end = x + 1
                while x_end < width and mask[y, x_end] and not consumed[y, x_end]:
                    x_end += 1

                # Extend rectangle downward as far as possible (vectorized row check)
                y_end = y + 1
                while y_end < height:
                    row_slice = mask[y_end, x:x_end] & ~consumed[y_end, x:x_end]
                    if not row_slice.all():
                        break
                    y_end += 1

                # Mark consumed
                consumed[y:y_end, x:x_end] = True
                rects.append((y, x, y_end, x_end))

        return rects

    def _build_box_mesh(self, rects, z_bottom_val, z_top_val, pixel_size):
        """Build mesh geometry for a list of rectangles at the same height (vectorized)

        Args:
            rects: List of (y0, x0, y1, x1) rectangles
            z_bottom_val: Z height for bottom face
            z_top_val: Z height for top face
            pixel_size: Size of each pixel in mm

        Returns:
            (vertices_array, faces_array) as numpy arrays
        """
        if not rects:
            return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3).astype(int)

        n = len(rects)
        rects_arr = np.array(rects)  # (n, 4): y0, x0, y1, x1

        # Physical coordinates for all rectangles at once
        px0 = rects_arr[:, 1] * pixel_size
        px1 = rects_arr[:, 3] * pixel_size
        py0 = rects_arr[:, 0] * pixel_size
        py1 = rects_arr[:, 2] * pixel_size
        zb = np.full(n, z_bottom_val)
        zt = np.full(n, z_top_val)

        # Build all 8 vertices per rectangle: (n*8, 3)
        vertices = np.empty((n * 8, 3))
        vertices[0::8] = np.column_stack([px0, py0, zb])   # 0: bottom-front-left
        vertices[1::8] = np.column_stack([px1, py0, zb])   # 1: bottom-front-right
        vertices[2::8] = np.column_stack([px1, py1, zb])   # 2: bottom-back-right
        vertices[3::8] = np.column_stack([px0, py1, zb])   # 3: bottom-back-left
        vertices[4::8] = np.column_stack([px0, py0, zt])   # 4: top-front-left
        vertices[5::8] = np.column_stack([px1, py0, zt])   # 5: top-front-right
        vertices[6::8] = np.column_stack([px1, py1, zt])   # 6: top-back-right
        vertices[7::8] = np.column_stack([px0, py1, zt])   # 7: top-back-left

        # 12 triangles per box, built with broadcasting
        base_faces = np.array([
            [0, 2, 1], [0, 3, 2],  # Bottom (-Z)
            [4, 5, 6], [4, 6, 7],  # Top (+Z)
            [0, 1, 5], [0, 5, 4],  # Front (-Y)
            [2, 3, 7], [2, 7, 6],  # Back (+Y)
            [0, 4, 7], [0, 7, 3],  # Left (-X)
            [1, 2, 6], [1, 6, 5],  # Right (+X)
        ], dtype=np.int64)  # (12, 3)

        offsets = (np.arange(n) * 8)[:, None, None]  # (n, 1, 1)
        faces = (base_faces[None, :, :] + offsets).reshape(-1, 3)

        return vertices, faces

    def _generate_topographical_stl(self, z_bottom, z_top, pixel_mask):
        """Generate topographical STL using shared-vertex grid.

        Each grid corner (H+1 x W+1) gets averaged z values from adjacent active
        pixels. Two vertices per corner (bottom + top) create smooth height-field
        surfaces. Boundary walls close the mesh where active pixels border
        inactive regions or edges.

        Fully vectorized with numpy — no Python loops over pixels.

        Args:
            z_bottom: 2D array of bottom z-heights (mm) for each pixel
            z_top: 2D array of top z-heights (mm) for each pixel
            pixel_mask: 2D boolean array of which pixels to include

        Returns:
            trimesh.Trimesh object (manifold, watertight)
        """
        if not pixel_mask.any():
            return trimesh.Trimesh()

        H, W = pixel_mask.shape
        ps = self.pixel_size

        # Filter degenerate pixels (z_top <= z_bottom)
        effective_mask = pixel_mask & (z_top > z_bottom + 1e-6)
        if not effective_mask.any():
            return trimesh.Trimesh()

        mask_f = effective_mask.astype(np.float64)

        # Compute corner z-values by averaging adjacent active pixels.
        # Pixel (py, px) contributes to corners (py, px), (py, px+1), (py+1, px), (py+1, px+1).
        # Corner grid is (H+1, W+1).
        count = np.zeros((H + 1, W + 1))
        zb_sum = np.zeros((H + 1, W + 1))
        zt_sum = np.zeros((H + 1, W + 1))

        for dy in range(2):
            for dx in range(2):
                count[dy:H + dy, dx:W + dx] += mask_f
                zb_sum[dy:H + dy, dx:W + dx] += z_bottom * mask_f
                zt_sum[dy:H + dy, dx:W + dx] += z_top * mask_f

        corner_active = count > 0
        zb_avg = np.divide(zb_sum, count, where=corner_active, out=np.zeros_like(zb_sum))
        zt_avg = np.divide(zt_sum, count, where=corner_active, out=np.zeros_like(zt_sum))

        # Build vertex arrays: bottom [0, n_active), top [n_active, 2*n_active)
        n_active = int(corner_active.sum())
        vertex_idx = np.full((H + 1, W + 1), -1, dtype=np.int64)
        active_cy, active_cx = np.where(corner_active)
        vertex_idx[active_cy, active_cx] = np.arange(n_active)

        bottom_verts = np.column_stack([active_cx * ps, active_cy * ps, zb_avg[active_cy, active_cx]])
        top_verts = np.column_stack([active_cx * ps, active_cy * ps, zt_avg[active_cy, active_cx]])
        vertices = np.vstack([bottom_verts, top_verts])

        # Get active pixel coordinates
        py, px = np.where(effective_mask)

        # Corner vertex indices for each pixel (bottom layer)
        b00 = vertex_idx[py, px]
        b01 = vertex_idx[py, px + 1]
        b10 = vertex_idx[py + 1, px]
        b11 = vertex_idx[py + 1, px + 1]

        # Top layer indices (offset by n_active)
        t00 = b00 + n_active
        t01 = b01 + n_active
        t10 = b10 + n_active
        t11 = b11 + n_active

        all_faces = []

        # Top surface (+Z normal)
        all_faces.append(np.column_stack([t00, t01, t10]))
        all_faces.append(np.column_stack([t01, t11, t10]))

        # Bottom surface (-Z normal)
        all_faces.append(np.column_stack([b00, b10, b01]))
        all_faces.append(np.column_stack([b01, b10, b11]))

        # Boundary walls: where active pixel borders inactive/edge
        def get_boundary(dy, dx):
            ny, nx = py + dy, px + dx
            oob = (ny < 0) | (ny >= H) | (nx < 0) | (nx >= W)
            inactive = oob | ~effective_mask[np.clip(ny, 0, H - 1), np.clip(nx, 0, W - 1)]
            return inactive

        # Left wall (-X direction): edge at x=px
        left = get_boundary(0, -1)
        if left.any():
            bl0 = vertex_idx[py[left], px[left]]
            bl1 = vertex_idx[py[left] + 1, px[left]]
            tl0 = bl0 + n_active
            tl1 = bl1 + n_active
            all_faces.append(np.column_stack([bl0, bl1, tl0]))
            all_faces.append(np.column_stack([bl1, tl1, tl0]))

        # Right wall (+X direction): edge at x=px+1
        right = get_boundary(0, 1)
        if right.any():
            br0 = vertex_idx[py[right], px[right] + 1]
            br1 = vertex_idx[py[right] + 1, px[right] + 1]
            tr0 = br0 + n_active
            tr1 = br1 + n_active
            all_faces.append(np.column_stack([br0, tr0, br1]))
            all_faces.append(np.column_stack([br1, tr0, tr1]))

        # Front wall (-Y direction): edge at y=py
        front = get_boundary(-1, 0)
        if front.any():
            bf0 = vertex_idx[py[front], px[front]]
            bf1 = vertex_idx[py[front], px[front] + 1]
            tf0 = bf0 + n_active
            tf1 = bf1 + n_active
            all_faces.append(np.column_stack([bf0, tf0, bf1]))
            all_faces.append(np.column_stack([bf1, tf0, tf1]))

        # Back wall (+Y direction): edge at y=py+1
        back = get_boundary(1, 0)
        if back.any():
            bb0 = vertex_idx[py[back] + 1, px[back]]
            bb1 = vertex_idx[py[back] + 1, px[back] + 1]
            tb0 = bb0 + n_active
            tb1 = bb1 + n_active
            all_faces.append(np.column_stack([bb0, bb1, tb0]))
            all_faces.append(np.column_stack([bb1, tb1, tb0]))

        combined_faces = np.vstack(all_faces)
        mesh = trimesh.Trimesh(vertices=vertices, faces=combined_faces, process=False)
        mesh.fix_normals()

        logger.info(f"  Shared-vertex grid: {effective_mask.sum():,} pixels, "
                     f"{n_active} corners -> {len(mesh.faces):,} faces")

        return mesh

    def _generate_flat_layer_stl(self, pixel_mask, layer_start, layer_end):
        """Generate flat STL for given pixel mask at specific layer range

        Uses greedy meshing to merge adjacent pixels into larger rectangles,
        dramatically reducing face count for flat layers.

        Args:
            pixel_mask: 2D boolean array of which pixels to include
            layer_start: Starting layer number
            layer_end: Ending layer number

        Returns:
            trimesh.Trimesh object
        """
        z_bottom = layer_start * self.layer_height
        z_top = layer_end * self.layer_height

        # All pixels at the same height - single greedy mesh pass
        rects = self._greedy_mesh_rects(pixel_mask)
        verts, faces = self._build_box_mesh(rects, z_bottom, z_top, self.pixel_size)

        if len(verts) == 0:
            return trimesh.Trimesh()

        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
        mesh.fix_normals()

        original_pixels = pixel_mask.sum()
        logger.info(f"    Flat layer: {original_pixels:,} pixels -> {len(rects)} rects, {len(faces)} faces")

        return mesh

    def _quantize_thicknesses_to_layers(self, thickness_maps, max_layers_per_color):
        """Convert continuous Beer-Lambert thickness maps to integer layer counts

        Args:
            thickness_maps: List of 2D arrays with thickness in mm per pixel per color
            max_layers_per_color: List of max layers available for each color

        Returns:
            List of 2D int arrays (one per color) with layer counts per pixel
        """
        layer_counts = []
        for i, t_map in enumerate(thickness_maps):
            raw_layers = t_map / self.layer_height
            quantized = np.round(raw_layers).astype(np.int32)
            quantized = np.clip(quantized, 0, max_layers_per_color[i])
            layer_counts.append(quantized)
        return layer_counts

    def _build_assignment_matrix(self, layer_counts, num_total_layers):
        """Build 3D voxel array: assignment[y, x, z] = filament_index

        Stacks all colors bottom-to-top. Remaining layers padded with first
        (lightest) filament to minimize color impact on transmission.

        Args:
            layer_counts: List of 2D int arrays (per-color layer counts)
            num_total_layers: Total number of Z-layers

        Returns:
            3D int8 array of shape (H, W, num_total_layers)
        """
        H, W = layer_counts[0].shape
        assignment = np.full((H, W, num_total_layers), -1, dtype=np.int8)

        # Stack ALL colors bottom-to-top
        z_cursor = np.zeros((H, W), dtype=np.int32)

        for color_idx in range(len(layer_counts)):
            count = layer_counts[color_idx]
            # For each pixel, fill z_cursor to z_cursor+count with this color
            for z_offset in range(int(count.max())):
                mask = z_offset < count
                z_pos = z_cursor + z_offset
                # Only assign where z_pos is valid
                valid = mask & (z_pos < num_total_layers)
                if valid.any():
                    y_idx, x_idx = np.where(valid)
                    assignment[y_idx, x_idx, z_pos[valid]] = color_idx
            z_cursor += count

        # Pad remaining with first (lightest) filament — minimal color impact
        for z in range(num_total_layers):
            unfilled = assignment[:, :, z] == -1
            if unfilled.any():
                assignment[unfilled, z] = 0

        return assignment

    def _generate_face_down_stls(self, assignment, sorted_filaments, output_base_path):
        """Convert assignment matrix to STL files (one per filament)

        For each filament, iterates Z-layers, merges consecutive layers with
        identical pixel masks, and uses greedy meshing for geometry.

        Args:
            assignment: 3D int8 array (H, W, num_layers)
            sorted_filaments: DataFrame of filaments
            output_base_path: Base path for output files

        Returns:
            List of (path, name, layer_start, layer_end) tuples
        """
        H, W, num_layers = assignment.shape
        generated_files = []

        for color_idx in range(len(sorted_filaments)):
            filament = sorted_filaments.iloc[color_idx]
            color_name = filament['name'].replace(' ', '_').replace('/', '_')
            output_path = output_base_path.parent / f"{output_base_path.stem}_{color_name}.stl"

            # Collect all Z-ranges for this filament with vertical run merging
            all_verts = []
            all_faces = []
            vertex_offset = 0
            total_rects = 0

            # Cache: mask hash -> greedy mesh rects
            mask_cache = {}

            z = 0
            layer_start_actual = None
            layer_end_actual = 0

            while z < num_layers:
                mask = assignment[:, :, z] == color_idx
                if not mask.any():
                    z += 1
                    continue

                if layer_start_actual is None:
                    layer_start_actual = z

                # Find consecutive layers with identical mask
                z_start = z
                z_end = z + 1
                while z_end < num_layers:
                    next_mask = assignment[:, :, z_end] == color_idx
                    if np.array_equal(mask, next_mask):
                        z_end += 1
                    else:
                        break

                layer_end_actual = z_end

                # Check cache for this mask pattern
                mask_key = mask.tobytes()
                if mask_key in mask_cache:
                    rects = mask_cache[mask_key]
                else:
                    rects = self._greedy_mesh_rects(mask)
                    mask_cache[mask_key] = rects

                if rects:
                    z_bottom_val = z_start * self.layer_height
                    z_top_val = z_end * self.layer_height
                    verts, faces = self._build_box_mesh(rects, z_bottom_val, z_top_val, self.pixel_size)
                    if len(verts) > 0:
                        all_verts.append(verts)
                        all_faces.append(faces + vertex_offset)
                        vertex_offset += len(verts)
                        total_rects += len(rects)

                z = z_end

            if all_verts:
                combined_verts = np.vstack(all_verts)
                combined_faces = np.vstack(all_faces)
                mesh = trimesh.Trimesh(vertices=combined_verts, faces=combined_faces, process=True)
                mesh.fix_normals()
                mesh.export(str(output_path))

                if layer_start_actual is None:
                    layer_start_actual = 0

                generated_files.append((output_path, filament['name'], layer_start_actual, layer_end_actual))
                logger.info(f"  {color_name}: {total_rects} rects, {len(combined_faces)} faces, "
                             f"layers {layer_start_actual}-{layer_end_actual}, watertight={mesh.is_watertight}")
            else:
                logger.warning(f"  {color_name}: no geometry generated")

        return generated_files

    def _render_face_down_preview(self, pixel_height, z_boundaries, sorted_filaments):
        """Render Beer-Lambert preview of face-down print as viewed from bed side

        Light enters from z=max (back/top of print, darkest filament), passes through
        each color band based on per-pixel thickness, and exits at z=0 (bed side).

        Args:
            pixel_height: 2D array of per-pixel heights (mm)
            z_boundaries: 1D array of color band boundaries (mm)
            sorted_filaments: DataFrame of filaments (light-to-dark order)

        Returns:
            HxWx3 RGB preview image (0-1 range)
        """
        H, W = pixel_height.shape
        num_colors = len(sorted_filaments)

        filament_rgbs = np.array([f['rgb'] for _, f in sorted_filaments.iterrows()])
        filament_tds = np.array([f['transmission_distance'] for _, f in sorted_filaments.iterrows()])

        # After user flips model: light enters from z=0 (darkest) through to z=max (lightest)
        light = np.ones((H, W, 3))  # White backlight from behind

        # Process from lightest (highest z, viewing surface after flip) to darkest (lowest z, back after flip)
        for i in range(num_colors):
            z_lo = z_boundaries[i]
            z_hi = z_boundaries[i + 1]

            # Per-pixel thickness in this color band
            thickness = np.clip(pixel_height, z_lo, z_hi) - z_lo
            td = max(filament_tds[i], 0.1)
            rgb = filament_rgbs[i]

            # Beer-Lambert: transmission fraction per pixel
            transmission = np.exp(-thickness / td)

            for c in range(3):
                light[:, :, c] = light[:, :, c] * transmission + rgb[c] * (1.0 - transmission)

        return np.clip(light, 0, 1)

    def _render_cap_layer_preview(self, enhanced_grayscale, sorted_filaments):
        """Render Beer-Lambert preview for cap-layer mode

        Cap layers use binary brightness thresholds (pixel has a color or not),
        unlike standard/face-down which use continuous heightmaps.

        Structure: flat base (darkest) + shaped middle colors + clear fill + flat top (lightest)

        Args:
            enhanced_grayscale: 2D array of contrast-enhanced grayscale values
            sorted_filaments: DataFrame of filaments sorted dark-to-light

        Returns:
            HxWx3 RGB preview image (0-1 range)
        """
        H, W = enhanced_grayscale.shape
        num_colors = len(sorted_filaments)
        num_middle_colors = num_colors - 2

        filament_rgbs = np.array([f['rgb'] for _, f in sorted_filaments.iterrows()])
        filament_tds = np.array([f['transmission_distance'] for _, f in sorted_filaments.iterrows()])

        # Layer allocation (matches _generate_with_cap_layers)
        base_layers = 2
        top_layers = 2
        middle_layers = self.num_layers - base_layers - top_layers
        layers_per_color = middle_layers // num_middle_colors if num_middle_colors > 0 else 0

        base_thickness = base_layers * self.layer_height
        middle_thickness = layers_per_color * self.layer_height
        top_thickness = top_layers * self.layer_height

        # Start with white backlight
        light = np.ones((H, W, 3))

        # Use subtractive (multiplicative) filter model for cap layers.
        # The opaque base reduces light to ~20%. With additive model,
        # colored middles would ADD brightness (filament_rgb > remaining light),
        # causing inversion. Subtractive model ensures light only decreases.

        # 1. Base layer (darkest filament) — always present, uniform thickness
        td = max(filament_tds[0], 0.1)
        transmission = np.exp(-base_thickness / td)
        for c in range(3):
            light[:, :, c] = light[:, :, c] * (transmission + filament_rgbs[0][c] * (1.0 - transmission))

        # 2. Middle colors — present where enhanced_grayscale <= threshold
        clear_td = max(filament_tds[-1], 0.1)
        clear_rgb = filament_rgbs[-1]

        for i in range(num_middle_colors):
            threshold = (i + 1) / (num_middle_colors + 1)
            present = enhanced_grayscale <= threshold

            td = max(filament_tds[i + 1], 0.1)
            rgb = filament_rgbs[i + 1]

            # Where color is present: subtractive filter by this color
            color_trans = np.exp(-middle_thickness / td)
            # Where color is absent: subtractive filter by clear fill
            clear_trans = np.exp(-middle_thickness / clear_td)

            for c in range(3):
                colored = light[:, :, c] * (color_trans + rgb[c] * (1.0 - color_trans))
                cleared = light[:, :, c] * (clear_trans + clear_rgb[c] * (1.0 - clear_trans))
                light[:, :, c] = np.where(present, colored, cleared)

        # 3. Clear top layer — always present, uniform thickness
        transmission = np.exp(-top_thickness / clear_td)
        for c in range(3):
            light[:, :, c] = light[:, :, c] * (transmission + clear_rgb[c] * (1.0 - transmission))

        return np.clip(light, 0, 1)

    def _generate_face_down(self, output_base_path):
        """Generate face-down mode using topographical approach with inverted heightmap

        Uses the same proven pipeline as standard mode:
        1. Contrast enhancement
        2. TD-proportional layer allocation
        3. Global heightmap (inverted: dark=tall, light=short)
        4. Topographical STL per color band

        Filaments sorted dark-to-light (same as standard mode).
        z=0 (bed) = darkest, z=max = lightest.
        User flips model in slicer; viewed from top, dark pixels have more material → more absorption.

        Returns:
            List of (path, name, layer_start, layer_end) tuples
        """
        from scipy import ndimage

        logger.info("=" * 60)
        logger.info("FACE-DOWN MODE: Topographical generation (inverted heightmap)")
        logger.info("=" * 60)

        generated_files = []

        # Sort dark-to-light by LAB luminosity (same as standard mode)
        sorted_filaments = self.selected_filaments.copy()
        sorted_filaments['luminosity'] = sorted_filaments['lab'].apply(
            lambda x: float(np.asarray(x).flat[0]))
        sorted_filaments = sorted_filaments.sort_values('luminosity').reset_index(drop=True)

        # 1. Contrast enhancement (same as standard)
        enhanced_grayscale = self._apply_contrast_enhancement(self.image_grayscale.copy())

        # 2. TD-proportional layer allocation (same as standard)
        filament_tds = np.array([f['transmission_distance'] for _, f in sorted_filaments.iterrows()])

        td_weights = np.log1p(np.maximum(filament_tds, 1.0))
        td_weights = td_weights / td_weights.sum()

        min_layers = 1
        available_layers = self.num_layers - min_layers * self.num_colors
        if available_layers < 0:
            available_layers = 0
            min_layers = max(1, self.num_layers // self.num_colors)

        layer_counts = np.round(td_weights * available_layers).astype(int) + min_layers
        layer_counts[-1] = self.num_layers - layer_counts[:-1].sum()

        layer_boundaries = np.concatenate([[0], np.cumsum(layer_counts)]).astype(int)

        for idx in range(self.num_colors):
            name = sorted_filaments.iloc[idx]['name']
            td = filament_tds[idx]
            lc = layer_counts[idx]
            height = lc * self.layer_height
            logger.info(f"  {name}: TD={td:.1f}mm → {lc} layers ({height:.2f}mm)")

        # 3. INVERTED heightmap: dark pixels → tall, light pixels → short
        alpha_pixels = self.alpha_mask >= 0.5
        global_brightness_max = np.max(enhanced_grayscale[alpha_pixels]) if np.any(alpha_pixels) else 1.0

        min_height = 2 * self.layer_height  # bed adhesion minimum
        normalized = enhanced_grayscale / max(global_brightness_max, 1e-6)
        pixel_height = min_height + (1.0 - normalized) * (self.model_height - min_height)
        pixel_height = np.clip(pixel_height, min_height, self.model_height)
        pixel_height = np.where(alpha_pixels, pixel_height, 0)

        # Fixed color boundary z-values
        z_boundaries = layer_boundaries * self.layer_height

        # Store for preview
        self.face_down_pixel_height = pixel_height
        self.face_down_z_boundaries = z_boundaries
        self.face_down_filaments = sorted_filaments

        min_thickness = self.layer_height * 0.5

        # 4. Generate topographical STL for each color band
        for i, filament in sorted_filaments.iterrows():
            color_name = filament['name'].replace(' ', '_').replace('/', '_')

            layer_start = layer_boundaries[i]
            layer_end = layer_boundaries[i + 1]

            z_bottom_flat = float(z_boundaries[i])
            z_top_boundary = float(z_boundaries[i + 1])

            # Pixels that reach into this color's z-band
            pixel_mask = (pixel_height > z_bottom_flat + min_thickness) & alpha_pixels

            pixel_count = np.sum(pixel_mask)

            # Clean up small disconnected regions
            labeled, num_regions = ndimage.label(pixel_mask)
            min_region_size = 8

            if num_regions > 1 and i > 0:
                region_sizes = np.bincount(labeled.ravel())
                small_regions = region_sizes < min_region_size
                small_regions[0] = False
                pixel_mask[small_regions[labeled]] = False

                pixel_count_filtered = np.sum(pixel_mask)
                removed_pixels = pixel_count - pixel_count_filtered
                if removed_pixels > 0:
                    logger.info(f"  Filtered {removed_pixels} pixels in {np.sum(small_regions) - 1} small regions (< {min_region_size} pixels)")
                pixel_count = pixel_count_filtered

            logger.info(f"  {color_name}: layers {layer_start}-{layer_end} ({layer_end - layer_start} layers, {pixel_count} px)")

            if pixel_count > 0:
                z_top_color = np.clip(pixel_height, z_bottom_flat, z_top_boundary)
                z_bottom_color = np.full_like(pixel_height, z_bottom_flat)

                thickness = z_top_color - z_bottom_flat
                color_effective_mask = pixel_mask & (thickness >= min_thickness)

                pixels_after_filter = np.sum(color_effective_mask)
                if pixels_after_filter < pixel_count:
                    logger.debug(f"    Filtered {pixel_count - pixels_after_filter} thin pixels")

                if color_effective_mask.any():
                    z_top_max = np.max(z_top_color[color_effective_mask])
                    logger.info(f"    z: {z_bottom_flat:.2f} - {z_top_max:.2f}mm")

                output_path = output_base_path.parent / f"{output_base_path.stem}_{color_name}.stl"
                mesh = self._generate_topographical_stl(z_bottom_color, z_top_color, color_effective_mask)

                if len(mesh.vertices) > 0:
                    mesh.export(str(output_path))
                    max_z_top = np.max(z_top_color[color_effective_mask])
                    layer_start_actual = int(np.floor(z_bottom_flat / self.layer_height))
                    layer_end_actual = int(np.ceil(max_z_top / self.layer_height))
                    generated_files.append((output_path, filament['name'], layer_start_actual, layer_end_actual))
                    logger.info(f"  Generated: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
                else:
                    logger.warning(f"  Empty mesh for {color_name} - skipping")
            else:
                logger.warning(f"  No pixels for {color_name} - skipping")

        logger.info(f"Face-down generation complete: {len(generated_files)} STL files")
        return generated_files
