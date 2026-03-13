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
                 contrast_strength=2.0, use_face_down=False, use_exploded=False,
                 use_exploded_multi=False, use_exploded_cmyk=False, sandwich_layers=1,
                 use_fill=False, base_layers=3, max_color_sandwiches=None):
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
            use_exploded: If True, generate standalone single-color sandwiches
            use_exploded_multi: If True, generate multi-color sandwiches (up to 3 colors each)
            max_color_sandwiches: Max sandwiches per colour in exploded modes (None = use mode default)
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
        self.use_exploded = use_exploded
        self.use_exploded_multi = use_exploded_multi
        self.use_exploded_cmyk = use_exploded_cmyk
        self.sandwich_layers = sandwich_layers
        self.use_fill = use_fill
        self.base_layers = base_layers
        self.max_color_sandwiches = max_color_sandwiches
        self.contrast_strength = contrast_strength

        self.num_layers = int(model_height / layer_height)
        self.num_colors = len(selected_filaments)

        # Calculate pixel size
        height_px, width_px = image_grayscale.shape
        self.pixel_size = width_mm / width_px

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

    def _sort_filaments_by_luminosity(self, cap_layers_sort=False):
        """Sort filaments by LAB luminosity (dark to light).

        Args:
            cap_layers_sort: If True, keep first (base) and last (clear) in place,
                             sort only middle filaments by luminosity.

        Returns:
            DataFrame of sorted filaments (reset index)
        """
        sorted_filaments = self.selected_filaments.copy()
        sorted_filaments['luminosity'] = sorted_filaments['lab'].apply(
            lambda x: float(np.asarray(x).flat[0]))

        if cap_layers_sort:
            middle = sorted_filaments.iloc[1:-1].sort_values('luminosity').reset_index(drop=True)
            sorted_filaments = pd.concat([
                sorted_filaments.iloc[:1],
                middle,
                sorted_filaments.iloc[-1:]
            ]).reset_index(drop=True)
        else:
            sorted_filaments = sorted_filaments.sort_values('luminosity').reset_index(drop=True)

        return sorted_filaments

    def _allocate_layers_td_proportional(self, filament_tds, total_layers):
        """Allocate layers proportionally to TD using log-dampened weighting.

        Args:
            filament_tds: 1D array of transmission distances
            total_layers: Total number of layers to allocate

        Returns:
            (layer_counts, layer_boundaries, z_boundaries) where:
            - layer_counts: int array of layers per color
            - layer_boundaries: int array of cumulative layer boundaries [0, ..., total_layers]
            - z_boundaries: float array of z-heights at boundaries (mm)
        """
        num_colors = len(filament_tds)
        td_weights = np.log1p(np.maximum(filament_tds, 1.0))
        td_weights = td_weights / td_weights.sum()

        min_layers = 1
        available_layers = total_layers - min_layers * num_colors
        if available_layers < 0:
            available_layers = 0
            min_layers = max(1, total_layers // num_colors)

        layer_counts = np.round(td_weights * available_layers).astype(int) + min_layers
        layer_counts[-1] = total_layers - layer_counts[:-1].sum()

        layer_boundaries = np.concatenate([[0], np.cumsum(layer_counts)]).astype(int)
        z_boundaries = layer_boundaries * self.layer_height

        return layer_counts, layer_boundaries, z_boundaries

    def _compute_heightmap(self, enhanced_grayscale, alpha_pixels, max_height, invert=False):
        """Compute per-pixel height from contrast-enhanced grayscale.

        Args:
            enhanced_grayscale: 2D array of contrast-enhanced brightness (0-1)
            alpha_pixels: 2D boolean mask of valid pixels
            max_height: Maximum height in mm
            invert: If True, dark pixels are tall (for face-down mode)

        Returns:
            2D array of per-pixel heights in mm (0 for transparent pixels)
        """
        global_brightness_max = np.max(enhanced_grayscale[alpha_pixels]) if np.any(alpha_pixels) else 1.0
        min_height = 2 * self.layer_height
        normalized = enhanced_grayscale / max(global_brightness_max, 1e-6)

        if invert:
            pixel_height = min_height + (1.0 - normalized) * (max_height - min_height)
        else:
            pixel_height = min_height + normalized * (max_height - min_height)

        pixel_height = np.clip(pixel_height, min_height, max_height)
        pixel_height = np.where(alpha_pixels, pixel_height, 0)
        return pixel_height

    def _generate_color_band_stls(self, sorted_filaments, pixel_height, z_boundaries,
                                   layer_boundaries, alpha_pixels, output_base_path,
                                   collect_meshes=False):
        """Generate one quantized STL per color band.

        Iterates colors, clips pixel height to each band, cleans small regions,
        and generates a quantized STL.

        Args:
            sorted_filaments: DataFrame of filaments (sorted)
            pixel_height: 2D array of per-pixel heights (mm)
            z_boundaries: 1D float array of z-heights at band boundaries
            layer_boundaries: 1D int array of layer indices at boundaries
            alpha_pixels: 2D boolean mask of valid pixels
            output_base_path: Path for output files
            collect_meshes: If True, return list of (mesh, path, name) instead of exporting

        Returns:
            List of (path, name, layer_start, layer_end) tuples, or
            list of (mesh, path, name) tuples if collect_meshes=True
        """
        from scipy import ndimage

        generated_files = []
        mesh_outputs = []
        min_thickness = self.layer_height * 0.5

        for i in range(len(sorted_filaments)):
            filament = sorted_filaments.iloc[i]
            color_name = filament['name'].replace(' ', '_').replace('/', '_')

            layer_start = int(layer_boundaries[i])
            layer_end = int(layer_boundaries[i + 1])

            z_bottom_flat = float(z_boundaries[i])
            z_top_boundary = float(z_boundaries[i + 1])

            pixel_mask = (pixel_height > z_bottom_flat + min_thickness) & alpha_pixels
            pixel_count = int(np.sum(pixel_mask))

            # Clean up small disconnected regions
            labeled, num_regions = ndimage.label(pixel_mask)
            min_region_size = 8

            if num_regions > 1 and i > 0:
                region_sizes = np.bincount(labeled.ravel())
                small_regions = region_sizes < min_region_size
                small_regions[0] = False
                pixel_mask[small_regions[labeled]] = False

                pixel_count_filtered = int(np.sum(pixel_mask))
                removed_pixels = pixel_count - pixel_count_filtered
                if removed_pixels > 0:
                    logger.info(f"  Filtered {removed_pixels} pixels in "
                                 f"{int(np.sum(small_regions)) - 1} small regions")
                pixel_count = pixel_count_filtered

            logger.info(f"  {color_name}: layers {layer_start}-{layer_end} "
                         f"({layer_end - layer_start} layers, {pixel_count} px)")

            if pixel_count > 0:
                z_top_color = np.clip(pixel_height, z_bottom_flat, z_top_boundary)
                z_bottom_color = np.full_like(pixel_height, z_bottom_flat)

                thickness = z_top_color - z_bottom_flat
                color_effective_mask = pixel_mask & (thickness >= min_thickness)

                if color_effective_mask.any():
                    z_top_max = np.max(z_top_color[color_effective_mask])
                    logger.info(f"    z: {z_bottom_flat:.2f} - {z_top_max:.2f}mm")

                    output_path = output_base_path.parent / f"{output_base_path.stem}_{color_name}.stl"
                    mesh = self._generate_quantized_stl(z_bottom_color, z_top_color, color_effective_mask)

                    if len(mesh.vertices) > 0:
                        if collect_meshes:
                            mesh_outputs.append((mesh, output_path, filament['name']))
                        else:
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

        if collect_meshes:
            return mesh_outputs
        return generated_files

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

    def generate_all(self, output_base_path):
        """Generate STL files for the configured mode.

        Pure dispatcher — delegates to the appropriate mode-specific method.

        Args:
            output_base_path: Path for output files

        Returns:
            List of (path, name, layer_start, layer_end) tuples
        """
        logger.info(f"Generating {self.num_colors}-color model: {self.model_height}mm tall, {self.num_layers} layers")

        if self.use_exploded_cmyk:
            return self._generate_exploded_cmyk(output_base_path)
        if self.use_exploded_multi:
            return self._generate_exploded_multi(output_base_path)
        if self.use_exploded:
            return self._generate_exploded(output_base_path)
        if self.use_face_down and self.use_cap_layers:
            return self._generate_face_down_with_cap_layers(output_base_path)
        if self.use_face_down:
            return self._generate_face_down(output_base_path)
        if self.use_cap_layers:
            return self._generate_with_cap_layers(output_base_path)
        return self._generate_standard(output_base_path)

    def _generate_standard(self, output_base_path):
        """Generate standard topographical STLs with fixed color boundaries.

        Uses global heightmap (brightness -> height) with TD-proportional layer
        allocation and quantized STL generation per color band.
        """
        sorted_filaments = self._sort_filaments_by_luminosity()
        enhanced_grayscale = self._apply_contrast_enhancement(self.image_grayscale.copy())
        alpha_pixels = self.alpha_mask >= 0.5

        filament_tds = np.array([f['transmission_distance'] for _, f in sorted_filaments.iterrows()])
        layer_counts, layer_boundaries, z_boundaries = self._allocate_layers_td_proportional(
            filament_tds, self.num_layers)

        for idx in range(self.num_colors):
            name = sorted_filaments.iloc[idx]['name']
            lc = layer_counts[idx]
            logger.info(f"  {name}: TD={filament_tds[idx]:.1f}mm -> {lc} layers ({lc * self.layer_height:.2f}mm)")

        pixel_height = self._compute_heightmap(enhanced_grayscale, alpha_pixels, self.model_height)

        return self._generate_color_band_stls(
            sorted_filaments, pixel_height, z_boundaries, layer_boundaries,
            alpha_pixels, output_base_path)

    def _compute_exploded_layer_counts(self, color_filaments, transparent_filament,
                                       max_layers_per_pixel, per_color_caps):
        """Compute optimal integer layer counts per pixel per color for exploded mode.

        Uses Beer-Lambert simulation over all valid integer layer-count combinations,
        builds a KDTree in LAB space, and finds the nearest match for each pixel.

        Args:
            color_filaments: DataFrame of color filaments (excludes transparent)
            transparent_filament: Series for the transparent filament
            max_layers_per_pixel: int, total layer budget per pixel
            per_color_caps: list of int, max sandwiches allocated to each color

        Returns:
            layer_counts: ndarray (H, W, num_color_filaments) int32
        """
        from itertools import product
        from scipy.spatial import cKDTree

        num_colors = len(color_filaments)
        H, W = self.image_grayscale.shape

        # Filament properties: colors + transparent
        filament_rgbs = np.array([f['rgb'] for _, f in color_filaments.iterrows()])
        filament_tds = np.array([f['transmission_distance'] for _, f in color_filaments.iterrows()])
        trans_rgb = np.array(transparent_filament['rgb'])
        trans_td = transparent_filament['transmission_distance']

        all_rgbs = np.vstack([filament_rgbs, trans_rgb.reshape(1, 3)])
        all_tds = np.append(filament_tds, trans_td)

        caps_str = ', '.join(f"{color_filaments.iloc[i]['name']}={per_color_caps[i]}"
                             for i in range(num_colors))
        logger.info(f"Exploded optimization: {num_colors} colors, {max_layers_per_pixel} total sandwiches")
        logger.info(f"  Allocation: {caps_str}")

        # Generate all valid layer count combinations respecting per-color caps
        ranges = [range(0, cap + 1) for cap in per_color_caps]
        all_combos = np.array(list(product(*ranges)))

        # Filter: total layers <= budget
        total_layers = all_combos.sum(axis=1)
        valid = total_layers <= max_layers_per_pixel
        combos = all_combos[valid]

        logger.info(f"  {len(combos)} valid combinations (from {len(all_combos)} total)")

        # Compute transparent layer count and convert to thicknesses
        # Each sandwich has sandwich_layers color layers in the middle
        color_thickness = self.layer_height * self.sandwich_layers
        trans_layers = max_layers_per_pixel - combos.sum(axis=1)
        color_thicknesses = combos * color_thickness
        # With fill: unused sandwiches are fully transparent (same thickness as color)
        # Without fill: unused sandwiches only have bottom transparent layer
        trans_per_sandwich = color_thickness if self.use_fill else self.layer_height
        trans_thicknesses = (trans_layers * trans_per_sandwich).reshape(-1, 1)
        thickness_combos = np.column_stack([color_thicknesses, trans_thicknesses])

        # Simulate Beer-Lambert for each combination
        combo_colors_rgb = self._vectorized_beer_lambert(all_rgbs, all_tds, thickness_combos)
        combo_colors_lab = color.rgb2lab(combo_colors_rgb.reshape(-1, 1, 3)).reshape(-1, 3)

        # Build KDTree for fast matching
        tree = cKDTree(combo_colors_lab)

        # Target pixel colors in LAB
        if self.image_rgb is not None:
            target_lab = color.rgb2lab(self.image_rgb).reshape(-1, 3)
        else:
            gray_rgb = np.stack([self.image_grayscale] * 3, axis=-1)
            target_lab = color.rgb2lab(gray_rgb).reshape(-1, 3)

        # K=1 nearest neighbor (need integer results, not interpolated)
        distances, indices = tree.query(target_lab, k=1)

        # Log matching quality
        valid_mask = (self.alpha_mask >= 0.5).ravel()
        valid_distances = distances[valid_mask]
        logger.info(f"  Beer-Lambert matching: mean deltaE={np.mean(valid_distances):.2f}, "
                     f"median={np.median(valid_distances):.2f}, "
                     f"95th={np.percentile(valid_distances, 95):.2f}")

        # Map back to layer counts
        pixel_combos = combos[indices]
        layer_counts = pixel_combos.reshape(H, W, num_colors).astype(np.int32)

        # Zero out transparent pixels
        alpha_mask_3d = (self.alpha_mask >= 0.5)[:, :, np.newaxis]
        layer_counts = np.where(alpha_mask_3d, layer_counts, 0)

        # Log per-color statistics
        total_sandwiches = 0
        for c in range(num_colors):
            name = color_filaments.iloc[c]['name']
            active = layer_counts[:, :, c][self.alpha_mask >= 0.5]
            max_k = int(layer_counts[:, :, c].max())
            mean_k = float(active.mean()) if len(active) > 0 else 0
            total_sandwiches += max_k
            logger.info(f"  {name}: {max_k} sandwiches (allocated {per_color_caps[c]}), "
                         f"mean {mean_k:.1f} layers/pixel")

        logger.info(f"  Total sandwiches: {total_sandwiches} ({total_sandwiches * 2} STL files)")

        return layer_counts

    def _generate_exploded_cmyk(self, output_base_path):
        """Generate exploded CMYK mode: 4 primaries with configurable sandwiches each."""
        cap = self.max_color_sandwiches if self.max_color_sandwiches is not None else 1
        return self._generate_exploded(output_base_path, max_sandwiches_per_color=cap)

    def _generate_exploded(self, output_base_path, max_sandwiches_per_color=3):
        """Generate exploded mode: per-pixel Beer-Lambert optimized sandwiches.

        Each color is capped at max_sandwiches_per_color sandwiches, allowing
        intensity control (0..N layers per pixel per color).
        Total sandwiches used <= model_height / layer_height.

        Each sandwich is 3 layers tall (transparent/color/transparent).
        For each color, generates one sandwich per demand level:
          - Sandwich k has mask = pixels needing >= k layers of that color
          - Produces 2 STLs: color pixels + transparent carrier

        Final stacked height = total_sandwiches * 3 * layer_height.
        """
        # Allow instance-level override
        if self.max_color_sandwiches is not None and not self.use_exploded_cmyk:
            max_sandwiches_per_color = self.max_color_sandwiches
        generated_files = []

        # Separate color vs transparent filaments (transparent is last)
        num_color_filaments = self.num_colors - 1
        if num_color_filaments <= 0:
            logger.warning("Exploded mode requires at least 1 color + transparent")
            return generated_files

        color_filaments = self.selected_filaments.iloc[:num_color_filaments]
        transparent_filament = self.selected_filaments.iloc[-1]

        # Fixed total sandwich count
        total_sandwiches = int(self.model_height / self.layer_height)
        logger.info(f"Exploded mode: {total_sandwiches} total sandwiches, {num_color_filaments} colors")

        per_color_caps = [min(max_sandwiches_per_color, total_sandwiches) for _ in range(num_color_filaments)]

        # Reduce caps if combinatorial space is too large
        combo_size = 1
        for cap in per_color_caps:
            combo_size *= (cap + 1)
        while combo_size > 500_000:
            # Reduce the largest cap by 1
            max_idx = per_color_caps.index(max(per_color_caps))
            per_color_caps[max_idx] -= 1
            combo_size = 1
            for cap in per_color_caps:
                combo_size *= (cap + 1)

        # Compute optimal layer counts per pixel per color
        layer_counts = self._compute_exploded_layer_counts(
            color_filaments, transparent_filament, total_sandwiches, per_color_caps
        )

        all_pixels_mask = self.alpha_mask >= 0.5

        # Sandwich layer structure:
        # Bottom transparent: layer 0 to base_layers
        # Color middle: layer base_layers to base_layers+sandwich_layers
        # Top transparent (fill only): layer base_layers+sandwich_layers to base_layers+sandwich_layers+1
        bl = self.base_layers
        sl = self.sandwich_layers
        layers_per_sandwich = bl + sl + (1 if self.use_fill else 0)

        # Pre-generate shared full-plate meshes (same for every sandwich)
        mesh_bottom = self._generate_flat_layer_stl(all_pixels_mask, 0, bl)
        mesh_top = self._generate_flat_layer_stl(all_pixels_mask, bl + sl, bl + sl + 1) if self.use_fill else None

        for c in range(num_color_filaments):
            filament = color_filaments.iloc[c]
            color_name = filament['name'].replace(' ', '_').replace('/', '_')
            max_k = int(layer_counts[:, :, c].max())

            for k in range(1, max_k + 1):
                # Pixels needing >= k layers of this color
                color_mask = (layer_counts[:, :, c] >= k) & all_pixels_mask
                inverse_mask = ~color_mask & all_pixels_mask

                pixel_count = int(color_mask.sum())
                if pixel_count == 0:
                    continue

                logger.info(f"  {color_name} sandwich {k}/{max_k}: {pixel_count} pixels "
                             f"({layers_per_sandwich} layers/sandwich, {sl} color, fill={self.use_fill})")

                # Color STL: middle layers
                suffix = f"_{k}" if max_k > 1 else ""
                color_stl_path = output_base_path.parent / f"{output_base_path.stem}_{color_name}{suffix}_color.stl"
                color_mesh = self._generate_flat_layer_stl(color_mask, bl, bl + sl)
                if len(color_mesh.vertices) > 0:
                    color_mesh.export(str(color_stl_path))
                    generated_files.append((color_stl_path, f"{filament['name']} (color {k})", bl, bl + sl))

                # Transparent STL: bottom + optional inverse middle + optional top
                trans_stl_path = output_base_path.parent / f"{output_base_path.stem}_{color_name}{suffix}_transparent.stl"

                vertices_list = []
                faces_list = []
                vertex_offset = 0

                # Part 1: Full bottom (base_layers thick)
                if len(mesh_bottom.vertices) > 0:
                    vertices_list.append(mesh_bottom.vertices)
                    faces_list.append(mesh_bottom.faces + vertex_offset)
                    vertex_offset += len(mesh_bottom.vertices)

                if self.use_fill:
                    # Part 2: Inverse middle fill
                    if inverse_mask.any():
                        mesh_middle = self._generate_flat_layer_stl(inverse_mask, bl, bl + sl)
                        if len(mesh_middle.vertices) > 0:
                            vertices_list.append(mesh_middle.vertices)
                            faces_list.append(mesh_middle.faces + vertex_offset)
                            vertex_offset += len(mesh_middle.vertices)

                    # Part 3: Full top
                    if len(mesh_top.vertices) > 0:
                        vertices_list.append(mesh_top.vertices)
                        faces_list.append(mesh_top.faces + vertex_offset)

                if vertices_list:
                    combined_vertices = np.vstack(vertices_list)
                    combined_faces = np.vstack(faces_list)
                    combined_mesh = trimesh.Trimesh(vertices=combined_vertices, faces=combined_faces, process=False)
                    combined_mesh.export(str(trans_stl_path))
                    generated_files.append((trans_stl_path, f"{transparent_filament['name']} (for {filament['name']} {k})", 0, layers_per_sandwich))

        logger.info(f"Exploded mode: generated {len(generated_files)} STL files")
        return generated_files

    def _pack_sandwich_groups(self, layer_counts):
        """Pack (color, level) pairs into physical sandwiches with up to 3 colors each.

        Uses greedy bin-packing: sorts pairs by mask size (largest first),
        assigns each to the first sandwich where its mask doesn't overlap
        with existing masks and the sandwich has < 3 colors.

        Args:
            layer_counts: ndarray (H, W, num_colors) int32

        Returns:
            List of sandwich groups. Each group is a list of (color_idx, level, mask) tuples.
        """
        num_colors = layer_counts.shape[2]
        valid_mask = self.alpha_mask >= 0.5

        # Extract all (color, level) pairs with their masks
        pairs = []
        for c in range(num_colors):
            max_k = int(layer_counts[:, :, c].max())
            for k in range(1, max_k + 1):
                mask = (layer_counts[:, :, c] >= k) & valid_mask
                pixel_count = int(mask.sum())
                if pixel_count > 0:
                    pairs.append((c, k, mask, pixel_count))

        # Sort by pixel count descending (largest masks first for better packing)
        pairs.sort(key=lambda x: -x[3])

        # Greedy bin-packing
        sandwiches = []  # Each sandwich: list of (color_idx, level, mask)
        sandwich_combined_masks = []  # Combined mask per sandwich

        for c, k, mask, pixel_count in pairs:
            placed = False
            for s_idx, sandwich in enumerate(sandwiches):
                if len(sandwich) >= 3:
                    continue
                # Check overlap: no pixel should be in both masks
                combined = sandwich_combined_masks[s_idx]
                if not np.any(mask & combined):
                    sandwich.append((c, k, mask))
                    sandwich_combined_masks[s_idx] = combined | mask
                    placed = True
                    break

            if not placed:
                sandwiches.append([(c, k, mask)])
                sandwich_combined_masks.append(mask.copy())

        return sandwiches

    def _generate_exploded_multi(self, output_base_path):
        """Generate exploded-multi mode: multi-color sandwiches with up to 3 colors each.

        Same pixel-level Beer-Lambert optimization as exploded mode but with higher
        per-color caps (5 vs 3). After optimization, color-level pairs are packed into
        physical sandwiches via greedy bin-packing (up to 3 non-overlapping color masks
        per sandwich).

        Each physical sandwich is 3 layers tall (transparent/multi-color middle/transparent).
        Outputs up to 4 STLs per sandwich: 1 transparent + up to 3 color STLs.
        """
        generated_files = []

        num_color_filaments = self.num_colors - 1
        if num_color_filaments <= 0:
            logger.warning("Exploded-multi mode requires at least 1 color + transparent")
            return generated_files

        color_filaments = self.selected_filaments.iloc[:num_color_filaments]
        transparent_filament = self.selected_filaments.iloc[-1]

        total_sandwiches = int(self.model_height / self.layer_height)
        logger.info(f"Exploded-multi mode: {total_sandwiches} sandwich budget, {num_color_filaments} colors")

        # Higher cap per color since multi-color packing is more efficient.
        # Iteratively reduce caps if packing exceeds the sandwich budget.
        MAX_SANDWICHES_PER_COLOR = self.max_color_sandwiches if self.max_color_sandwiches is not None else 5

        for attempt in range(MAX_SANDWICHES_PER_COLOR):
            current_cap = MAX_SANDWICHES_PER_COLOR - attempt
            per_color_caps = [min(current_cap, total_sandwiches) for _ in range(num_color_filaments)]

            # Reduce caps if combinatorial space is too large
            combo_size = 1
            for cap in per_color_caps:
                combo_size *= (cap + 1)
            while combo_size > 500_000:
                max_idx = per_color_caps.index(max(per_color_caps))
                per_color_caps[max_idx] -= 1
                combo_size = 1
                for cap in per_color_caps:
                    combo_size *= (cap + 1)

            # Compute optimal layer counts per pixel per color
            layer_counts = self._compute_exploded_layer_counts(
                color_filaments, transparent_filament, total_sandwiches, per_color_caps
            )

            # Pack into physical sandwiches (up to 3 colors each)
            sandwich_groups = self._pack_sandwich_groups(layer_counts)
            num_physical = len(sandwich_groups)

            if num_physical <= total_sandwiches:
                logger.info(f"Exploded-multi: packed into {num_physical} physical sandwiches "
                             f"(budget: {total_sandwiches}, cap: {current_cap})")
                break
            else:
                logger.info(f"Exploded-multi: cap={current_cap} produced {num_physical} sandwiches "
                             f"(budget: {total_sandwiches}), reducing cap...")
        else:
            # Even cap=1 exceeded budget — use what we have
            logger.info(f"Exploded-multi: packed into {num_physical} physical sandwiches "
                         f"(budget: {total_sandwiches}, cap: 1)")

        all_pixels_mask = self.alpha_mask >= 0.5

        # Sandwich layer structure (same as _generate_exploded)
        bl = self.base_layers
        sl = self.sandwich_layers
        layers_per_sandwich = bl + sl + (1 if self.use_fill else 0)

        # Pre-generate shared full-plate meshes (same for every sandwich)
        mesh_bottom = self._generate_flat_layer_stl(all_pixels_mask, 0, bl)
        mesh_top = self._generate_flat_layer_stl(all_pixels_mask, bl + sl, bl + sl + 1) if self.use_fill else None

        for s_idx, group in enumerate(sandwich_groups):
            sandwich_num = s_idx + 1
            color_names_in_sandwich = []
            combined_color_mask = np.zeros_like(all_pixels_mask)

            # Generate color STLs for this sandwich
            for color_idx, level, mask in group:
                filament = color_filaments.iloc[color_idx]
                color_name = filament['name'].replace(' ', '_').replace('/', '_')
                color_names_in_sandwich.append(filament['name'])
                combined_color_mask |= mask

                pixel_count = int(mask.sum())
                logger.info(f"  Sandwich {sandwich_num}/{num_physical}: "
                             f"{filament['name']} level {level} ({pixel_count} pixels)")

                color_stl_path = (output_base_path.parent /
                    f"{output_base_path.stem}_S{sandwich_num:02d}_{color_name}_color.stl")
                color_mesh = self._generate_flat_layer_stl(mask, bl, bl + sl)
                if len(color_mesh.vertices) > 0:
                    color_mesh.export(str(color_stl_path))
                    generated_files.append((
                        color_stl_path,
                        f"{filament['name']} (sandwich {sandwich_num})",
                        bl, bl + sl
                    ))

            # Generate transparent STL: bottom + optional inverse middle + optional top
            inverse_mask = ~combined_color_mask & all_pixels_mask

            trans_stl_path = (output_base_path.parent /
                f"{output_base_path.stem}_S{sandwich_num:02d}_transparent.stl")

            vertices_list = []
            faces_list = []
            vertex_offset = 0

            if len(mesh_bottom.vertices) > 0:
                vertices_list.append(mesh_bottom.vertices)
                faces_list.append(mesh_bottom.faces + vertex_offset)
                vertex_offset += len(mesh_bottom.vertices)

            if self.use_fill:
                if inverse_mask.any():
                    mesh_middle = self._generate_flat_layer_stl(inverse_mask, bl, bl + sl)
                    if len(mesh_middle.vertices) > 0:
                        vertices_list.append(mesh_middle.vertices)
                        faces_list.append(mesh_middle.faces + vertex_offset)
                        vertex_offset += len(mesh_middle.vertices)

                if len(mesh_top.vertices) > 0:
                    vertices_list.append(mesh_top.vertices)
                    faces_list.append(mesh_top.faces + vertex_offset)

            if vertices_list:
                combined_vertices = np.vstack(vertices_list)
                combined_faces = np.vstack(faces_list)
                combined_mesh = trimesh.Trimesh(
                    vertices=combined_vertices, faces=combined_faces, process=False
                )
                combined_mesh.export(str(trans_stl_path))
                names_str = ', '.join(color_names_in_sandwich)
                generated_files.append((
                    trans_stl_path,
                    f"Transparent (sandwich {sandwich_num}: {names_str})",
                    0, layers_per_sandwich
                ))

        logger.info(f"Exploded-multi: generated {len(generated_files)} STL files "
                     f"across {num_physical} sandwiches")
        return generated_files

    def generate_preview_scene(self):
        """Generate a trimesh.Scene with colored meshes for interactive 3D preview

        Mirrors the heightmap pipeline from generate_all() / _generate_face_down()
        and returns a Scene instead of exporting STLs. Downsamples to ~100K pixels
        for browser-renderable mesh size while preserving good detail.

        Returns:
            trimesh.Scene with one colored mesh per filament
        """
        if self.use_exploded or self.use_exploded_multi or self.use_exploded_cmyk:
            return self._generate_exploded_preview_scene()

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
        cap_sort = self.use_cap_layers and not self.use_face_down
        sorted_filaments = self._sort_filaments_by_luminosity(cap_layers_sort=cap_sort)

        # Swap only pixel_size and alpha_mask (used by called methods)
        orig_pixel_size = self.pixel_size
        orig_alpha = self.alpha_mask
        try:
            self.pixel_size = ds_pixel_size
            self.alpha_mask = ds_alpha

            # Contrast enhancement (uses self.alpha_mask internally)
            enhanced = self._apply_contrast_enhancement(ds_grayscale.copy())

            alpha_pixels = ds_alpha >= 0.5

            if self.use_face_down and self.use_cap_layers:
                # Face-down + cap layers: image colors exclude transparent filament
                image_filaments = sorted_filaments.iloc[:-1].reset_index(drop=True)
                transparent_filament = sorted_filaments.iloc[-1]

                cap_layers_count = self.base_layers
                cap_height = cap_layers_count * self.layer_height
                model_height_minus_cap = self.model_height - cap_height

                image_tds = np.array([f['transmission_distance'] for _, f in image_filaments.iterrows()])
                _, _, z_boundaries = self._allocate_layers_td_proportional(
                    image_tds, self.num_layers - cap_layers_count)
                pixel_height = self._compute_heightmap(enhanced, alpha_pixels, model_height_minus_cap)
            else:
                filament_tds = np.array([f['transmission_distance'] for _, f in sorted_filaments.iterrows()])
                _, _, z_boundaries = self._allocate_layers_td_proportional(
                    filament_tds, self.num_layers)
                pixel_height = self._compute_heightmap(
                    enhanced, alpha_pixels, self.model_height, invert=self.use_face_down)

            scene = trimesh.Scene()
            min_thickness = self.layer_height * 0.5

            combined_mask = alpha_pixels & (pixel_height > min_thickness)
            z_bottom_all = np.zeros_like(pixel_height)

            if self.use_cap_layers:
                # Flat top for cap-layer modes (both standalone and face-down+cap)
                z_top = np.full_like(pixel_height, self.model_height)
            else:
                # Topographical heightmap for standard and face-down modes
                z_top = pixel_height

            z_top = np.where(combined_mask, z_top, 0)
            mesh = self._generate_topographical_stl(z_bottom_all, z_top, combined_mask)

            if len(mesh.vertices) > 0:
                if self.use_face_down and self.use_cap_layers:
                    preview_rgb = self._render_face_down_cap_preview(
                        pixel_height, z_boundaries, image_filaments,
                        transparent_filament, cap_height)
                elif self.use_cap_layers:
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

    def _generate_exploded_preview_scene(self):
        """Generate 3D preview for exploded/exploded-multi modes.

        Shows each sandwich as a flat slab at its stacking position with a visible
        gap between sandwiches. Color pixels are shown in their filament RGB,
        transparent pixels in near-white. Runs the full Beer-Lambert optimization
        at downsampled resolution.

        Returns:
            trimesh.Scene with one colored mesh per sandwich
        """
        from math import sqrt

        H, W = self.image_grayscale.shape

        # Downsample for exploded preview (~250K pixels — greedy meshing keeps face count low)
        ds = max(1, round(sqrt(H * W / 250000)))
        ds_alpha = self.alpha_mask[::ds, ::ds]
        ds_H, ds_W = ds_alpha.shape
        ds_pixel_size = self.width_mm / ds_W
        ds_image_rgb = self.image_rgb[::ds, ::ds] if self.image_rgb is not None else None

        logger.info(f"3D exploded preview: {H}x{W} -> {ds_H}x{ds_W} (ds={ds})")

        num_color_filaments = self.num_colors - 1
        if num_color_filaments <= 0:
            return trimesh.Scene()

        color_filaments = self.selected_filaments.iloc[:num_color_filaments]
        transparent_filament = self.selected_filaments.iloc[-1]
        total_sandwiches = int(self.model_height / self.layer_height)

        # Compute caps (same logic as generation methods)
        if self.max_color_sandwiches is not None:
            max_cap = self.max_color_sandwiches
        elif self.use_exploded_cmyk:
            max_cap = 1
        elif self.use_exploded_multi:
            max_cap = 5
        else:
            max_cap = 3
        per_color_caps = [min(max_cap, total_sandwiches) for _ in range(num_color_filaments)]

        # Reduce combo space
        combo_size = 1
        for cap in per_color_caps:
            combo_size *= (cap + 1)
        while combo_size > 500_000:
            max_idx = per_color_caps.index(max(per_color_caps))
            per_color_caps[max_idx] -= 1
            combo_size = 1
            for cap in per_color_caps:
                combo_size *= (cap + 1)

        # Swap all resolution-dependent attributes for downsampled resolution
        orig_pixel_size = self.pixel_size
        orig_alpha = self.alpha_mask
        orig_rgb = self.image_rgb
        orig_grayscale = self.image_grayscale
        ds_grayscale = self.image_grayscale[::ds, ::ds]
        try:
            self.pixel_size = ds_pixel_size
            self.alpha_mask = ds_alpha
            self.image_rgb = ds_image_rgb
            self.image_grayscale = ds_grayscale

            # Run Beer-Lambert optimization at downsampled resolution
            # For exploded-multi, retry with lower caps if packing exceeds budget
            if self.use_exploded_multi:
                for attempt in range(max_cap):
                    current_cap = max_cap - attempt
                    per_color_caps = [min(current_cap, total_sandwiches) for _ in range(num_color_filaments)]
                    cs = 1
                    for cap in per_color_caps:
                        cs *= (cap + 1)
                    while cs > 500_000:
                        mi = per_color_caps.index(max(per_color_caps))
                        per_color_caps[mi] -= 1
                        cs = 1
                        for cap in per_color_caps:
                            cs *= (cap + 1)

                    layer_counts = self._compute_exploded_layer_counts(
                        color_filaments, transparent_filament, total_sandwiches, per_color_caps
                    )
                    sandwich_groups = self._pack_sandwich_groups(layer_counts)
                    if len(sandwich_groups) <= total_sandwiches:
                        break
            else:
                layer_counts = self._compute_exploded_layer_counts(
                    color_filaments, transparent_filament, total_sandwiches, per_color_caps
                )

            # For standard exploded, build sandwich groups directly
            if not self.use_exploded_multi:
                # For standard exploded, one sandwich per (color, level)
                sandwich_groups = []
                valid_mask = ds_alpha >= 0.5
                for c in range(num_color_filaments):
                    max_k = int(layer_counts[:, :, c].max())
                    for k in range(1, max_k + 1):
                        mask = (layer_counts[:, :, c] >= k) & valid_mask
                        if mask.any():
                            sandwich_groups.append([(c, k, mask)])

            num_physical = len(sandwich_groups)
            logger.info(f"3D exploded preview: {num_physical} sandwiches to render")

            # Visual parameters — meshes are generated collapsed (gap=0).
            # The HTML viewer slider controls the gap dynamically.
            slab_thickness = 0.5  # mm per sandwich slab (exaggerated for visibility)
            step = slab_thickness  # no gap — slider adds it

            # Transparent fill: very faint so it doesn't obscure color layers
            trans_rgb_uint8 = np.array([240, 240, 235], dtype=np.uint8)
            trans_alpha = 15  # ~6% opacity — just enough to see sandwich outline

            # Build filament RGB + Beer-Lambert alpha lookup
            # Alpha = opacity for sandwich_layers of material, with min floor for visibility
            MIN_ALPHA = 40  # ~15% — ensures even high-TD filaments are visible
            color_thickness = self.layer_height * self.sandwich_layers
            filament_rgb_uint8 = {}
            filament_alpha = {}
            for c in range(num_color_filaments):
                f = color_filaments.iloc[c]
                filament_rgb_uint8[c] = (np.array(f['rgb']) * 255).astype(np.uint8)
                td = max(f['transmission_distance'], 0.1)
                opacity = 1.0 - np.exp(-color_thickness / td)
                filament_alpha[c] = max(int(opacity * 255), MIN_ALPHA)

            all_pixels_mask = ds_alpha >= 0.5
            scene = trimesh.Scene()

            for s_idx, group in enumerate(sandwich_groups):
                z_base = s_idx * step

                # Generate separate meshes for each color region + transparent fill
                combined_color_mask = np.zeros((ds_H, ds_W), dtype=bool)
                color_names = []

                for color_idx, level, mask in group:
                    if not mask.any():
                        continue
                    combined_color_mask |= mask

                    name = color_filaments.iloc[color_idx]['name']
                    if name not in color_names:
                        color_names.append(name)

                    # Color mesh for this region
                    rects = self._greedy_mesh_rects(mask)
                    if rects:
                        verts, faces = self._build_box_mesh(
                            rects, z_base, z_base + slab_thickness, ds_pixel_size)
                        if len(verts) > 0:
                            mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
                            fc = np.zeros((len(mesh.faces), 4), dtype=np.uint8)
                            fc[:, :3] = filament_rgb_uint8[color_idx]
                            fc[:, 3] = filament_alpha[color_idx]
                            mesh.visual.face_colors = fc
                            scene.add_geometry(
                                mesh,
                                geom_name=f"S{s_idx+1:02d}_{name.replace(' ','_')}_L{level}"
                            )

                # Transparent fill (all valid pixels minus color pixels)
                trans_mask = all_pixels_mask & ~combined_color_mask
                if trans_mask.any():
                    rects = self._greedy_mesh_rects(trans_mask)
                    if rects:
                        verts, faces = self._build_box_mesh(
                            rects, z_base, z_base + slab_thickness, ds_pixel_size)
                        if len(verts) > 0:
                            mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
                            fc = np.zeros((len(mesh.faces), 4), dtype=np.uint8)
                            fc[:, :3] = trans_rgb_uint8
                            fc[:, 3] = trans_alpha
                            mesh.visual.face_colors = fc
                            geom_name = f"S{s_idx+1:02d}_transparent"
                            scene.add_geometry(mesh, geom_name=geom_name)

            logger.info(f"3D exploded preview: {len(scene.geometry)} meshes in scene")
            return scene

        finally:
            self.pixel_size = orig_pixel_size
            self.alpha_mask = orig_alpha
            self.image_rgb = orig_rgb
            self.image_grayscale = orig_grayscale

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
        bl = self.base_layers
        middle_layers = self.num_layers - 2 * bl
        layers_per_color = middle_layers // num_middle_colors if num_middle_colors > 0 else 0

        # 1. FLAT BASE (darkest filament)
        darkest_filament = sorted_filaments.iloc[0]
        color_name = darkest_filament['name'].replace(' ', '_').replace('/', '_')
        output_path = output_base_path.parent / f"{output_base_path.stem}_{color_name}_base.stl"

        logger.info(f"Generating flat base: {color_name} (layers 0-{bl})")

        # Full flat layer covering all valid pixels
        all_pixels_mask = self.alpha_mask >= 0.5
        mesh = self._generate_flat_layer_stl(all_pixels_mask, 0, bl)
        mesh.export(str(output_path))
        generated_files.append((output_path, f"{darkest_filament['name']} (base)", 0, bl))

        # Track which pixels are covered by colored layers (for negative calculation)
        combined_color_mask = np.zeros_like(self.image_grayscale, dtype=bool)

        # 2. MIDDLE COLOR LAYERS (shaped)
        for i in range(num_middle_colors):
            filament = sorted_filaments.iloc[i + 1]  # Skip darkest (used for base)
            color_name = filament['name'].replace(' ', '_').replace('/', '_')
            output_path = output_base_path.parent / f"{output_base_path.stem}_{color_name}.stl"

            layer_start = bl + (i * layers_per_color)
            layer_end = bl + ((i + 1) * layers_per_color)

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

        middle_layer_start = bl
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
            mesh_negative = self._generate_flat_layer_stl(negative_mask, middle_layer_start, self.num_layers - bl)
            vertices_list.append(mesh_negative.vertices)
            faces_list.append(mesh_negative.faces + vertex_offset)
            vertex_offset += len(mesh_negative.vertices)
            logger.info(f"  Negative fill: {negative_mask.sum()} pixels (fills gaps in layers {middle_layer_start}-{self.num_layers - bl})")

        # Part 2: Flat top (top layers)
        mesh_top = self._generate_flat_layer_stl(all_pixels_mask, self.num_layers - bl, middle_layer_end)
        vertices_list.append(mesh_top.vertices)
        faces_list.append(mesh_top.faces + vertex_offset)
        logger.info(f"  Flat top: all pixels (layers {self.num_layers - bl}-{middle_layer_end})")

        # Combine into single mesh
        if len(vertices_list) > 0:
            combined_vertices = np.vstack(vertices_list)
            combined_faces = np.vstack(faces_list)
            combined_mesh = trimesh.Trimesh(vertices=combined_vertices, faces=combined_faces, process=False)
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

    def _generate_quantized_stl(self, z_bottom, z_top, pixel_mask):
        """Generate STL with height quantization + incremental slab greedy meshing.

        Rounds heights to nearest layer_height multiple, then decomposes the height
        range into incremental horizontal slabs. Each slab covers all pixels whose
        column spans that Z range. Within each slab every pixel has the same height,
        so greedy meshing produces non-overlapping geometry that slicers handle well.

        Same interface as _generate_topographical_stl.

        Args:
            z_bottom: 2D array of bottom z-heights (mm) for each pixel
            z_top: 2D array of top z-heights (mm) for each pixel
            pixel_mask: 2D boolean array of which pixels to include

        Returns:
            trimesh.Trimesh object
        """
        if not pixel_mask.any():
            return trimesh.Trimesh()

        lh = self.layer_height

        # Quantize heights to nearest layer_height multiple
        q_z_top = np.round(z_top / lh) * lh
        q_z_bottom = np.round(z_bottom / lh) * lh

        # Filter: only pixels where quantized top > quantized bottom
        effective_mask = pixel_mask & (q_z_top > q_z_bottom + 1e-6)
        if not effective_mask.any():
            return trimesh.Trimesh()

        # Collect all unique Z levels from both bottom and top arrays
        active_zb = q_z_bottom[effective_mask]
        active_zt = q_z_top[effective_mask]
        all_z_values = np.unique(np.concatenate([active_zb, active_zt]))
        # Convert to integer layer indices to avoid float comparison issues
        z_layers = np.round(all_z_values / lh).astype(np.int64)
        z_layers = np.unique(z_layers)

        # Also convert per-pixel values to integer layers for fast comparison
        q_zb_layers = np.round(q_z_bottom / lh).astype(np.int64)
        q_zt_layers = np.round(q_z_top / lh).astype(np.int64)

        all_verts = []
        all_faces = []
        vertex_offset = 0
        total_rects = 0

        # Process each consecutive pair of Z levels as a slab
        for s in range(len(z_layers) - 1):
            slab_bottom_layer = z_layers[s]
            slab_top_layer = z_layers[s + 1]

            # Slab mask: pixels whose column fully spans this slab
            # i.e. quantized bottom <= slab bottom AND quantized top >= slab top
            slab_mask = effective_mask & (q_zb_layers <= slab_bottom_layer) & (q_zt_layers >= slab_top_layer)

            if not slab_mask.any():
                continue

            rects = self._greedy_mesh_rects(slab_mask)
            if rects:
                zb_val = slab_bottom_layer * lh
                zt_val = slab_top_layer * lh
                verts, faces = self._build_box_mesh(rects, zb_val, zt_val, self.pixel_size)
                if len(verts) > 0:
                    all_verts.append(verts)
                    all_faces.append(faces + vertex_offset)
                    vertex_offset += len(verts)
                    total_rects += len(rects)

        if not all_verts:
            return trimesh.Trimesh()

        combined_verts = np.vstack(all_verts)
        combined_faces = np.vstack(all_faces)
        mesh = trimesh.Trimesh(vertices=combined_verts, faces=combined_faces, process=False)

        total_pixels = effective_mask.sum()
        num_slabs = len(z_layers) - 1
        logger.info(f"  Quantized mesh: {total_pixels:,} pixels, {num_slabs} slabs, "
                     f"{total_rects} rects -> {len(combined_faces):,} faces "
                     f"(vs ~{total_pixels * 6:,} topographical)")

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

        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

        original_pixels = pixel_mask.sum()
        logger.info(f"    Flat layer: {original_pixels:,} pixels -> {len(rects)} rects, {len(faces)} faces")

        return mesh

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

            transmission_3d = transmission[:, :, np.newaxis]
            light = light * transmission_3d + rgb * (1.0 - transmission_3d)

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
        bl = self.base_layers
        middle_layers = self.num_layers - 2 * bl
        layers_per_color = middle_layers // num_middle_colors if num_middle_colors > 0 else 0

        base_thickness = bl * self.layer_height
        middle_thickness = layers_per_color * self.layer_height
        top_thickness = bl * self.layer_height

        # Start with white backlight
        light = np.ones((H, W, 3))

        # Use subtractive (multiplicative) filter model for cap layers.
        # The opaque base reduces light to ~20%. With additive model,
        # colored middles would ADD brightness (filament_rgb > remaining light),
        # causing inversion. Subtractive model ensures light only decreases.

        # 1. Base layer (darkest filament) — always present, uniform thickness
        td = max(filament_tds[0], 0.1)
        transmission = np.exp(-base_thickness / td)
        light = light * (transmission + filament_rgbs[0] * (1.0 - transmission))

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

            present_3d = present[:, :, np.newaxis]
            colored = light * (color_trans + rgb * (1.0 - color_trans))
            cleared = light * (clear_trans + clear_rgb * (1.0 - clear_trans))
            light = np.where(present_3d, colored, cleared)

        # 3. Clear top layer — always present, uniform thickness
        transmission = np.exp(-top_thickness / clear_td)
        light = light * (transmission + clear_rgb * (1.0 - transmission))

        return np.clip(light, 0, 1)

    def _render_face_down_cap_preview(self, pixel_height, z_boundaries,
                                       image_filaments, transparent_filament, cap_height):
        """Render Beer-Lambert preview for face-down + cap layers mode

        Uses standard heightmap (bright=tall, dark=short). After physical flip,
        light enters from back (darkest color at highest z-band), passes through
        color bands from dark-to-light, through negative fill, and exits through
        the transparent cap (viewing surface).

        Args:
            pixel_height: 2D array of per-pixel heights (mm), standard heightmap
            z_boundaries: 1D array of image color band boundaries (mm), dark-to-light
            image_filaments: DataFrame of image filaments (dark-to-light order)
            transparent_filament: Series for the transparent filament
            cap_height: Height of transparent cap (mm)

        Returns:
            HxWx3 RGB preview image (0-1 range)
        """
        H, W = pixel_height.shape
        num_image_colors = len(image_filaments)
        model_height_minus_cap = self.model_height - cap_height

        image_rgbs = np.array([f['rgb'] for _, f in image_filaments.iterrows()])
        image_tds = np.array([f['transmission_distance'] for _, f in image_filaments.iterrows()])
        trans_rgb = transparent_filament['rgb']
        trans_td = max(transparent_filament['transmission_distance'], 0.1)

        # White backlight from behind (after flip: enters from darkest side)
        light = np.ones((H, W, 3))

        # After physical flip, light path is:
        # 1. Darkest color bands (highest z in generation space, back after flip)
        # 2. Through lighter color bands toward viewing surface
        # 3. Through transparent negative fill
        # 4. Through transparent cap (viewing surface)

        # Process from darkest (highest z) to lightest (lowest z)
        for i in range(num_image_colors - 1, -1, -1):
            z_lo = z_boundaries[i]
            z_hi = z_boundaries[i + 1]

            # Per-pixel thickness in this color band
            thickness = np.clip(pixel_height, z_lo, z_hi) - z_lo
            td = max(image_tds[i], 0.1)
            rgb = image_rgbs[i]

            transmission = np.exp(-thickness / td)
            transmission_3d = transmission[:, :, np.newaxis]
            light = light * transmission_3d + rgb * (1.0 - transmission_3d)

        # Transparent negative fill (from pixel_height to model_height - cap_height)
        neg_thickness = model_height_minus_cap - pixel_height
        neg_thickness = np.clip(neg_thickness, 0, model_height_minus_cap)
        transmission = np.exp(-neg_thickness / trans_td)
        transmission_3d = transmission[:, :, np.newaxis]
        light = light * transmission_3d + trans_rgb * (1.0 - transmission_3d)

        # Transparent cap (uniform thickness, viewing surface)
        transmission = np.exp(-cap_height / trans_td)
        light = light * transmission + trans_rgb * (1.0 - transmission)

        return np.clip(light, 0, 1)

    def _generate_face_down(self, output_base_path):
        """Generate face-down mode with inverted heightmap (dark=tall, light=short).

        Same pipeline as standard but with inverted heightmap for face-down printing.
        """
        logger.info("=" * 60)
        logger.info("FACE-DOWN MODE: Topographical generation (inverted heightmap)")
        logger.info("=" * 60)

        sorted_filaments = self._sort_filaments_by_luminosity()
        enhanced_grayscale = self._apply_contrast_enhancement(self.image_grayscale.copy())
        alpha_pixels = self.alpha_mask >= 0.5

        filament_tds = np.array([f['transmission_distance'] for _, f in sorted_filaments.iterrows()])
        layer_counts, layer_boundaries, z_boundaries = self._allocate_layers_td_proportional(
            filament_tds, self.num_layers)

        for idx in range(self.num_colors):
            name = sorted_filaments.iloc[idx]['name']
            lc = layer_counts[idx]
            logger.info(f"  {name}: TD={filament_tds[idx]:.1f}mm -> {lc} layers ({lc * self.layer_height:.2f}mm)")

        pixel_height = self._compute_heightmap(enhanced_grayscale, alpha_pixels, self.model_height, invert=True)

        # Store for preview
        self.face_down_pixel_height = pixel_height
        self.face_down_z_boundaries = z_boundaries
        self.face_down_filaments = sorted_filaments

        generated_files = self._generate_color_band_stls(
            sorted_filaments, pixel_height, z_boundaries, layer_boundaries,
            alpha_pixels, output_base_path)

        logger.info(f"Face-down generation complete: {len(generated_files)} STL files")
        return generated_files

    def _generate_face_down_with_cap_layers(self, output_base_path):
        """Generate face-down + cap layers: standard heightmap + negative fill + transparent cap, then Z-flip.

        After Z-flip: cap at z=0 (bed), darkest at z=model_height (back).
        """
        logger.info("=" * 60)
        logger.info("FACE-DOWN + CAP LAYERS: Standard generation + Z-flip")
        logger.info("=" * 60)

        sorted_filaments = self._sort_filaments_by_luminosity()

        # Last filament (lightest/most transparent) = cap + negative fill
        transparent_filament = sorted_filaments.iloc[-1]
        image_filaments = sorted_filaments.iloc[:-1].reset_index(drop=True)
        num_image_colors = len(image_filaments)

        logger.info(f"Transparent (cap + negative): {transparent_filament['name']} "
                     f"(TD={transparent_filament['transmission_distance']:.1f})")
        logger.info(f"Image colors: {num_image_colors}")

        # Layer allocation: cap layers + image colors
        cap_layers_count = self.base_layers
        cap_height = cap_layers_count * self.layer_height
        image_total_layers = self.num_layers - cap_layers_count
        model_height_minus_cap = self.model_height - cap_height

        image_tds = np.array([f['transmission_distance'] for _, f in image_filaments.iterrows()])
        layer_counts, image_layer_boundaries, z_boundaries = self._allocate_layers_td_proportional(
            image_tds, image_total_layers)

        for idx in range(num_image_colors):
            name = image_filaments.iloc[idx]['name']
            lc = layer_counts[idx]
            logger.info(f"  {name}: TD={image_tds[idx]:.1f}mm -> {lc} layers ({lc * self.layer_height:.2f}mm)")

        # Standard heightmap (bright=tall), capped at model_height - cap_height
        enhanced_grayscale = self._apply_contrast_enhancement(self.image_grayscale.copy())
        alpha_pixels = self.alpha_mask >= 0.5
        pixel_height = self._compute_heightmap(enhanced_grayscale, alpha_pixels, model_height_minus_cap)

        # Store for preview
        self.face_down_pixel_height = pixel_height
        self.face_down_z_boundaries = z_boundaries
        self.face_down_image_filaments = image_filaments
        self.face_down_transparent_filament = transparent_filament
        self.face_down_cap_height = cap_height

        # 1. IMAGE COLOR STLs (collect meshes for Z-flip)
        all_mesh_outputs = self._generate_color_band_stls(
            image_filaments, pixel_height, z_boundaries, image_layer_boundaries,
            alpha_pixels, output_base_path, collect_meshes=True)

        # 2. TRANSPARENT NEGATIVE FILL + FLAT CAP (combined into single STL)
        min_thickness = self.layer_height * 0.5
        z_bottom_negative = pixel_height.copy()
        z_top_negative = np.full_like(pixel_height, model_height_minus_cap)
        negative_mask = alpha_pixels & (pixel_height < model_height_minus_cap - min_thickness)

        logger.info(f"Generating transparent negative fill: {int(np.sum(negative_mask))} pixels")

        cap_mesh = self._generate_flat_layer_stl(
            alpha_pixels,
            int(round(model_height_minus_cap / self.layer_height)),
            int(round(self.model_height / self.layer_height)))

        color_name = transparent_filament['name'].replace(' ', '_').replace('/', '_')
        cap_output_path = output_base_path.parent / f"{output_base_path.stem}_{color_name}.stl"

        vertices_list = [cap_mesh.vertices]
        faces_list = [cap_mesh.faces]
        vertex_offset = len(cap_mesh.vertices)

        if negative_mask.any():
            negative_mesh = self._generate_quantized_stl(
                z_bottom_negative, z_top_negative, negative_mask)
            if len(negative_mesh.vertices) > 0:
                vertices_list.append(negative_mesh.vertices)
                faces_list.append(negative_mesh.faces + vertex_offset)
                logger.info(f"  Negative fill: {len(negative_mesh.vertices)} vertices, "
                             f"{len(negative_mesh.faces)} faces")

        combined_mesh = trimesh.Trimesh(
            vertices=np.vstack(vertices_list),
            faces=np.vstack(faces_list), process=False)
        all_mesh_outputs.append((combined_mesh, cap_output_path, transparent_filament['name']))

        # 3. Z-FLIP ALL MESHES
        logger.info(f"Z-flipping all meshes (model_height={self.model_height:.2f}mm)")
        generated_files = []

        for mesh, output_path, filament_name in all_mesh_outputs:
            mesh.vertices[:, 2] = self.model_height - mesh.vertices[:, 2]
            mesh.fix_normals()
            mesh.export(str(output_path))

            min_z = float(np.min(mesh.vertices[:, 2]))
            max_z = float(np.max(mesh.vertices[:, 2]))
            layer_start_actual = int(np.floor(min_z / self.layer_height))
            layer_end_actual = int(np.ceil(max_z / self.layer_height))
            generated_files.append((output_path, filament_name, layer_start_actual, layer_end_actual))
            logger.info(f"  {filament_name}: z={min_z:.2f}-{max_z:.2f}mm (flipped)")

        logger.info(f"Face-down cap layers complete: {len(generated_files)} STL files")
        return generated_files
