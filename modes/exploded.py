"""Exploded mode for HueCLI STL generation.

Per-pixel Beer-Lambert optimized sandwiches. Each color is capped at
max_sandwiches_per_color sandwiches, allowing intensity control
(0..N layers per pixel per color).

Each sandwich has a transparent base, color middle, and optional transparent top.
"""

import numpy as np
import trimesh
import logging

from color_science import compute_exploded_layer_counts
from mesh import generate_flat_layer_stl

logger = logging.getLogger(__name__)


def generate(output_base_path, config, processed_image, selected_filaments,
             max_sandwiches_per_color=None):
    """Generate exploded mode: per-pixel Beer-Lambert optimized sandwiches.

    Args:
        output_base_path: Path for output files
        config: PipelineConfig with layer_height, model_height, sandwich_layers, use_fill,
                base_layers, max_color_sandwiches, dither_mode, width_mm
        processed_image: ProcessedImage with grayscale, image_rgb, alpha_mask, width_px
        selected_filaments: DataFrame of selected filaments
        max_sandwiches_per_color: Override for max sandwiches per color (default: use config or 3)

    Returns:
        List of (path, name, layer_start, layer_end) tuples
    """
    num_colors = len(selected_filaments)
    pixel_size = config.width_mm / processed_image.width_px
    generated_files = []

    # Determine max_sandwiches_per_color
    if max_sandwiches_per_color is None:
        if config.max_color_sandwiches is not None:
            max_sandwiches_per_color = config.max_color_sandwiches
        else:
            max_sandwiches_per_color = 3

    # Separate color vs transparent filaments (transparent is last)
    num_color_filaments = num_colors - 1
    if num_color_filaments <= 0:
        logger.warning("Exploded mode requires at least 1 color + transparent")
        return generated_files

    color_filaments = selected_filaments.iloc[:num_color_filaments]
    transparent_filament = selected_filaments.iloc[-1]

    # Fixed total sandwich count
    total_sandwiches = int(config.model_height / config.layer_height)
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

    H, W = processed_image.grayscale.shape

    # Compute optimal layer counts per pixel per color
    layer_counts = compute_exploded_layer_counts(
        color_filaments, transparent_filament, total_sandwiches, per_color_caps,
        (H, W), processed_image.image_rgb, processed_image.alpha_mask,
        config.layer_height, config.sandwich_layers, config.use_fill,
        dither_mode=config.dither_mode
    )

    all_pixels_mask = processed_image.alpha_mask >= 0.5

    # Sandwich layer structure:
    # Bottom transparent: layer 0 to base_layers
    # Color middle: layer base_layers to base_layers+sandwich_layers
    # Top transparent (fill only): layer base_layers+sandwich_layers to base_layers+sandwich_layers+1
    bl = config.base_layers
    sl = config.sandwich_layers
    layers_per_sandwich = bl + sl + (1 if config.use_fill else 0)

    # Pre-generate shared full-plate meshes (same for every sandwich)
    mesh_bottom = generate_flat_layer_stl(all_pixels_mask, 0, bl, pixel_size, config.layer_height)
    mesh_top = generate_flat_layer_stl(all_pixels_mask, bl + sl, bl + sl + 1, pixel_size, config.layer_height) if config.use_fill else None

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

            logger.debug(f"  {color_name} sandwich {k}/{max_k}: {pixel_count} pixels "
                          f"({layers_per_sandwich} layers/sandwich, {sl} color, fill={config.use_fill})")

            # Color STL: middle layers
            suffix = f"_{k}" if max_k > 1 else ""
            color_stl_path = output_base_path.parent / f"{output_base_path.stem}_{color_name}{suffix}_color.stl"
            color_mesh = generate_flat_layer_stl(color_mask, bl, bl + sl, pixel_size, config.layer_height)
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

            if config.use_fill:
                # Part 2: Inverse middle fill
                if inverse_mask.any():
                    mesh_middle = generate_flat_layer_stl(inverse_mask, bl, bl + sl, pixel_size, config.layer_height)
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
