"""Exploded-multi mode for HueCLI STL generation.

Multi-color sandwiches with up to 3 colors each. Same pixel-level
Beer-Lambert optimization as exploded mode but with higher per-color caps.
After optimization, color-level pairs are packed into physical sandwiches
via greedy bin-packing (up to 3 non-overlapping color masks per sandwich).
"""

import numpy as np
import trimesh
import logging

from ..color_science import compute_exploded_layer_counts
from ..mesh import generate_flat_layer_stl

logger = logging.getLogger(__name__)


def pack_sandwich_groups(layer_counts, alpha_mask):
    """Pack (color, level) pairs into physical sandwiches with up to 3 colors each.

    Uses greedy bin-packing: sorts pairs by mask size (largest first),
    assigns each to the first sandwich where its mask doesn't overlap
    with existing masks and the sandwich has < 3 colors.

    Args:
        layer_counts: ndarray (H, W, num_colors) int32
        alpha_mask: 2D array, valid pixels where >= 0.5

    Returns:
        List of sandwich groups. Each group is a list of (color_idx, level, mask) tuples.
    """
    num_colors = layer_counts.shape[2]
    valid_mask = alpha_mask >= 0.5

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


def generate(output_base_path, config, processed_image, selected_filaments):
    """Generate exploded-multi mode: multi-color sandwiches with up to 3 colors each.

    Args:
        output_base_path: Path for output files
        config: PipelineConfig with layer_height, model_height, sandwich_layers, use_fill,
                base_layers, max_color_sandwiches, dither_mode, width_mm
        processed_image: ProcessedImage with grayscale, image_rgb, alpha_mask, width_px
        selected_filaments: DataFrame of selected filaments

    Returns:
        List of (path, name, layer_start, layer_end) tuples
    """
    num_colors = len(selected_filaments)
    pixel_size = config.width_mm / processed_image.width_px
    generated_files = []

    num_color_filaments = num_colors - 1
    if num_color_filaments <= 0:
        logger.warning("Exploded-multi mode requires at least 1 color + transparent")
        return generated_files

    color_filaments = selected_filaments.iloc[:num_color_filaments]
    transparent_filament = selected_filaments.iloc[-1]

    total_sandwiches = int(config.model_height / config.layer_height)
    logger.info(f"Exploded-multi mode: {total_sandwiches} sandwich budget, {num_color_filaments} colors")

    # Higher cap per color since multi-color packing is more efficient.
    # Iteratively reduce caps if packing exceeds the sandwich budget.
    MAX_SANDWICHES_PER_COLOR = config.max_color_sandwiches if config.max_color_sandwiches is not None else 5

    H, W = processed_image.grayscale.shape
    num_physical = 0

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
        layer_counts = compute_exploded_layer_counts(
            color_filaments, transparent_filament, total_sandwiches, per_color_caps,
            (H, W), processed_image.image_rgb, processed_image.alpha_mask,
            config.layer_height, config.sandwich_layers, config.use_fill,
            dither_mode=config.dither_mode
        )

        # Pack into physical sandwiches (up to 3 colors each)
        sandwich_groups = pack_sandwich_groups(layer_counts, processed_image.alpha_mask)
        num_physical = len(sandwich_groups)

        if num_physical <= total_sandwiches:
            logger.info(f"Exploded-multi: packed into {num_physical} physical sandwiches "
                         f"(budget: {total_sandwiches}, cap: {current_cap})")
            break
        else:
            logger.info(f"Exploded-multi: cap={current_cap} produced {num_physical} sandwiches "
                         f"(budget: {total_sandwiches}), reducing cap...")
    else:
        # Even cap=1 exceeded budget -- use what we have
        logger.info(f"Exploded-multi: packed into {num_physical} physical sandwiches "
                     f"(budget: {total_sandwiches}, cap: 1)")

    all_pixels_mask = processed_image.alpha_mask >= 0.5

    # Sandwich layer structure (same as exploded)
    bl = config.base_layers
    sl = config.sandwich_layers
    layers_per_sandwich = bl + sl + (1 if config.use_fill else 0)

    # Pre-generate shared full-plate meshes (same for every sandwich)
    mesh_bottom = generate_flat_layer_stl(all_pixels_mask, 0, bl, pixel_size, config.layer_height)
    mesh_top = generate_flat_layer_stl(all_pixels_mask, bl + sl, bl + sl + 1, pixel_size, config.layer_height) if config.use_fill else None

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
            logger.debug(f"  Sandwich {sandwich_num}/{num_physical}: "
                          f"{filament['name']} level {level} ({pixel_count} pixels)")

            color_stl_path = (output_base_path.parent /
                f"{output_base_path.stem}_S{sandwich_num:02d}_{color_name}_color.stl")
            color_mesh = generate_flat_layer_stl(mask, bl, bl + sl, pixel_size, config.layer_height)
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

        if config.use_fill:
            if inverse_mask.any():
                mesh_middle = generate_flat_layer_stl(inverse_mask, bl, bl + sl, pixel_size, config.layer_height)
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
