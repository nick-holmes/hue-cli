"""Flat and flat-cap modes for HueCLI STL generation.

Flat mode: per-pixel varying layer counts via Beer-Lambert optimization.
For each pixel, finds the optimal integer layer count (0..max) per filament
that best reproduces the target color. Colors stack cumulatively per-pixel.

Flat-cap: separates transparent filament, adds transparent fill in gaps
and cap layer on top.
"""

import numpy as np
import trimesh
import logging

from ..color_science import (
    sort_filaments_by_luminosity,
    compute_flat_layer_counts,
)
from ..mesh import generate_quantized_stl

logger = logging.getLogger(__name__)


def generate(output_base_path, config, processed_image, selected_filaments):
    """Generate flat or flat-cap mode STLs.

    Args:
        output_base_path: Path for output files
        config: PipelineConfig with layer_height, model_height, base_layers, dither_mode, width_mm, use_flat_cap
        processed_image: ProcessedImage with grayscale, image_rgb, alpha_mask, width_px
        selected_filaments: DataFrame of selected filaments

    Returns:
        List of (path, name, layer_start, layer_end) tuples
    """
    num_layers = int(config.model_height / config.layer_height)
    pixel_size = config.width_mm / processed_image.width_px
    generated_files = []

    if config.use_flat_cap:
        logger.info("Generating flat-cap mode: flat color slabs + transparent cap")
        sorted_filaments = sort_filaments_by_luminosity(selected_filaments)
        transparent_filament = sorted_filaments.iloc[-1]
        color_filaments = sorted_filaments.iloc[:-1].reset_index(drop=True)

        cap_layers = config.base_layers
        cap_height_layers = cap_layers
        color_total_layers = num_layers - cap_layers

        logger.info(f"Transparent (cap + fill): {transparent_filament['name']} "
                     f"(TD={transparent_filament['transmission_distance']:.1f})")
        logger.info(f"Color filaments: {len(color_filaments)}, "
                     f"cap layers: {cap_layers}, color layers: {color_total_layers}")
    else:
        logger.info("Generating flat mode: multi-level per pixel")
        color_filaments = sort_filaments_by_luminosity(selected_filaments)
        transparent_filament = None
        cap_layers = 0
        cap_height_layers = 0
        color_total_layers = num_layers

    alpha_pixels = processed_image.alpha_mask >= 0.5

    # Compute per-pixel layer counts via budget-constrained Beer-Lambert optimization
    pixel_counts = compute_flat_layer_counts(
        color_filaments, color_total_layers, processed_image.image_rgb, alpha_pixels,
        config.layer_height, dither_mode=config.dither_mode)

    # Cumulative per-pixel stacking: each color sits on top of the previous
    N = len(color_filaments)
    z_cursor = np.zeros_like(processed_image.grayscale)

    for k in range(N):
        filament = color_filaments.iloc[k]
        color_name = filament['name'].replace(' ', '_').replace('/', '_')

        z_bottom_k = z_cursor.copy()
        z_top_k = z_bottom_k + pixel_counts[:, :, k] * config.layer_height

        pixel_mask = pixel_counts[:, :, k] > 0
        pixel_count = int(pixel_mask.sum())

        if pixel_count > 0:
            output_path = output_base_path.parent / f"{output_base_path.stem}_{color_name}.stl"
            mesh_obj = generate_quantized_stl(z_bottom_k, z_top_k, pixel_mask, pixel_size, config.layer_height)
            if len(mesh_obj.vertices) > 0:
                mesh_obj.export(str(output_path))
                generated_files.append((output_path, filament['name'], k, k + 1))
                max_layers = int(pixel_counts[:, :, k].max())
                logger.debug(f"  {color_name}: max {max_layers} layers, {pixel_count} pixels")
        else:
            logger.debug(f"  No pixels for {color_name} - skipping")

        # Advance cursor for all pixels (even those with 0 layers for this color)
        z_cursor = z_top_k

    # Flat-cap: generate transparent fill + cap as single STL
    if config.use_flat_cap and transparent_filament is not None:
        color_name = transparent_filament['name'].replace(' ', '_').replace('/', '_')
        output_path = output_base_path.parent / f"{output_base_path.stem}_{color_name}.stl"

        vertices_list = []
        faces_list = []
        vertex_offset = 0

        # Per-pixel total color height = z_cursor after all colors
        # Fill from z_cursor up to max total height across all pixels
        max_color_height = float(np.max(z_cursor[alpha_pixels])) if alpha_pixels.any() else 0.0

        # Transparent fill: from per-pixel color top to max color height
        fill_z_bottom = z_cursor
        fill_z_top = np.where(alpha_pixels, max_color_height, 0.0)
        fill_mask = alpha_pixels & (fill_z_top > fill_z_bottom + config.layer_height * 0.5)

        if fill_mask.any():
            fill_mesh = generate_quantized_stl(fill_z_bottom, fill_z_top, fill_mask, pixel_size, config.layer_height)
            if len(fill_mesh.vertices) > 0:
                vertices_list.append(fill_mesh.vertices)
                faces_list.append(fill_mesh.faces + vertex_offset)
                vertex_offset += len(fill_mesh.vertices)

        # Cap layer on top of max color height
        cap_z_bottom = np.where(alpha_pixels, max_color_height, 0.0)
        cap_z_top = np.where(alpha_pixels, max_color_height + cap_height_layers * config.layer_height, 0.0)
        cap_mask = alpha_pixels
        cap_mesh = generate_quantized_stl(cap_z_bottom, cap_z_top, cap_mask, pixel_size, config.layer_height)
        if len(cap_mesh.vertices) > 0:
            vertices_list.append(cap_mesh.vertices)
            faces_list.append(cap_mesh.faces + vertex_offset)

        if vertices_list:
            combined_vertices = np.vstack(vertices_list)
            combined_faces = np.vstack(faces_list)
            combined_mesh = trimesh.Trimesh(
                vertices=combined_vertices, faces=combined_faces, process=False)
            combined_mesh.export(str(output_path))
            generated_files.append((output_path, transparent_filament['name'], N, N + 1))
            logger.info(f"  Transparent fill + cap: {cap_height_layers} cap layers")

    logger.info(f"Flat mode: generated {len(generated_files)} STL files")
    return generated_files
