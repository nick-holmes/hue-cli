"""Standard topographical mode for HueCLI STL generation.

Generates stacked topographical STLs with fixed color boundaries.
Uses global heightmap (brightness -> height) with TD-proportional layer
allocation and quantized STL generation per color band.
"""

import numpy as np
import logging

from color_science import (
    sort_filaments_by_luminosity,
    apply_contrast_enhancement,
    apply_unsharp_mask,
    allocate_layers_td_proportional,
    compute_heightmap,
)
from mesh import generate_color_band_stls

logger = logging.getLogger(__name__)


def generate(output_base_path, config, processed_image, selected_filaments):
    """Generate standard topographical STLs with fixed color boundaries.

    Args:
        output_base_path: Path for output files
        config: PipelineConfig with layer_height, model_height, contrast_strength, width_mm
        processed_image: ProcessedImage with grayscale, alpha_mask, width_px
        selected_filaments: DataFrame of selected filaments

    Returns:
        List of (path, name, layer_start, layer_end) tuples
    """
    num_layers = int(config.model_height / config.layer_height)
    num_colors = len(selected_filaments)
    pixel_size = config.width_mm / processed_image.width_px

    sorted_filaments = sort_filaments_by_luminosity(selected_filaments)
    enhanced_grayscale = apply_contrast_enhancement(
        processed_image.grayscale.copy(), processed_image.alpha_mask, config.contrast_strength)
    enhanced_grayscale = apply_unsharp_mask(enhanced_grayscale, processed_image.alpha_mask)
    alpha_pixels = processed_image.alpha_mask >= 0.5

    filament_tds = np.array([f['transmission_distance'] for _, f in sorted_filaments.iterrows()])
    layer_counts, layer_boundaries, z_boundaries = allocate_layers_td_proportional(
        filament_tds, num_layers, config.layer_height)

    for idx in range(num_colors):
        name = sorted_filaments.iloc[idx]['name']
        lc = layer_counts[idx]
        logger.info(f"  {name}: TD={filament_tds[idx]:.1f}mm -> {lc} layers ({lc * config.layer_height:.2f}mm)")

    pixel_height = compute_heightmap(enhanced_grayscale, alpha_pixels, config.model_height, config.layer_height)

    return generate_color_band_stls(
        sorted_filaments, pixel_height, z_boundaries, layer_boundaries,
        alpha_pixels, output_base_path, pixel_size, config.layer_height)
