"""Standard topographical mode for HueCLI STL generation.

Generates stacked topographical STLs via brightness heightmap + TD-proportional
z-bands. Each pixel is a solid column from z=0 up to a brightness-derived height,
with fixed z-bands per filament color.
"""

import numpy as np
import logging

from ..color_science import (
    sort_filaments_by_luminosity,
    apply_contrast_enhancement,
    apply_unsharp_mask,
    allocate_layers_td_proportional,
    compute_heightmap,
)
from ..mesh import generate_color_band_stls

logger = logging.getLogger(__name__)


def generate(output_base_path, config, processed_image, selected_filaments):
    """Generate standard topographical STLs with heightmap + z-band pipeline.

    Args:
        output_base_path: Path for output files
        config: PipelineConfig with layer_height, model_height, width_mm, contrast_strength
        processed_image: ProcessedImage with grayscale, alpha_mask, width_px
        selected_filaments: DataFrame of selected filaments

    Returns:
        List of (path, name, layer_start, layer_end) tuples
    """
    num_layers = int(config.model_height / config.layer_height)
    pixel_size = config.width_mm / processed_image.width_px

    sorted_filaments = sort_filaments_by_luminosity(selected_filaments)
    enhanced_grayscale = apply_contrast_enhancement(
        processed_image.grayscale.copy(), processed_image.alpha_mask, config.contrast_strength)
    enhanced_grayscale = apply_unsharp_mask(enhanced_grayscale, processed_image.alpha_mask)
    alpha_pixels = processed_image.alpha_mask >= 0.5

    filament_tds = np.array([f['transmission_distance'] for _, f in sorted_filaments.iterrows()])
    layer_counts, layer_boundaries, z_boundaries = allocate_layers_td_proportional(
        filament_tds, num_layers, config.layer_height)

    pixel_height = compute_heightmap(enhanced_grayscale, alpha_pixels, config.model_height, config.layer_height,
                                     min_height=float(z_boundaries[1]))

    return generate_color_band_stls(
        sorted_filaments, pixel_height, z_boundaries, layer_boundaries,
        alpha_pixels, output_base_path, pixel_size, config.layer_height)
