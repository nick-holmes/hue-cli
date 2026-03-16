"""Exploded CMYK mode for HueCLI STL generation.

Fixed CMYK primaries with configurable sandwiches each.
Delegates to the exploded module with an appropriate max_sandwiches_per_color override.
"""


def generate(output_base_path, config, processed_image, selected_filaments):
    """Generate exploded CMYK mode: 4 primaries with configurable sandwiches each.

    Args:
        output_base_path: Path for output files
        config: PipelineConfig with max_color_sandwiches and all exploded mode config
        processed_image: ProcessedImage with grayscale, image_rgb, alpha_mask, width_px
        selected_filaments: DataFrame of selected filaments

    Returns:
        List of (path, name, layer_start, layer_end) tuples
    """
    from .exploded import generate as exploded_generate

    cap = config.max_color_sandwiches if config.max_color_sandwiches is not None else 1
    return exploded_generate(output_base_path, config, processed_image, selected_filaments,
                              max_sandwiches_per_color=cap)
