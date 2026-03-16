"""Mode registry and dispatch for HueCLI STL generation."""


def generate(mode, output_base_path, config, processed_image, selected_filaments):
    """Dispatch to the appropriate mode generator.

    Args:
        mode: Mode string ('standard', 'flat', 'flat-cap', 'exploded', 'exploded-multi', 'exploded-cmyk')
        output_base_path: Path for output files
        config: PipelineConfig instance (needs layer_height, model_height, sandwich_layers, use_fill, base_layers, max_color_sandwiches, dither_mode, contrast_strength)
        processed_image: ProcessedImage instance (needs image_rgb, grayscale, alpha_mask)
        selected_filaments: DataFrame of selected filaments

    Returns:
        List of (path, name, layer_start, layer_end) tuples
    """
    if mode == 'exploded-cmyk':
        from .exploded_cmyk import generate as gen
    elif mode == 'exploded-multi':
        from .exploded_multi import generate as gen
    elif mode == 'exploded':
        from .exploded import generate as gen
    elif mode in ('flat', 'flat-cap'):
        from .flat import generate as gen
    else:
        from .standard import generate as gen

    return gen(output_base_path, config, processed_image, selected_filaments)
