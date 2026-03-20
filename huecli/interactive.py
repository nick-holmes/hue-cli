"""Interactive prompts — fill missing CLI values via user input."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


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


def fill_interactive(args):
    """Fill missing config fields via user prompts.

    Args:
        args: argparse Namespace from cli.parse_args()

    Returns:
        dict with all resolved configuration values
    """
    filaments_csv = args.filaments if args.filaments is not None else prompt_with_default(
        "Filament library CSV path",
        "data/filaments.csv",
        str
    )

    # Validate filaments CSV exists
    if not Path(filaments_csv).exists():
        logger.error(f"Filaments CSV not found: {filaments_csv}")
        return None

    color_count = args.colors if args.colors is not None else None  # Resolved after mode flags

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

    # Model height — compute recommended default for standard mode when color count is known
    if args.model_height is not None:
        model_height = args.model_height
    else:
        model_height = prompt_with_default(
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
        return None

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

    if args.mode is not None:
        mode = args.mode
    else:
        print("\nAvailable modes:")
        print("  1. standard        - Multi-color stacked topographical STLs")
        print("  2. flat            - Flat color slabs, multi-color per pixel")
        print("  3. flat-cap        - Flat + transparent cap layer")
        print("  4. exploded        - Each color as standalone transparent sandwich")
        print("  5. exploded-multi  - Up to 3 colors per sandwich, higher fidelity")
        print("  6. exploded-cmyk   - 4 CMYK primaries, max 8 sandwiches")
        mode_choice = input("Select mode [1]: ").strip() or "1"
        mode_map = {
            '1': 'standard', '2': 'flat', '3': 'flat-cap',
            '4': 'exploded', '5': 'exploded-multi',
            '6': 'exploded-cmyk',
        }
        mode = mode_map.get(mode_choice, mode_choice)  # Accept number or name

    # Derive boolean flags from mode
    use_flat = mode == 'flat'
    use_flat_cap = mode == 'flat-cap'
    use_exploded = mode == 'exploded'
    use_exploded_multi = mode == 'exploded-multi'
    use_exploded_cmyk = mode == 'exploded-cmyk'
    exploded_any = use_exploded or use_exploded_multi or use_exploded_cmyk

    # Resolve color_count now that mode flags are known
    if use_exploded_cmyk:
        color_count = 4  # CMYK always uses exactly 4 colors
    elif color_count is None and not exploded_any:
        color_count = prompt_with_default("Number of colors/filaments", 4, int)

    if not exploded_any and use_flat_cap and color_count < 2:
        logger.warning("Flat-cap requires at least 2 colors. Setting to 2.")
        color_count = 2

    # Log recommended model height for standard mode
    if mode == 'standard' and color_count is not None:
        recommended_layers = 2 + (color_count - 1)  # base=2, rest=1 each minimum
        recommended_height = recommended_layers * layer_height
        logger.info(f"Standard mode: minimum {recommended_layers} layers ({recommended_height:.2f}mm) "
                    f"for {color_count} colors (2 base + {color_count - 1} glaze)")

    # Resolve sandwich_layers (color layers per sandwich in exploded modes)
    if args.sandwich_layers is not None:
        sandwich_layers = args.sandwich_layers
    elif use_exploded_cmyk:
        sandwich_layers = 3  # CMYK needs thicker color layers for high-TD filaments
    else:
        sandwich_layers = 1  # Standard exploded default

    # Resolve fill (transparent fill in exploded sandwiches)
    use_fill = (args.fill.lower() in ('y', 'yes', 'true', '1')) if args.fill is not None else False

    # Resolve base_layers (transparent base layers per sandwich / cap layers in flat-cap mode)
    if args.base_layers is not None:
        base_layers = args.base_layers
    elif use_flat_cap:
        base_layers = 1
    elif exploded_any:
        base_layers = 3
    else:
        base_layers = 2

    # Resolve dither mode
    dither_mode = args.dither if args.dither is not None else 'none'

    # Resolve max_color_sandwiches (max sandwiches per colour in exploded modes)
    if args.max_color_sandwiches is not None:
        max_color_sandwiches = args.max_color_sandwiches
    elif use_exploded_cmyk:
        max_color_sandwiches = 1
    elif use_exploded_multi:
        max_color_sandwiches = 5
    else:
        max_color_sandwiches = 3

    return {
        'filaments_csv': filaments_csv,
        'color_count': color_count,
        'nozzle_diameter': nozzle_diameter,
        'layer_height': layer_height,
        'model_height': model_height,
        'width': width,
        'height': height,
        'output_name': output_name,
        'min_color_difference': min_color_difference,
        'mode': mode,
        'use_flat': use_flat,
        'use_flat_cap': use_flat_cap,
        'use_exploded': use_exploded,
        'use_exploded_multi': use_exploded_multi,
        'use_exploded_cmyk': use_exploded_cmyk,
        'exploded_any': exploded_any,
        'sandwich_layers': sandwich_layers,
        'use_fill': use_fill,
        'base_layers': base_layers,
        'dither_mode': dither_mode,
        'max_color_sandwiches': max_color_sandwiches,
    }
