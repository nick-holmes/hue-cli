#!/usr/bin/env python3
"""
HueCLI - Generate multi-color 3D print STLs from images.
Converts images to topographical STL files with Beer-Lambert color stacking.

Slim orchestrator: parse -> prompt -> pipeline -> preview -> generate.
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from skimage import color

from config import PipelineConfig, ProcessedImage, COLOR_SCHEMES
from filaments import FilamentLibrary
from image import ImageProcessor
from cli import parse_args
from interactive import fill_interactive
from preview import show_3d_preview
import color_science
import modes

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main CLI entry point"""
    args = parse_args()

    try:
        # Validate image exists
        if not Path(args.image).exists():
            logger.error(f"Image file not found: {args.image}")
            return 1

        print("\nHueCLI - Multi-Color 3D Print STL Generator")
        print(f"Input: {args.image}\n")

        # Fill missing values interactively
        cfg = fill_interactive(args)
        if cfg is None:
            return 1

        width = cfg['width']
        mode = cfg['mode']
        layer_height = cfg['layer_height']
        model_height = cfg['model_height']
        color_count = cfg['color_count']
        min_color_difference = cfg['min_color_difference']
        sandwich_layers = cfg['sandwich_layers']
        use_fill = cfg['use_fill']
        base_layers = cfg['base_layers']
        dither_mode = cfg['dither_mode']
        max_color_sandwiches = cfg['max_color_sandwiches']
        nozzle_diameter = cfg['nozzle_diameter']
        use_flat_cap = cfg['use_flat_cap']
        use_exploded_cmyk = cfg['use_exploded_cmyk']
        exploded_any = cfg['exploded_any']

        print("\n" + "=" * 60)

        logger.info("Starting STL generation...")

        # 1. Load filament library
        filament_lib = FilamentLibrary(cfg['filaments_csv'])

        # 2. Load and prepare image
        img_processor = ImageProcessor(args.image, width, color_count or 4)
        img_processor.load_and_prepare(nozzle_diameter=nozzle_diameter)

        # Apply user-requested flip
        if args.flip in ('horizontal', 'both'):
            img_processor.image = np.fliplr(img_processor.image)
            img_processor.image_lab = np.fliplr(img_processor.image_lab)
            if img_processor.alpha_mask is not None:
                img_processor.alpha_mask = np.fliplr(img_processor.alpha_mask)
            logger.info("Applied horizontal flip")
        if args.flip in ('vertical', 'both'):
            img_processor.image = np.flipud(img_processor.image)
            img_processor.image_lab = np.flipud(img_processor.image_lab)
            if img_processor.alpha_mask is not None:
                img_processor.alpha_mask = np.flipud(img_processor.alpha_mask)
            logger.info("Applied vertical flip")

        # Apply colour scheme remapping
        if args.scheme is not None:
            scheme_name = args.scheme
            palette_hex = COLOR_SCHEMES[scheme_name]
            palette_rgb = np.array([[int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)]
                                    for h in palette_hex]) / 255.0
            palette_lab = color.rgb2lab(palette_rgb.reshape(-1, 1, 3)).reshape(-1, 3)

            pixels_lab = img_processor.image_lab.reshape(-1, 3)
            from scipy.spatial import cKDTree
            tree = cKDTree(palette_lab)
            _, indices = tree.query(pixels_lab, k=1)

            remapped_rgb = palette_rgb[indices].reshape(img_processor.image.shape)
            img_processor.image = remapped_rgb
            img_processor.image_lab = color.rgb2lab(remapped_rgb)

            logger.info(f"Remapped image to '{scheme_name}' scheme ({len(palette_hex)} colours)")

            if color_count is None and not use_exploded_cmyk:
                color_count = len(palette_hex)
                img_processor.color_count = color_count
                logger.info(f"Defaulting colour count to scheme size: {color_count}")

        if exploded_any and not use_exploded_cmyk and color_count is None:
            # Iterative color count optimization
            from itertools import product as _prod
            from scipy.spatial import cKDTree

            grayscale_tmp = (0.2126 * img_processor.image[:, :, 0] +
                             0.7152 * img_processor.image[:, :, 1] +
                             0.0722 * img_processor.image[:, :, 2])
            if grayscale_tmp.max() > grayscale_tmp.min():
                grayscale_tmp = (grayscale_tmp - grayscale_tmp.min()) / (grayscale_tmp.max() - grayscale_tmp.min())

            total_sandwiches = int(model_height / layer_height)
            MAX_S_PER_COLOR = max_color_sandwiches
            max_try_k = min(total_sandwiches, 12)
            mode_name = "exploded-multi" if cfg['use_exploded_multi'] else "exploded"
            logger.info(f"Auto-determining optimal color count for {mode_name} (K=3..{max_try_k}, cap={MAX_S_PER_COLOR})...")
            k_scores = []
            H, W = img_processor.image.shape[:2]

            for try_k in range(3, max_try_k + 1):
                img_processor.color_count = try_k
                try:
                    dom_colors, _, _ = img_processor.quantize_colors()
                except Exception:
                    continue

                try_filaments = filament_lib.select_for_exploded(
                    dom_colors,
                    min_color_difference=min_color_difference,
                    layer_height=layer_height,
                    model_height=model_height,
                )

                num_color_fils = len(try_filaments) - 1
                color_fils = try_filaments.iloc[:num_color_fils]
                trans_fil = try_filaments.iloc[-1]

                caps = [min(MAX_S_PER_COLOR, total_sandwiches) for _ in range(num_color_fils)]

                combo_size = 1
                for c in caps:
                    combo_size *= (c + 1)
                while combo_size > 500_000:
                    max_i = caps.index(max(caps))
                    caps[max_i] -= 1
                    combo_size = 1
                    for c in caps:
                        combo_size *= (c + 1)

                layer_counts = color_science.compute_exploded_layer_counts(
                    color_fils, trans_fil, total_sandwiches, caps,
                    (H, W), img_processor.image, img_processor.alpha_mask,
                    layer_height, sandwich_layers, use_fill,
                    dither_mode=dither_mode)

                fil_tds = np.array([f['transmission_distance'] for _, f in color_fils.iterrows()])
                all_rgbs = np.vstack([
                    np.array([f['rgb'] for _, f in color_fils.iterrows()]),
                    np.array(trans_fil['rgb']).reshape(1, 3)
                ])
                all_tds = np.append(fil_tds, trans_fil['transmission_distance'])

                ranges = [range(0, c + 1) for c in caps]
                all_combos = np.array(list(_prod(*ranges)))
                valid = all_combos.sum(axis=1) <= total_sandwiches
                combos = all_combos[valid]
                trans_l = total_sandwiches - combos.sum(axis=1)
                color_thickness = layer_height * sandwich_layers
                trans_per_sandwich = color_thickness if use_fill else layer_height
                thickness = np.column_stack([combos * color_thickness, (trans_l * trans_per_sandwich).reshape(-1, 1)])
                sim_rgb = color_science.vectorized_beer_lambert(all_rgbs, all_tds, thickness)
                sim_lab = color.rgb2lab(sim_rgb.reshape(-1, 1, 3)).reshape(-1, 3)
                tree = cKDTree(sim_lab)
                target_lab = color.rgb2lab(img_processor.image).reshape(-1, 3)
                dists, _ = tree.query(target_lab, k=1)
                valid_mask = (img_processor.alpha_mask >= 0.5).ravel()
                mean_de = float(np.mean(dists[valid_mask]))

                fil_names = [f['name'] for _, f in color_fils.iterrows()]
                k_scores.append((try_k, mean_de, fil_names))
                logger.debug(f"  K={try_k}: mean deltaE={mean_de:.2f} ({', '.join(fil_names)})")

            if k_scores:
                best_score = min(s for _, s, _ in k_scores)
                for try_k, score, names in k_scores:
                    if score <= best_score + 2.0:
                        color_count = try_k
                        logger.info(f"Selected K={color_count} (deltaE={score:.2f}, best={best_score:.2f})")
                        break
            else:
                color_count = 6

            img_processor.color_count = color_count
        elif exploded_any:
            img_processor.color_count = color_count

        # 3. Select filaments
        num_layers = int(model_height / layer_height)

        if args.use_filaments:
            # Explicit filament names from CLI — bypass selection and SA
            requested_names = [n.strip() for n in args.use_filaments.split(',')]
            rows = []
            for name in requested_names:
                matches = filament_lib.df[filament_lib.df['name'] == name]
                if len(matches) == 0:
                    logger.error(f"Filament not found: '{name}'")
                    logger.info("Available filaments:")
                    for _, row in filament_lib.df.iterrows():
                        logger.info(f"  {row['name']}")
                    return 1
                rows.append(matches.iloc[0])
            selected_filaments = pd.DataFrame(rows).reset_index(drop=True)
            color_count = len(selected_filaments)
            logger.info(f"Using specified filaments: {', '.join(requested_names)}")
        else:
            # Quantize colors and select filaments
            dominant_colors_lab, kmeans, sorted_indices = img_processor.quantize_colors()

            if use_exploded_cmyk:
                selected_filaments = filament_lib.select_for_exploded_cmyk(
                    layer_height=layer_height,
                    model_height=model_height,
                    sandwich_layers=sandwich_layers,
                    max_color_sandwiches=max_color_sandwiches,
                )
            elif exploded_any:
                selected_filaments = filament_lib.select_for_exploded(
                    dominant_colors_lab,
                    min_color_difference=min_color_difference,
                    layer_height=layer_height,
                    model_height=model_height,
                )
            else:
                selected_filaments = filament_lib.select_best_filaments(
                    dominant_colors_lab, color_count,
                    layer_height=layer_height,
                    min_color_difference=min_color_difference,
                    use_flat_cap=use_flat_cap,
                    model_height=model_height,
                )

            logger.info("Selected filaments:")
            for i, row in selected_filaments.iterrows():
                logger.info(f"  {i+1}. {row['name']} ({row['color_hex']}) TD={row['transmission_distance']:.1f}mm")

            # Always-on filament optimization via simulated annealing
            selected_filaments = filament_lib.optimize_filament_set(
                selected_filaments, img_processor.image, img_processor.alpha_mask,
                layer_height, model_height, num_layers,
                mode=mode, sandwich_layers=sandwich_layers, use_fill=use_fill,
            )

        # 4c. Gamut coverage report (skip when filaments explicitly specified)
        if not args.use_filaments:
            from color_science import compute_effective_color as _eff_color
            from color_science import allocate_layers_td_proportional as _alloc
            sorted_tds = np.array([f['transmission_distance'] for _, f in selected_filaments.iterrows()])
            layer_counts_report, _, _ = _alloc(sorted_tds, num_layers, layer_height)
            logger.info("Gamut coverage report:")
            for i, (_, row) in enumerate(selected_filaments.iterrows()):
                thickness = layer_counts_report[i] * layer_height
                rendered_lab = _eff_color(row['rgb'], row['transmission_distance'], thickness)
                best_de = float('inf')
                for t_idx, t_lab in enumerate(dominant_colors_lab):
                    de = float(np.sqrt(np.sum((np.array(rendered_lab) - np.array(t_lab)) ** 2)))
                    if de < best_de:
                        best_de = de
                logger.info(f"  {row['name']}: {thickness:.2f}mm thick, nearest target deltaE={best_de:.1f}")

        # 5. Convert to grayscale
        logger.info("Converting to grayscale...")
        grayscale = (0.2126 * img_processor.image[:, :, 0] +
                     0.7152 * img_processor.image[:, :, 1] +
                     0.0722 * img_processor.image[:, :, 2])
        if grayscale.max() > grayscale.min():
            grayscale = (grayscale - grayscale.min()) / (grayscale.max() - grayscale.min())
        else:
            logger.warning("WARNING: Image has no brightness variation!")
            grayscale = np.ones_like(grayscale) * 0.5
        logger.info(f"Grayscale range: {grayscale.min():.3f} to {grayscale.max():.3f}")

        # Build config + processed image for new module API
        pipeline_config = PipelineConfig(
            image_path=args.image,
            layer_height=layer_height,
            model_height=model_height,
            width_mm=width,
            mode=mode,
            sandwich_layers=sandwich_layers,
            use_fill=use_fill,
            base_layers=base_layers,
            max_color_sandwiches=max_color_sandwiches,
            dither_mode=dither_mode,
        )
        processed_image = ProcessedImage(
            image_rgb=img_processor.image,
            image_lab=img_processor.image_lab,
            grayscale=grayscale,
            alpha_mask=img_processor.alpha_mask,
        )

        # 6. Preview loop
        show_3d_preview(pipeline_config, processed_image, selected_filaments)

        while True:
            print("\n" + "=" * 60)
            print("3D preview opened in browser. What would you like to do?")
            print("1. Generate STLs")
            print("2. Adjust colors (reduce delta-E, re-preview)")
            print("3. Cancel")
            print("=" * 60)
            choice = input("Enter choice [1]: ").strip() or "1"

            if choice == "3":
                logger.info("Cancelled by user")
                return 0
            elif choice == "2":
                min_color_difference = min_color_difference * 0.7
                logger.info(f"Reducing color delta to {min_color_difference:.1f} and re-selecting filaments...")
                if use_exploded_cmyk:
                    selected_filaments = filament_lib.select_for_exploded_cmyk(
                        layer_height=layer_height, model_height=model_height,
                        sandwich_layers=sandwich_layers, max_color_sandwiches=max_color_sandwiches,
                    )
                elif exploded_any:
                    selected_filaments = filament_lib.select_for_exploded(
                        dominant_colors_lab, min_color_difference=min_color_difference,
                        layer_height=layer_height, model_height=model_height,
                    )
                else:
                    selected_filaments = filament_lib.select_best_filaments(
                        dominant_colors_lab, color_count, layer_height=layer_height,
                        min_color_difference=min_color_difference, use_flat_cap=use_flat_cap,
                        randomize=False, model_height=model_height,
                    )
                logger.info("New filaments selected:")
                for i, row in selected_filaments.iterrows():
                    logger.info(f"  {i+1}. {row['name']} ({row['color_hex']}) - L={row['lab'][0]:.1f}")
                show_3d_preview(pipeline_config, processed_image, selected_filaments)
            else:
                break

        logger.info("Generating STLs...")

        # 7. Create output directory
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)

        output_filename = Path(cfg['output_name']).name
        output_path = output_dir / output_filename

        # 7a. Generate STLs via mode dispatch
        logger.info("\nGenerating STLs...")
        generated_files = modes.generate(mode, output_path, pipeline_config,
                                          processed_image, selected_filaments)
        logger.info(f"Generated {len(generated_files)} STL file(s)")

        # 8. Generate description file
        desc_path = output_path.with_suffix('.txt')
        with open(desc_path, 'w') as f:
            f.write("HueCLI Multi-Color STL Generation\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Model: {width}mm wide x {model_height}mm tall\n")
            f.write(f"Layer height: {layer_height}mm\n")
            f.write(f"Total layers: {int(model_height / layer_height)}\n")
            f.write(f"Colors: {len(selected_filaments)}\n\n")

            f.write("Generated Files:\n")
            f.write("-" * 60 + "\n")
            for file_path, color_name, layer_start, layer_end in generated_files:
                layer_range_mm = f"{layer_start * layer_height:.2f}-{layer_end * layer_height:.2f}mm"
                f.write(f"{file_path.name}\n")
                f.write(f"  {color_name}: layers {layer_start}-{layer_end} ({layer_range_mm})\n\n")

            f.write("\nPrinting Instructions:\n")
            f.write("-" * 60 + "\n")
            if use_exploded_cmyk:
                f.write("EXPLODED CMYK MODE\n")
                f.write("Uses 4 subtractive primaries (Cyan, Magenta, Yellow, Black) + transparent.\n")
                f.write(f"Each colour has up to {max_color_sandwiches} intensity level(s) = max {4 * max_color_sandwiches} sandwiches.\n")
                f.write(f"Each sandwich has {base_layers} base layer(s) + {sandwich_layers} colour layer(s)")
                if use_fill:
                    f.write(f" + 1 top = {base_layers + sandwich_layers + 1} layers.\n\n")
                else:
                    f.write(f" = {base_layers + sandwich_layers} layers.\n\n")
                f.write("For each sandwich:\n")
                f.write("1. Load both STLs (_color.stl and _transparent.stl) into slicer\n")
                f.write("2. Assign the colour filament to the _color.stl part\n")
                f.write("3. Assign transparent filament to the _transparent.stl part\n")
                layers_total = base_layers + sandwich_layers + (1 if use_fill else 0)
                f.write(f"4. Print as a single {layers_total}-layer print\n\n")
                f.write("After printing all sandwiches:\n")
                f.write("5. Stack all sandwiches in order and backlight for effect\n")
            elif cfg['use_exploded_multi']:
                layers_total = base_layers + sandwich_layers + (1 if use_fill else 0)
                f.write("EXPLODED-MULTI MODE\n")
                f.write(f"Each sandwich has {base_layers} base + {sandwich_layers} colour layer(s) with up to 3 colors")
                if use_fill:
                    f.write(f" + 1 top = {layers_total} layers.\n")
                else:
                    f.write(f" = {layers_total} layers.\n")
                f.write("Requires a multi-material printer (or filament swaps) per sandwich.\n\n")
                f.write("For each sandwich (S01, S02, ...):\n")
                f.write("1. Load the _transparent.stl and all _color.stl files for that sandwich\n")
                f.write("2. Assign each color filament to its _color.stl part\n")
                f.write("3. Assign transparent filament to the _transparent.stl part\n")
                f.write(f"4. Print as a single {layers_total}-layer print\n\n")
                f.write("After printing all sandwiches:\n")
                f.write("5. Stack all sandwiches in order and backlight for effect\n")
            elif cfg['use_exploded']:
                layers_total = base_layers + sandwich_layers + (1 if use_fill else 0)
                f.write("EXPLODED MODE\n")
                f.write(f"Each color is a standalone sandwich: {base_layers} base + {sandwich_layers} colour")
                if use_fill:
                    f.write(f" + 1 top = {layers_total} layers.\n")
                else:
                    f.write(f" = {layers_total} layers.\n")
                f.write("Print each sandwich separately on any single-material printer.\n\n")
                f.write("For each color:\n")
                f.write("1. Load both STLs (_color.stl and _transparent.stl) into slicer\n")
                f.write("2. Assign the color filament to the _color.stl part\n")
                f.write("3. Assign transparent filament to the _transparent.stl part\n")
                f.write(f"4. Print as a single {layers_total}-layer print\n\n")
                f.write("After printing all sandwiches:\n")
                f.write("5. Stack all sandwiches in order and backlight for effect\n")
            elif use_flat_cap:
                f.write("FLAT-CAP MODE\n")
                f.write("Flat color slabs with multi-color per pixel + transparent cap.\n\n")
                f.write("1. Load all STL files into slicer at once\n")
                f.write("2. Right-click each part and assign the corresponding filament\n")
                f.write("3. Print flat (geometry is already optimized)\n")
                f.write(f"4. Transparent cap is {base_layers} layer(s) on top\n")
                f.write("5. Backlight for best effect\n")
            elif cfg['use_flat']:
                f.write("FLAT MODE\n")
                f.write("Flat color slabs with multi-color per pixel.\n\n")
                f.write("1. Load all STL files into slicer at once\n")
                f.write("2. Right-click each part and assign the corresponding filament\n")
                f.write("3. Print flat (geometry is already optimized)\n")
                f.write("4. Backlight for best effect\n")
            else:
                f.write("1. Load all STL files into slicer at once\n")
                f.write("2. Right-click each part and assign the corresponding filament\n")
                f.write("3. Print flat (geometry is already optimized)\n")
                f.write("4. Backlight for best effect\n")

        logger.info(f"\nDone! Generated {len(generated_files)} STL files in {output_dir}/")
        for file_path, color_name, layer_start, layer_end in generated_files:
            logger.info(f"  {file_path.name} ({color_name})")
        logger.info(f"  {desc_path.name} (printing instructions)")

        return 0

    except Exception as e:
        logger.error(f"Error during STL generation: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
