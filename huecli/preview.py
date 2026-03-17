"""3D preview generation and browser display.

Standalone functions that replace the preview methods previously on STLGenerator.
No state-swapping hack — downsampled data is passed explicitly to pure functions.
"""

import numpy as np
import trimesh
import logging
import base64
import json
import tempfile
import webbrowser

from .color_science import (
    sort_filaments_by_luminosity,
    compute_flat_layer_counts,
    compute_exploded_layer_counts,
    auto_contrast_preview,
    apply_preview_face_colors,
    apply_contrast_enhancement,
    apply_unsharp_mask,
    allocate_layers_td_proportional,
    compute_heightmap,
    render_standard_preview,
)
from .mesh import (
    generate_topographical_stl,
    greedy_mesh_rects,
    build_box_mesh,
    generate_flat_layer_stl,
)

logger = logging.getLogger(__name__)


def generate_preview_scene(config, processed_image, selected_filaments,
                           max_pixels=500000):
    """Generate a trimesh.Scene with colored meshes for interactive 3D preview.

    Mirrors the heightmap pipeline from generate_all() and returns a Scene
    instead of exporting STLs. Downsamples to max_pixels for browser-renderable
    mesh size while preserving good detail.

    Args:
        config: needs mode flags (use_flat, use_flat_cap, use_exploded, use_exploded_multi,
                use_exploded_cmyk), layer_height, model_height, num_layers, width_mm,
                sandwich_layers, use_fill, base_layers, max_color_sandwiches,
                dither_mode, contrast_strength
        processed_image: needs image_rgb, grayscale, alpha_mask, width_px, height_px
        selected_filaments: DataFrame
        max_pixels: target pixel count for downsampling (default 500K)

    Returns:
        (trimesh.Scene, sorted_filaments_df)
    """
    is_exploded = (config.use_exploded or config.use_exploded_multi
                   or config.use_exploded_cmyk)
    if is_exploded:
        return _generate_exploded_preview_scene(config, processed_image,
                                                 selected_filaments)

    from math import sqrt

    H, W = processed_image.grayscale.shape
    width_mm = config.width_mm

    # Downsample for browser rendering
    ds = max(1, int(sqrt(H * W / max_pixels)))
    ds_grayscale = processed_image.grayscale[::ds, ::ds]
    ds_alpha = processed_image.alpha_mask[::ds, ::ds]
    ds_image_rgb = (processed_image.image_rgb[::ds, ::ds]
                    if processed_image.image_rgb is not None else None)
    ds_H, ds_W = ds_grayscale.shape
    ds_pixel_size = width_mm / ds_W

    logger.info(f"3D preview: {H}x{W} -> {ds_H}x{ds_W} (ds={ds}), "
                 f"pixel_size={ds_pixel_size:.3f}mm")

    sorted_filaments = sort_filaments_by_luminosity(selected_filaments)

    alpha_pixels = ds_alpha >= 0.5
    scene = trimesh.Scene()

    layer_height = config.layer_height
    num_layers = config.num_layers
    model_height = config.model_height

    if config.use_flat or config.use_flat_cap:
        # Flat modes: one mesh per color, cumulative stacking
        if config.use_flat_cap:
            transparent_filament = sorted_filaments.iloc[-1]
            color_filaments = sorted_filaments.iloc[:-1].reset_index(drop=True)
            cap_layers = config.base_layers
            color_total_layers = num_layers - cap_layers
        else:
            color_filaments = sorted_filaments
            transparent_filament = None
            cap_layers = 0
            color_total_layers = num_layers

        pixel_counts = compute_flat_layer_counts(
            color_filaments, color_total_layers, ds_image_rgb, alpha_pixels,
            layer_height)

        # Front-lit preview: show topmost color filament at each pixel.
        # Layers are stacked dark-to-light, so the highest present color
        # filament is the visible surface color. Transparent cap (flat-cap)
        # is ignored since you see through it to the color layers below.
        filament_rgbs = np.array([f['rgb'] for _, f in color_filaments.iterrows()])

        combined_preview = np.ones((ds_H, ds_W, 3)) * 0.95  # default near-white

        # Iterate dark-to-light; each present filament overwrites the preview.
        # The last (lightest) present filament at each pixel wins.
        for k in range(len(color_filaments)):
            present = pixel_counts[:, :, k] > 0
            if present.any():
                combined_preview[present] = filament_rgbs[k]

        combined_preview = auto_contrast_preview(combined_preview, alpha_pixels)

        # Build one mesh per color with cumulative z stacking
        z_cursor = np.zeros((ds_H, ds_W))

        for k in range(len(color_filaments)):
            filament = color_filaments.iloc[k]
            z_bottom_k = z_cursor.copy()
            z_top_k = z_bottom_k + pixel_counts[:, :, k] * layer_height
            pixel_mask = pixel_counts[:, :, k] > 0

            if pixel_mask.any():
                mesh = generate_topographical_stl(
                    z_bottom_k, z_top_k, pixel_mask, ds_pixel_size,
                    layer_height, preview=True)
                if len(mesh.vertices) > 0:
                    apply_preview_face_colors(
                        mesh, combined_preview, ds_W, ds_H, ds_pixel_size)
                    name = filament['name'].replace(' ', '_')
                    hex_color = '%02x%02x%02x' % tuple(
                        (np.array(filament['rgb']) * 255).astype(int))
                    scene.add_geometry(
                        mesh, geom_name=f"S{k+1:02d}_{name}_C{hex_color}")

            z_cursor = z_top_k

    else:
        # Standard mode: heightmap + TD-proportional z-bands
        enhanced_grayscale = apply_contrast_enhancement(
            ds_grayscale.copy(), ds_alpha, config.contrast_strength)
        enhanced_grayscale = apply_unsharp_mask(enhanced_grayscale, ds_alpha)

        num_colors = len(sorted_filaments)
        filament_tds = np.array([f['transmission_distance']
                                  for _, f in sorted_filaments.iterrows()])
        layer_counts_arr, layer_boundaries, z_boundaries = allocate_layers_td_proportional(
            filament_tds, num_layers, layer_height)

        pixel_height = compute_heightmap(
            enhanced_grayscale, alpha_pixels, model_height, layer_height,
            min_height=float(z_boundaries[1]))

        # Beer-Lambert preview via heightmap + z-bands
        combined_preview = auto_contrast_preview(
            render_standard_preview(pixel_height, z_boundaries, sorted_filaments),
            alpha_pixels)

        # Build one mesh per color band
        for k in range(num_colors):
            filament = sorted_filaments.iloc[k]
            z_lo = z_boundaries[k]
            z_hi = z_boundaries[k + 1]

            band_bottom = np.full((ds_H, ds_W), z_lo)
            band_top = np.clip(pixel_height, z_lo, z_hi)
            pixel_mask = (pixel_height > z_lo + layer_height * 0.5) & alpha_pixels

            if pixel_mask.any():
                mesh = generate_topographical_stl(
                    band_bottom, band_top, pixel_mask, ds_pixel_size,
                    layer_height, preview=True)
                if len(mesh.vertices) > 0:
                    apply_preview_face_colors(
                        mesh, combined_preview, ds_W, ds_H, ds_pixel_size)
                    name = filament['name'].replace(' ', '_')
                    hex_color = '%02x%02x%02x' % tuple(
                        (np.array(filament['rgb']) * 255).astype(int))
                    scene.add_geometry(
                        mesh, geom_name=f"S{k+1:02d}_{name}_C{hex_color}")

    logger.info(f"3D preview scene: {len(scene.geometry)} meshes")
    return scene, sorted_filaments


def _generate_exploded_preview_scene(config, processed_image, selected_filaments):
    """Generate 3D preview for exploded/exploded-multi/cmyk modes.

    Shows each sandwich as a flat slab at its stacking position with a visible
    gap between sandwiches. Color pixels are shown in their filament RGB,
    transparent pixels in near-white. Runs the full Beer-Lambert optimization
    at downsampled resolution.

    Args:
        config: pipeline configuration
        processed_image: processed image data
        selected_filaments: DataFrame of selected filaments

    Returns:
        (trimesh.Scene, color_filaments_df)
    """
    from math import sqrt

    H, W = processed_image.grayscale.shape
    width_mm = config.width_mm

    # Downsample for exploded preview (~250K pixels)
    ds = max(1, round(sqrt(H * W / 250000)))
    ds_alpha = processed_image.alpha_mask[::ds, ::ds]
    ds_H, ds_W = ds_alpha.shape
    ds_pixel_size = width_mm / ds_W
    ds_image_rgb = (processed_image.image_rgb[::ds, ::ds]
                    if processed_image.image_rgb is not None else None)
    ds_grayscale = processed_image.grayscale[::ds, ::ds]

    logger.info(f"3D exploded preview: {H}x{W} -> {ds_H}x{ds_W} (ds={ds})")

    num_colors = len(selected_filaments)
    num_color_filaments = num_colors - 1
    if num_color_filaments <= 0:
        return trimesh.Scene(), selected_filaments

    color_filaments = selected_filaments.iloc[:num_color_filaments]
    transparent_filament = selected_filaments.iloc[-1]

    layer_height = config.layer_height
    total_sandwiches = int(config.model_height / layer_height)

    # Compute caps (same logic as generation methods)
    if config.max_color_sandwiches is not None:
        max_cap = config.max_color_sandwiches
    elif config.use_exploded_cmyk:
        max_cap = 1
    elif config.use_exploded_multi:
        max_cap = 5
    else:
        max_cap = 3
    per_color_caps = [min(max_cap, total_sandwiches)
                      for _ in range(num_color_filaments)]

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

    # Run Beer-Lambert optimization at downsampled resolution
    if config.use_exploded_multi:
        for attempt in range(max_cap):
            current_cap = max_cap - attempt
            per_color_caps = [min(current_cap, total_sandwiches)
                              for _ in range(num_color_filaments)]
            cs = 1
            for cap in per_color_caps:
                cs *= (cap + 1)
            while cs > 500_000:
                mi = per_color_caps.index(max(per_color_caps))
                per_color_caps[mi] -= 1
                cs = 1
                for cap in per_color_caps:
                    cs *= (cap + 1)

            layer_counts = compute_exploded_layer_counts(
                color_filaments, transparent_filament,
                total_sandwiches, per_color_caps,
                (ds_H, ds_W), ds_image_rgb, ds_alpha,
                layer_height, config.sandwich_layers, config.use_fill)
            sandwich_groups = _pack_sandwich_groups(layer_counts, ds_alpha)
            if len(sandwich_groups) <= total_sandwiches:
                break
    else:
        layer_counts = compute_exploded_layer_counts(
            color_filaments, transparent_filament,
            total_sandwiches, per_color_caps,
            (ds_H, ds_W), ds_image_rgb, ds_alpha,
            layer_height, config.sandwich_layers, config.use_fill)

    # For standard exploded, build sandwich groups directly
    if not config.use_exploded_multi:
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

    # Visual parameters
    slab_thickness = 0.5  # mm per sandwich slab (exaggerated for visibility)
    step = slab_thickness  # no gap — slider adds it

    # Transparent fill: very faint
    trans_rgb_uint8 = np.array([240, 240, 235], dtype=np.uint8)
    trans_alpha_val = 15  # ~6% opacity

    # Build filament RGB + Beer-Lambert alpha lookup
    MIN_ALPHA = 40  # ~15%
    color_thickness = layer_height * config.sandwich_layers
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
            rects = greedy_mesh_rects(mask)
            if rects:
                verts, faces = build_box_mesh(
                    rects, z_base, z_base + slab_thickness, ds_pixel_size)
                if len(verts) > 0:
                    mesh = trimesh.Trimesh(vertices=verts, faces=faces,
                                            process=False)
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
            rects = greedy_mesh_rects(trans_mask)
            if rects:
                verts, faces = build_box_mesh(
                    rects, z_base, z_base + slab_thickness, ds_pixel_size)
                if len(verts) > 0:
                    mesh = trimesh.Trimesh(vertices=verts, faces=faces,
                                            process=False)
                    fc = np.zeros((len(mesh.faces), 4), dtype=np.uint8)
                    fc[:, :3] = trans_rgb_uint8
                    fc[:, 3] = trans_alpha_val
                    mesh.visual.face_colors = fc
                    geom_name = f"S{s_idx+1:02d}_transparent"
                    scene.add_geometry(mesh, geom_name=geom_name)

    logger.info(f"3D exploded preview: {len(scene.geometry)} meshes in scene")
    return scene, color_filaments


def _pack_sandwich_groups(layer_counts, alpha_mask):
    """Pack (color, level) pairs into physical sandwiches with up to 3 colors each.

    Uses greedy bin-packing: sorts pairs by mask size (largest first),
    assigns each to the first sandwich where its mask doesn't overlap
    with existing masks and the sandwich has < 3 colors.

    Args:
        layer_counts: ndarray (H, W, num_colors) int32
        alpha_mask: HxW alpha mask

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
    sandwiches = []
    sandwich_combined_masks = []

    for c, k, mask, pixel_count in pairs:
        placed = False
        for s_idx, sandwich in enumerate(sandwiches):
            if len(sandwich) >= 3:
                continue
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


def show_3d_preview(config, processed_image, selected_filaments):
    """Generate and display an interactive 3D preview in the default browser.

    Exports the scene as GLB, embeds it as base64 in a custom HTML viewer
    that uses Three.js (CDN) with GLTFLoader.parse() to avoid data-URL
    size limits that cause blank pages in some browsers.

    Args:
        config: pipeline configuration
        processed_image: processed image data
        selected_filaments: DataFrame of selected filaments
    """
    logger.info("Generating 3D preview scene...")
    scene, preview_filaments = generate_preview_scene(
        config, processed_image, selected_filaments)

    if len(scene.geometry) == 0:
        logger.warning("No geometry generated for 3D preview")
        return

    logger.info("Exporting scene to GLB...")
    glb_data = scene.export(file_type="glb")
    b64 = base64.b64encode(glb_data).decode("utf-8")
    logger.info(f"GLB size: {len(glb_data)/1024/1024:.1f} MB")

    # Build filament info for sidebar
    total_faces = sum(g.faces.shape[0] for g in scene.geometry.values())
    filament_info = []
    for k, (_, f) in enumerate(preview_filaments.iterrows()):
        prefix = f"S{k+1:02d}_"
        mesh_faces = sum(g.faces.shape[0] for name, g in scene.geometry.items()
                         if name.startswith(prefix))
        coverage = (mesh_faces / total_faces * 100) if total_faces > 0 else 0
        filament_info.append({
            'name': f['name'],
            'hex': '#%02x%02x%02x' % tuple(
                (np.array(f['rgb']) * 255).astype(int)),
            'brand': f.get('Brand', ''),
            'td': round(float(f['transmission_distance']), 2),
            'coverage': round(coverage, 1),
            'layer': k + 1,
            'meshPrefix': f"S{k+1:02d}",
        })
    filament_info_json = json.dumps(filament_info)

    is_exploded = (config.use_exploded or config.use_exploded_multi
                   or config.use_exploded_cmyk)
    html = _build_viewer_html(b64, use_transparency=is_exploded, use_slider=True,
                               default_gap=2.0 if is_exploded else 0.0,
                               filament_info_json=filament_info_json)

    with tempfile.NamedTemporaryFile('w', suffix='.html', delete=False) as f:
        f.write(html)
        tmp_path = f.name

    logger.info("Opening 3D preview in browser...")
    webbrowser.open('file://' + tmp_path)


def _build_viewer_html(b64_glb, use_transparency=False, use_slider=False,
                        default_gap=2.0, filament_info_json='[]'):
    """Build a self-contained HTML page with an embedded Three.js GLB viewer.

    Uses CDN-loaded Three.js with GLTFLoader.parse() to decode the base64
    GLB data directly into an ArrayBuffer, bypassing data-URL size limits.

    Args:
        b64_glb: base64-encoded GLB binary string
        use_transparency: if True, enable transparent materials with vertex alpha
        use_slider: if True, show layer gap slider and parse S## mesh names
        default_gap: initial gap value for the slider (0.0 for standard/flat)
        filament_info_json: JSON string of filament metadata for sidebar

    Returns:
        Complete HTML string
    """
    show_controls = use_transparency or use_slider
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HueCLI — 3D Preview</title>
<style>
  body {{ margin: 0; overflow: hidden; background: #2a2a2a; }}
  canvas {{ display: block; }}
  #info {{
    position: absolute; bottom: 70px; left: 50%; transform: translateX(-50%);
    color: #ccc; font: 13px/1.4 system-ui, sans-serif; pointer-events: none;
    text-align: center;
  }}
  #error {{
    position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%);
    color: #f44; font: 16px system-ui; text-align: center; display: none;
  }}
  #controls {{
    position: absolute; bottom: 20px; left: 50%; transform: translateX(-50%);
    display: {'flex' if show_controls else 'none'}; align-items: center; gap: 12px;
    background: rgba(0,0,0,0.6); padding: 8px 16px; border-radius: 8px;
  }}
  #controls label {{ color: #ccc; font: 13px system-ui; white-space: nowrap; }}
  #gap-slider {{ width: 200px; cursor: pointer; }}
  #gap-value {{ color: #fff; font: 13px monospace; min-width: 40px; }}
  #color-mode {{
    display: flex; border-radius: 4px; overflow: hidden;
    border: 1px solid rgba(255,255,255,0.3);
  }}
  #color-mode button {{
    background: rgba(255,255,255,0.08); color: rgba(255,255,255,0.5);
    border: none; padding: 4px 14px; cursor: pointer; font: 13px system-ui;
    white-space: nowrap; transition: background 0.15s, color 0.15s;
  }}
  #color-mode button:hover {{ background: rgba(255,255,255,0.15); }}
  #color-mode button.active {{
    background: rgba(255,255,255,0.3); color: #fff;
    cursor: default;
  }}
  #info-btn {{
    width: 22px; height: 22px; border-radius: 50%; border: 1px solid rgba(255,255,255,0.3);
    background: rgba(255,255,255,0.08); color: rgba(255,255,255,0.5);
    font: italic 13px Georgia, serif; cursor: pointer; display: flex;
    align-items: center; justify-content: center; flex-shrink: 0;
    transition: background 0.15s, color 0.15s;
  }}
  #info-btn:hover {{ background: rgba(255,255,255,0.2); color: #fff; }}
  #info-tooltip {{
    display: none; position: absolute; bottom: 60px; right: 10px;
    background: rgba(0,0,0,0.85); color: #ddd; font: 12px/1.5 system-ui;
    padding: 10px 14px; border-radius: 8px; max-width: 300px;
    border: 1px solid rgba(255,255,255,0.15);
  }}
  #info-tooltip strong {{ color: #fff; }}
  #info-tooltip.visible {{ display: block; }}

  /* Sidebar */
  #sidebar {{
    position: absolute; top: 0; left: 0; bottom: 0; width: 180px;
    background: rgba(0,0,0,0.7); backdrop-filter: blur(8px);
    display: flex; flex-direction: column; padding: 12px 10px;
    transition: transform 0.25s ease; z-index: 10;
    overflow-y: auto; border-right: 1px solid rgba(255,255,255,0.1);
    scrollbar-width: thin; scrollbar-color: rgba(255,255,255,0.2) transparent;
  }}
  #sidebar.collapsed {{ transform: translateX(-100%); }}
  #sidebar-header {{
    text-align: center;
    margin-bottom: 10px; flex-shrink: 0;
  }}
  #sidebar-header span {{ color: #fff; font: bold 13px system-ui, sans-serif; }}
  #sidebar-toggle {{
    position: absolute; top: 12px; left: 180px; z-index: 11;
    width: 24px; height: 28px; border: 1px solid rgba(255,255,255,0.2);
    border-left: none; border-radius: 0 6px 6px 0;
    background: rgba(0,0,0,0.6); color: rgba(255,255,255,0.6);
    cursor: pointer; font: 12px system-ui; display: flex;
    align-items: center; justify-content: center;
    transition: left 0.25s ease, background 0.15s;
  }}
  #sidebar-toggle:hover {{ background: rgba(0,0,0,0.8); color: #fff; }}
  #sidebar.collapsed ~ #sidebar-toggle {{ left: 0; }}
  .swatch-item {{
    text-align: center; margin-bottom: 10px; cursor: pointer;
    padding: 6px 4px; border-radius: 6px; transition: opacity 0.15s;
    position: relative;
  }}
  .swatch-item:hover {{ background: rgba(255,255,255,0.08); }}
  .swatch-item.hidden {{ opacity: 0.25; }}
  .swatch-color {{
    width: 44px; height: 44px; border-radius: 8px; margin: 0 auto;
    border: 2px solid rgba(255,255,255,0.2);
  }}
  .swatch-name {{
    color: #ccc; font: 11px system-ui, sans-serif; margin-top: 4px;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  }}
  .swatch-tooltip {{
    position: fixed; background: rgba(0,0,0,0.92);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 8px; padding: 10px 12px; color: #ddd;
    font: 12px/1.6 system-ui, sans-serif;
    white-space: nowrap; pointer-events: none; z-index: 20;
  }}
  .swatch-tooltip strong {{ color: #fff; }}
</style>
</head>
<body>
<div id="info">Drag to rotate &middot; Scroll to zoom &middot; Right-drag to pan</div>
<div id="error"></div>
<div id="controls">
  <label for="gap-slider">Gap:</label>
  <input type="range" id="gap-slider" min="0" max="5" value="{default_gap:.1f}" step="0.1">
  <span id="gap-value">{default_gap:.1f}mm</span>
  <div id="color-mode">
    <button id="btn-realistic" class="active">Realistic</button>
    <button id="btn-filaments">Filaments</button>
  </div>
  <button id="info-btn">i</button>
</div>
<div id="info-tooltip">
  <strong>Realistic</strong> &mdash; Simulates light passing through stacked filament layers (Beer-Lambert), showing how the print will actually look when backlit.<br><br>
  <strong>Filaments</strong> &mdash; Shows the raw filament colour for each layer, so you can see which filament is assigned where. Layers separate slightly for visibility.
</div>

<div id="sidebar">
  <div id="sidebar-header">
    <span>Filaments</span>
  </div>
  <div id="swatch-list"></div>
</div>
<button id="sidebar-toggle">&#9664;</button>

<script type="importmap">
{{
  "imports": {{
    "three": "https://unpkg.com/three@0.170.0/build/three.module.js",
    "three/addons/": "https://unpkg.com/three@0.170.0/examples/jsm/"
  }}
}}
</script>

<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
import {{ GLTFLoader }} from 'three/addons/loaders/GLTFLoader.js';

const base64 = "{b64_glb}";
const FILAMENT_DATA = {filament_info_json};

// Decode base64 → ArrayBuffer
const raw = atob(base64);
const buf = new Uint8Array(raw.length);
for (let i = 0; i < raw.length; i++) buf[i] = raw.charCodeAt(i);

const scene    = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a1a);

const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.toneMapping = THREE.NoToneMapping;
renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
document.body.appendChild(renderer.domElement);

// Lighting — dim ambient only; mesh uses emissive material for backlit effect
const ambient = new THREE.AmbientLight(0xffffff, 0.15);
scene.add(ambient);

// Camera (will be repositioned after model loads)
const camera = new THREE.PerspectiveCamera(
  45, window.innerWidth / window.innerHeight, 0.01, 10000
);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.12;

// Parse GLB
const loader = new GLTFLoader();
const useTransparency = {'true' if use_transparency else 'false'};
const useSlider = {'true' if use_slider else 'false'};

// Track meshes for slider + color toggle
const layerMeshes = [];  // {{mesh, idx, filamentColor, realisticColors}}
let showFilamentColors = false;
let gapBeforeFilamentMode = null;  // stored gap when entering filament mode

loader.parse(buf.buffer, '', (gltf) => {{
  gltf.scene.traverse((child) => {{
    if (child.isMesh) {{
      const colorAttr = child.geometry.attributes.color;
      const hasAlpha = useTransparency && colorAttr && colorAttr.itemSize === 4;

      let opacity = 1.0;
      if (hasAlpha) {{
        opacity = colorAttr.getW(0);
        if (opacity > 1.0) opacity = opacity / 255.0;
      }}

      // Parse layer index from mesh name for polygon offset
      let layerIdx = 0;
      const idxMatch = child.name.match(/^S(\\d+)/);
      if (idxMatch) layerIdx = parseInt(idxMatch[1], 10);

      child.material = new THREE.MeshBasicMaterial({{
        vertexColors: true,
        transparent: hasAlpha,
        opacity: opacity,
        depthWrite: !hasAlpha,
        side: hasAlpha ? THREE.DoubleSide : THREE.FrontSide,
        polygonOffset: true,
        polygonOffsetFactor: -layerIdx,
        polygonOffsetUnits: -layerIdx,
      }});
      child.renderOrder = layerIdx;

      // Parse mesh name: "S01_Name_Crrggbb" or "S01_Name_L1"
      if (useSlider) {{
        const m = idxMatch;
        if (m) {{
          const entry = {{ mesh: child, idx: parseInt(m[1], 10) - 1,
                          filamentColor: null, realisticColors: null }};

          // Extract filament hex color from name (e.g. "_Cff6f4f")
          const cm = child.name.match(/_C([0-9a-f]{{6}})$/i);
          if (cm && colorAttr) {{
            const hex = cm[1];
            const ri = parseInt(hex.slice(0,2), 16);
            const gi = parseInt(hex.slice(2,4), 16);
            const bi = parseInt(hex.slice(4,6), 16);
            // Store both int (0-255) and float (0-1) for different array types
            entry.filamentColor = [ri, gi, bi];
            entry.filamentColorF = [ri / 255, gi / 255, bi / 255];

            // Save realistic (Beer-Lambert) vertex colors
            const arr = colorAttr.array;
            entry.realisticColors = new Float32Array(arr.length);
            entry.realisticColors.set(arr);
          }}

          layerMeshes.push(entry);
        }}
      }}
    }}
  }});
  scene.add(gltf.scene);

  // Apply initial gap from slider
  updateGap();

  // Auto-fit camera to model bounds (after gap applied)
  const box    = new THREE.Box3().setFromObject(gltf.scene);
  const center = box.getCenter(new THREE.Vector3());
  const size   = box.getSize(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z);
  const dist   = maxDim * 2.0;

  camera.position.set(center.x, center.y - dist * 0.6, center.z + dist * 0.8);
  camera.near = maxDim * 0.001;
  camera.far  = maxDim * 100;
  camera.updateProjectionMatrix();

  controls.target.copy(center);
  controls.update();
}},
(err) => {{
  console.error('GLB parse error:', err);
  const el = document.getElementById('error');
  el.textContent = 'Failed to load 3D model — see console for details.';
  el.style.display = 'block';
}});

// Gap slider: reposition meshes along Z
function updateGap() {{
  const slider = document.getElementById('gap-slider');
  const label = document.getElementById('gap-value');
  if (!slider) return;
  const gap = parseFloat(slider.value);
  label.textContent = gap.toFixed(1) + 'mm';
  for (const entry of layerMeshes) {{
    entry.mesh.position.z = entry.idx * gap;
  }}
}}

// Color toggle: switch between realistic (Beer-Lambert) and solid filament colors
function applyColorMode() {{
  const btnR = document.getElementById('btn-realistic');
  const btnF = document.getElementById('btn-filaments');
  if (btnR && btnF) {{
    btnR.classList.toggle('active', !showFilamentColors);
    btnF.classList.toggle('active', showFilamentColors);
  }}

  for (const entry of layerMeshes) {{
    if (!entry.filamentColor || !entry.realisticColors) continue;
    const colorAttr = entry.mesh.geometry.attributes.color;
    if (!colorAttr) continue;
    const arr = colorAttr.array;
    const itemSize = colorAttr.itemSize;
    const count = colorAttr.count;

    if (showFilamentColors) {{
      // Use int (0-255) for Uint8/Uint16 arrays, float (0-1) for Float arrays
      const useInt = arr instanceof Uint8Array || arr instanceof Uint16Array
                     || arr instanceof Uint8ClampedArray;
      const [r, g, b] = useInt ? entry.filamentColor : entry.filamentColorF;
      for (let i = 0; i < count; i++) {{
        arr[i * itemSize] = r;
        arr[i * itemSize + 1] = g;
        arr[i * itemSize + 2] = b;
      }}
    }} else {{
      arr.set(entry.realisticColors);
    }}
    colorAttr.needsUpdate = true;
  }}
}}

const slider = document.getElementById('gap-slider');
if (slider) {{
  slider.addEventListener('input', () => {{
    updateGap();
    // Re-center camera target on the midpoint of the stack
    if (layerMeshes.length > 0) {{
      const maxIdx = Math.max(...layerMeshes.map(s => s.idx));
      const gap = parseFloat(slider.value);
      const midZ = (maxIdx * gap) / 2;
      controls.target.z = midZ;
    }}
  }});
}}

// Segmented control: Realistic / Filaments
function setColorMode(filaments) {{
  if (showFilamentColors === filaments) return;
  const slider = document.getElementById('gap-slider');

  if (filaments && slider) {{
    // Entering filament mode: save current gap, apply minimum gap so layers are visible
    gapBeforeFilamentMode = parseFloat(slider.value);
    if (gapBeforeFilamentMode < 0.5) {{
      slider.value = '0.5';
      updateGap();
    }}
  }} else if (!filaments && slider && gapBeforeFilamentMode !== null) {{
    // Leaving filament mode: restore previous gap
    slider.value = String(gapBeforeFilamentMode);
    gapBeforeFilamentMode = null;
    updateGap();
  }}

  showFilamentColors = filaments;
  applyColorMode();
}}

const btnR = document.getElementById('btn-realistic');
const btnF = document.getElementById('btn-filaments');
if (btnR) btnR.addEventListener('click', () => setColorMode(false));
if (btnF) btnF.addEventListener('click', () => setColorMode(true));

const infoBtn = document.getElementById('info-btn');
const infoTip = document.getElementById('info-tooltip');
if (infoBtn && infoTip) {{
  infoBtn.addEventListener('click', (e) => {{
    e.stopPropagation();
    infoTip.classList.toggle('visible');
  }});
  document.addEventListener('click', () => infoTip.classList.remove('visible'));
}}

window.addEventListener('resize', () => {{
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}});

// --- Sidebar: build swatches from FILAMENT_DATA ---
const swatchList = document.getElementById('swatch-list');
const tooltip = document.createElement('div');
tooltip.className = 'swatch-tooltip';
tooltip.style.display = 'none';
document.body.appendChild(tooltip);

FILAMENT_DATA.forEach((f, i) => {{
  const item = document.createElement('div');
  item.className = 'swatch-item';
  item.dataset.prefix = f.meshPrefix;
  item.innerHTML = `
    <div class="swatch-color" style="background:${{f.hex}}"></div>
    <div class="swatch-name">${{f.name}}</div>
  `;

  // Hover tooltip
  item.addEventListener('mouseenter', (e) => {{
    const r = item.getBoundingClientRect();
    tooltip.innerHTML = `<strong>${{f.name}}</strong><br>` +
      (f.brand ? `Brand: ${{f.brand}}<br>` : '') +
      `TD: ${{f.td}}mm<br>Hex: ${{f.hex}}<br>Coverage: ${{f.coverage}}%<br>Layer: ${{f.layer}}`;
    tooltip.style.display = 'block';
    tooltip.style.left = (r.right + 8) + 'px';
    tooltip.style.top = r.top + 'px';
  }});
  item.addEventListener('mouseleave', () => {{ tooltip.style.display = 'none'; }});

  // Click to toggle visibility
  item.addEventListener('click', () => {{
    const isHidden = item.classList.toggle('hidden');
    const targetIdx = f.layer - 1;
    for (const entry of layerMeshes) {{
      if (entry.idx === targetIdx) {{
        entry.mesh.visible = !isHidden;
      }}
    }}
  }});

  swatchList.appendChild(item);
}});

// Sidebar toggle
document.getElementById('sidebar-toggle').addEventListener('click', () => {{
  const sidebar = document.getElementById('sidebar');
  const btn = document.getElementById('sidebar-toggle');
  sidebar.classList.toggle('collapsed');
  btn.innerHTML = sidebar.classList.contains('collapsed') ? '&#9654;' : '&#9664;';
}});


(function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}})();
</script>
</body>
</html>'''
