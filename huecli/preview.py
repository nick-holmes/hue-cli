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
    apply_contrast_enhancement,
    apply_unsharp_mask,
    allocate_layers_td_proportional,
    compute_heightmap,
    render_standard_preview,
    render_standard_preview_layers,
    compute_contrast_params,
    apply_contrast_params,
    apply_edge_inset,
)
from .mesh import (
    generate_preview_surface,
    greedy_mesh_rects,
    build_box_mesh,
)

logger = logging.getLogger(__name__)


def generate_preview_scene(config, processed_image, selected_filaments,
                           max_total_faces=1_000_000):
    """Generate a trimesh.Scene with colored meshes for interactive 3D preview.

    Mirrors the heightmap pipeline from generate_all() and returns a Scene
    instead of exporting STLs. Downsamples to fit within max_total_faces for
    browser-renderable mesh size while preserving good detail.

    Args:
        config: needs mode flags (use_flat, use_flat_cap, use_exploded, use_exploded_multi,
                use_exploded_cmyk), layer_height, model_height, num_layers, width_mm,
                sandwich_layers, use_fill, base_layers, max_color_sandwiches,
                dither_mode, contrast_strength
        processed_image: needs image_rgb, grayscale, alpha_mask, width_px, height_px
        selected_filaments: DataFrame
        max_total_faces: target face budget across all meshes (default 1M)

    Returns:
        (trimesh.Scene, sorted_filaments_df)
    """
    is_exploded = (config.use_exploded or config.use_exploded_multi
                   or config.use_exploded_cmyk)
    if is_exploded:
        return _generate_exploded_preview_scene(config, processed_image,
                                                 selected_filaments)

    from math import sqrt, ceil
    from PIL import Image as PILImage
    from trimesh.visual.material import PBRMaterial
    from trimesh.visual import TextureVisuals

    H, W = processed_image.grayscale.shape
    width_mm = config.width_mm
    num_colors = len(selected_filaments)

    # Two resolutions: high-res texture for colors, low-res mesh for geometry.
    # Texture: up to 2M pixels (independent of color count). Textures are
    # lightweight PNGs that don't affect render performance — only mesh
    # face count matters for load time.
    # Geometry: face-budget / (2 * num_colors) pixels.
    max_tex_pixels = 2_000_000
    max_geo_pixels = max_total_faces // (2 * num_colors)
    ds_tex = max(1, ceil(sqrt(H * W / max_tex_pixels)))
    ds_geo = max(1, ceil(sqrt(H * W / max_geo_pixels)))

    # Ensure texture is at least as detailed as geometry
    ds_tex = min(ds_tex, ds_geo)

    sorted_filaments = sort_filaments_by_luminosity(selected_filaments)

    layer_height = config.layer_height
    num_layers = config.num_layers
    model_height = config.model_height

    if config.use_flat or config.use_flat_cap:
        # Flat modes: vertex colors at geometry resolution (discrete colors,
        # texture doesn't help since each pixel is a solid filament color)
        ds = ds_geo
        ds_grayscale = processed_image.grayscale[::ds, ::ds]
        ds_alpha = processed_image.alpha_mask[::ds, ::ds]
        ds_image_rgb = (processed_image.image_rgb[::ds, ::ds]
                        if processed_image.image_rgb is not None else None)
        ds_H, ds_W = ds_grayscale.shape
        ds_pixel_size = width_mm / ds_W

        logger.info(f"3D preview: {H}x{W} -> {ds_H}x{ds_W} (ds={ds}), "
                     f"pixel_size={ds_pixel_size:.3f}mm")

        alpha_pixels = ds_alpha >= 0.5
        scene = trimesh.Scene()

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

        filament_rgbs = np.array([f['rgb'] for _, f in color_filaments.iterrows()])
        combined_preview = np.ones((ds_H, ds_W, 3)) * 0.95

        for k in range(len(color_filaments)):
            present = pixel_counts[:, :, k] > 0
            if present.any():
                combined_preview[present] = filament_rgbs[k]

        combined_preview = auto_contrast_preview(combined_preview, alpha_pixels)

        z_cursor = np.zeros((ds_H, ds_W))

        for k in range(len(color_filaments)):
            filament = color_filaments.iloc[k]
            z_bottom_k = z_cursor.copy()
            z_top_k = z_bottom_k + pixel_counts[:, :, k] * layer_height
            pixel_mask = pixel_counts[:, :, k] > 0

            if pixel_mask.any():
                mesh = generate_preview_surface(
                    z_top_k, pixel_mask, ds_pixel_size, combined_preview)
                if len(mesh.vertices) > 0:
                    name = filament['name'].replace(' ', '_')
                    hex_color = '%02x%02x%02x' % tuple(
                        (np.array(filament['rgb']) * 255).astype(int))
                    scene.add_geometry(
                        mesh, geom_name=f"S{k+1:02d}_{name}_C{hex_color}")

            z_cursor = z_top_k

    else:
        # Standard mode: texture-mapped preview.
        # Compute preview image at texture resolution (high-res colors).
        tex_gray = processed_image.grayscale[::ds_tex, ::ds_tex]
        tex_alpha = processed_image.alpha_mask[::ds_tex, ::ds_tex]
        tex_H, tex_W = tex_gray.shape
        tex_alpha_pixels = tex_alpha >= 0.5

        num_colors = len(sorted_filaments)
        filament_tds = np.array([f['transmission_distance']
                                  for _, f in sorted_filaments.iterrows()])
        layer_counts_arr, layer_boundaries, z_boundaries = allocate_layers_td_proportional(
            filament_tds, num_layers, layer_height)

        tex_enhanced = apply_contrast_enhancement(
            tex_gray.copy(), tex_alpha, config.contrast_strength)
        tex_enhanced = apply_unsharp_mask(tex_enhanced, tex_alpha)
        tex_pixel_height = compute_heightmap(
            tex_enhanced, tex_alpha_pixels, model_height, layer_height,
            min_height=float(z_boundaries[1]))
        if config.nozzle_diameter is not None:
            tex_pixel_size = width_mm / tex_W
            tex_pixel_height = apply_edge_inset(
                tex_pixel_height, z_boundaries, config.nozzle_diameter, tex_pixel_size)

        # Render per-layer cumulative composites
        layer_rgbas = render_standard_preview_layers(
            tex_pixel_height, z_boundaries, sorted_filaments, layer_height)

        # Compute contrast params from topmost layer, apply consistently to all
        top_rgb = layer_rgbas[-1][:, :, :3]
        contrast_params = compute_contrast_params(top_rgb, tex_alpha_pixels)

        materials = []
        for k, rgba in enumerate(layer_rgbas):
            rgb_k = apply_contrast_params(rgba[:, :, :3], tex_alpha_pixels,
                                           contrast_params)
            # Build RGBA texture: present pixels are fully opaque (A=255),
            # absent pixels are transparent (A=0). The RGB already contains the
            # correct cumulative blended color — alpha is only used for clipping.
            z_lo = z_boundaries[k]
            band_present = (tex_pixel_height > z_lo + layer_height * 0.5) & tex_alpha_pixels
            tex_rgba = np.zeros((tex_H, tex_W, 4), dtype=np.uint8)
            tex_rgba[:, :, :3] = (np.clip(rgb_k, 0, 1) * 255).astype(np.uint8)
            tex_rgba[band_present, 3] = 255
            texture_img = PILImage.fromarray(tex_rgba, 'RGBA')
            materials.append(PBRMaterial(
                baseColorTexture=texture_img,
                metallicFactor=0.0, roughnessFactor=1.0))

        logger.info(f"3D preview texture: {tex_W}x{tex_H} (ds_tex={ds_tex})")

        # Build per-layer filament color textures at texture resolution.
        # Each layer gets its own texture showing its filament color where
        # that band has pixels, dark elsewhere.  Used by the JS filament-mode
        # toggle for crisp band rendering at any gap setting.
        # (Completely separate from realistic rendering.)
        filament_rgbs = np.array(
            [f['rgb'] for _, f in sorted_filaments.iterrows()])
        per_layer_fil_maps = []
        for k in range(num_colors):
            z_lo = z_boundaries[k]
            band_mask = (tex_pixel_height > z_lo + layer_height * 0.5) & tex_alpha_pixels
            layer_map = np.zeros((tex_H, tex_W, 4), dtype=np.uint8)
            rgb_bytes = (np.clip(filament_rgbs[k], 0, 1) * 255).astype(np.uint8)
            layer_map[band_mask, :3] = rgb_bytes
            layer_map[band_mask, 3] = 255
            per_layer_fil_maps.append(PILImage.fromarray(layer_map, 'RGBA'))

        # Compute geometry at lower resolution
        geo_gray = processed_image.grayscale[::ds_geo, ::ds_geo]
        geo_alpha = processed_image.alpha_mask[::ds_geo, ::ds_geo]
        geo_H, geo_W = geo_gray.shape
        geo_pixel_size = width_mm / geo_W
        geo_alpha_pixels = geo_alpha >= 0.5

        geo_enhanced = apply_contrast_enhancement(
            geo_gray.copy(), geo_alpha, config.contrast_strength)
        geo_enhanced = apply_unsharp_mask(geo_enhanced, geo_alpha)
        geo_pixel_height = compute_heightmap(
            geo_enhanced, geo_alpha_pixels, model_height, layer_height,
            min_height=float(z_boundaries[1]))
        if config.nozzle_diameter is not None:
            geo_pixel_height = apply_edge_inset(
                geo_pixel_height, z_boundaries, config.nozzle_diameter, geo_pixel_size)

        logger.info(f"3D preview mesh: {geo_W}x{geo_H} (ds_geo={ds_geo}), "
                     f"pixel_size={geo_pixel_size:.3f}mm")

        scene = trimesh.Scene()

        # Build one mesh per color band with UV-mapped texture
        for k in range(num_colors):
            filament = sorted_filaments.iloc[k]
            z_lo = z_boundaries[k]
            z_hi = z_boundaries[k + 1]

            band_top = np.clip(geo_pixel_height, z_lo, z_hi)
            pixel_mask = (geo_pixel_height > z_lo + layer_height * 0.5) & geo_alpha_pixels

            if pixel_mask.any():
                mesh = generate_preview_surface(
                    band_top, pixel_mask, geo_pixel_size,
                    z_bottom=z_lo)  # UV mode with thickness
                if len(mesh.vertices) > 0:
                    mesh.visual = TextureVisuals(
                        uv=mesh.visual.uv, material=materials[k])
                    name = filament['name'].replace(' ', '_')
                    hex_color = '%02x%02x%02x' % tuple(
                        (np.array(filament['rgb']) * 255).astype(int))
                    scene.add_geometry(
                        mesh, geom_name=f"S{k+1:02d}_{name}_C{hex_color}")

        # Attach per-layer filament textures to scene metadata (not exported
        # in GLB, only used by show_3d_preview to build the HTML viewer).
        scene.metadata['filament_layer_maps'] = per_layer_fil_maps
        scene.metadata['band_metadata'] = {
            'bands': [{'z_lo': float(z_boundaries[k]), 'z_hi': float(z_boundaries[k+1])}
                      for k in range(num_colors)],
            'width': float(width_mm),
            'depth': float(geo_H * geo_pixel_size),
        }

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

    # Extract per-layer filament textures (standard mode only) — passed
    # separately from the GLB so filament rendering is fully decoupled.
    filament_layer_textures_json = None
    layer_maps = scene.metadata.get('filament_layer_maps')
    if layer_maps is not None:
        import io
        layer_tex_list = []
        total_bytes = 0
        for k, img in enumerate(layer_maps):
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            total_bytes += len(buf.getvalue())
            layer_tex_list.append(base64.b64encode(buf.getvalue()).decode('utf-8'))
        filament_layer_textures_json = json.dumps(layer_tex_list)
        logger.info(f"Filament layer textures: {len(layer_tex_list)} layers, "
                     f"{total_bytes/1024:.0f} KB total")

    # Extract band metadata (standard mode only) for JS-side box slabs
    band_metadata = scene.metadata.get('band_metadata')
    band_metadata_json = json.dumps(band_metadata) if band_metadata else None

    is_exploded = (config.use_exploded or config.use_exploded_multi
                   or config.use_exploded_cmyk)
    html = _build_viewer_html(b64, use_transparency=is_exploded, use_slider=True,
                               default_gap=2.0 if is_exploded else 0.0,
                               filament_info_json=filament_info_json,
                               filament_layer_textures_json=filament_layer_textures_json,
                               band_metadata_json=band_metadata_json)

    with tempfile.NamedTemporaryFile('w', suffix='.html', delete=False) as f:
        f.write(html)
        tmp_path = f.name

    logger.info("Opening 3D preview in browser...")
    webbrowser.open('file://' + tmp_path)


def _build_viewer_html(b64_glb, use_transparency=False, use_slider=False,
                        default_gap=2.0, filament_info_json='[]',
                        filament_layer_textures_json=None,
                        band_metadata_json=None):
    """Build a self-contained HTML page with an embedded Three.js GLB viewer.

    Uses CDN-loaded Three.js with GLTFLoader.parse() to decode the base64
    GLB data directly into an ArrayBuffer, bypassing data-URL size limits.

    Args:
        b64_glb: base64-encoded GLB binary string
        use_transparency: if True, enable transparent materials with vertex alpha
        use_slider: if True, show layer gap slider and parse S## mesh names
        default_gap: initial gap value for the slider (0.0 for standard/flat)
        filament_info_json: JSON string of filament metadata for sidebar
        filament_layer_textures_json: JSON array of base64-encoded PNGs, one
            per layer (standard mode only). Each texture shows that layer's
            filament color where it has pixels. Fully decoupled from realistic.
        band_metadata_json: JSON string with band z-boundaries and model
            dimensions for JS-side box slab generation (standard mode only).

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
  #bg-picker {{
    display: flex; align-items: center; gap: 4px;
    margin-left: 4px;
  }}
  #bg-picker label {{ color: #888; font: 11px system-ui; margin-right: 2px; }}
  .bg-dot {{
    width: 16px; height: 16px; border-radius: 50%; cursor: pointer;
    border: 2px solid transparent; transition: border-color 0.15s;
    box-sizing: border-box;
  }}
  .bg-dot:hover {{ border-color: rgba(255,255,255,0.5); }}
  .bg-dot.active {{ border-color: #fff; }}
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
  <input type="range" id="gap-slider" min="0" max="10" value="{default_gap:.1f}" step="0.1">
  <span id="gap-value">{default_gap:.1f}mm</span>
  <div id="color-mode">
    <button id="btn-realistic" class="active">Realistic</button>
    <button id="btn-filaments">Filaments</button>
  </div>
  <div id="bg-picker">
    <label>BG:</label>
    <div class="bg-dot active" data-bg="0x1a1a1a" style="background:#1a1a1a;border-color:#fff" title="Dark gray"></div>
    <div class="bg-dot" data-bg="0xffffff" style="background:#fff" title="White"></div>
    <div class="bg-dot" data-bg="0x808080" style="background:#808080" title="Mid gray"></div>
    <div class="bg-dot" data-bg="0xf5e6d3" style="background:#f5e6d3" title="Warm beige"></div>
    <div class="bg-dot" data-bg="0x3d4f5f" style="background:#3d4f5f" title="Slate"></div>
  </div>
  <button id="info-btn">i</button>
</div>
<div id="info-tooltip">
  <strong>Realistic</strong> &mdash; Each layer shows the cumulative front-lit appearance of all layers beneath it. Separate layers with the gap slider to see how colour builds up from bottom to top.<br><br>
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
const BAND_META = {band_metadata_json or 'null'};

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
const layerMeshes = [];  // {{mesh, idx, filamentColor, ...}}
const filamentSlabs = [];  // JS-generated box slabs for filament mode
let showFilamentColors = false;
let gapBeforeFilamentMode = null;  // stored gap when entering filament mode

loader.parse(buf.buffer, '', (gltf) => {{
  gltf.scene.traverse((child) => {{
    if (child.isMesh) {{
      const colorAttr = child.geometry.attributes.color;
      const origMaterial = child.material;
      const hasTexture = origMaterial && origMaterial.map;
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

      if (hasTexture) {{
        // Textured mesh (standard mode): use MeshBasicMaterial with texture
        child.material = new THREE.MeshBasicMaterial({{
          map: origMaterial.map,
          alphaTest: 0.5,
          transparent: true,
          side: THREE.FrontSide,
          polygonOffset: true,
          polygonOffsetFactor: -layerIdx,
          polygonOffsetUnits: -layerIdx,
        }});
      }} else {{
        // Vertex-colored mesh (flat/exploded modes)
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
      }}
      child.renderOrder = layerIdx;

      // Parse mesh name: "S01_Name_Crrggbb" or "S01_Name_L1"
      if (useSlider) {{
        const m = idxMatch;
        if (m) {{
          const entry = {{ mesh: child, idx: parseInt(m[1], 10) - 1,
                          filamentColor: null, filamentColorF: null,
                          realisticColors: null, realisticMaterial: null,
                          isTextured: hasTexture }};

          // Extract filament hex color from name (e.g. "_Cff6f4f")
          const cm = child.name.match(/_C([0-9a-f]{{6}})$/i);
          if (cm) {{
            const hex = cm[1];
            const ri = parseInt(hex.slice(0,2), 16);
            const gi = parseInt(hex.slice(2,4), 16);
            const bi = parseInt(hex.slice(4,6), 16);
            entry.filamentColor = [ri, gi, bi];
            entry.filamentColorF = [ri / 255, gi / 255, bi / 255];
          }}

          if (hasTexture) {{
            // Save textured material for realistic/filament toggle
            entry.realisticMaterial = child.material;
          }} else if (cm && colorAttr) {{
            // Save vertex colors for realistic/filament toggle
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

  // (Filament-mode box slabs are built below when textures load.)
  // Load per-layer filament textures and build JS-side box slabs
  // (standard mode only). Slabs are completely independent geometry from the
  // GLB heightfield meshes — they provide crisp filament-color rendering at
  // texture resolution with visible z-thickness.
  const filLayerTexB64 = {filament_layer_textures_json if filament_layer_textures_json else 'null'};
  if (filLayerTexB64 && BAND_META) {{
    filLayerTexB64.forEach((b64, layerIdx) => {{
      const img = new Image();
      img.onload = () => {{
        const tex = new THREE.Texture(img);
        tex.colorSpace = THREE.SRGBColorSpace;
        tex.needsUpdate = true;

        const band = BAND_META.bands[layerIdx];
        if (!band) return;
        const w = BAND_META.width;
        const d = BAND_META.depth;
        const zThick = band.z_hi - band.z_lo;
        if (zThick <= 0) return;

        // Build box slab with single RGBA texture on all faces.
        // BoxGeometry groups: 0=+x, 1=-x, 2=+y, 3=-y, 4=+z(top), 5=-z(bottom)
        // Side face UVs are remapped to sample the nearest edge row/column
        // of the texture, so alphaTest clips them at the image boundary
        // (no rectangular border on rounded/irregular shapes).
        const geo = new THREE.BoxGeometry(w, d, zThick);
        const uvAttr = geo.attributes.uv;
        const posAttr = geo.attributes.position;

        // Remap side face UVs to sample edge strips from the texture.
        // 4 vertices per face, 24 total. We read vertex positions to
        // determine where along the edge each vertex falls.
        // Group 0 (+X): verts 0-3, sample right column (u=1, v from y)
        for (let vi = 0; vi < 4; vi++) {{
          const y = posAttr.getY(vi);
          uvAttr.setXY(vi, 1.0, 1.0 - (y / d + 0.5));
        }}
        // Group 1 (-X): verts 4-7, sample left column (u=0, v from y)
        for (let vi = 4; vi < 8; vi++) {{
          const y = posAttr.getY(vi);
          uvAttr.setXY(vi, 0.0, 1.0 - (y / d + 0.5));
        }}
        // Group 2 (+Y): verts 8-11, sample top row (v=0, u from x)
        for (let vi = 8; vi < 12; vi++) {{
          const x = posAttr.getX(vi);
          uvAttr.setXY(vi, x / w + 0.5, 0.0);
        }}
        // Group 3 (-Y): verts 12-15, sample bottom row (v=1, u from x)
        for (let vi = 12; vi < 16; vi++) {{
          const x = posAttr.getX(vi);
          uvAttr.setXY(vi, x / w + 0.5, 1.0);
        }}
        // Group 4 (+Z top): verts 16-19, flip V for image convention
        for (let vi = 16; vi < 20; vi++) {{
          uvAttr.setY(vi, 1.0 - uvAttr.getY(vi));
        }}
        // Group 5 (-Z bottom): verts 20-23, default UVs are fine
        uvAttr.needsUpdate = true;

        const texMat = new THREE.MeshBasicMaterial({{
          map: tex, alphaTest: 0.5, transparent: true,
          side: THREE.FrontSide,
        }});
        const slab = new THREE.Mesh(geo, texMat);

        // Position: BoxGeometry is centered, so offset to align with GLB origin
        const zMid = (band.z_lo + band.z_hi) / 2;
        slab.position.set(w / 2, d / 2, zMid);
        slab.visible = false;  // hidden until filament mode activated
        scene.add(slab);

        filamentSlabs.push({{
          slab: slab,
          idx: layerIdx,
          originalZ: zMid,
          band: band,
        }});

        // If already in filament mode, show immediately
        if (showFilamentColors) applyColorMode();
      }};
      img.src = 'data:image/png;base64,' + b64;
    }});
  }}

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
  // Reposition filament slabs with gap offset
  for (const s of filamentSlabs) {{
    s.slab.position.z = s.originalZ + s.idx * gap;
  }}
  // Re-apply filament mode visibility when gap changes
  if (showFilamentColors) applyColorMode();
}}

// Color toggle: switch between realistic (Beer-Lambert) and solid filament colors
function applyColorMode() {{
  const btnR = document.getElementById('btn-realistic');
  const btnF = document.getElementById('btn-filaments');
  if (btnR && btnF) {{
    btnR.classList.toggle('active', !showFilamentColors);
    btnF.classList.toggle('active', showFilamentColors);
  }}

  // Check which layers are hidden via sidebar
  const hiddenLayers = new Set();
  document.querySelectorAll('.swatch-item.hidden').forEach(item => {{
    const prefix = item.dataset.prefix;
    const m = prefix && prefix.match(/^S(\\d+)/);
    if (m) hiddenLayers.add(parseInt(m[1], 10) - 1);
  }});

  for (const entry of layerMeshes) {{
    if (!entry.filamentColorF) continue;
    const layerHidden = hiddenLayers.has(entry.idx);

    if (entry.isTextured) {{
      // Textured mesh (standard mode): toggle between GLB heightfield
      // (realistic) and JS box slabs (filament mode).
      if (showFilamentColors) {{
        // Hide GLB mesh, show corresponding slab
        entry.mesh.visible = false;
        const slab = filamentSlabs.find(s => s.idx === entry.idx);
        if (slab) slab.slab.visible = !layerHidden;
      }} else {{
        // Show GLB mesh, hide slab
        if (entry.realisticMaterial) entry.mesh.material = entry.realisticMaterial;
        entry.mesh.visible = !layerHidden;
        const slab = filamentSlabs.find(s => s.idx === entry.idx);
        if (slab) slab.slab.visible = false;
      }}
    }} else {{
      // Vertex-colored mesh: swap color arrays
      if (!entry.realisticColors) continue;
      const colorAttr = entry.mesh.geometry.attributes.color;
      if (!colorAttr) continue;
      const arr = colorAttr.array;
      const itemSize = colorAttr.itemSize;
      const count = colorAttr.count;

      if (showFilamentColors) {{
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
    // Save current gap when entering filament mode
    gapBeforeFilamentMode = parseFloat(slider.value);
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

// Background color picker
document.querySelectorAll('.bg-dot').forEach(dot => {{
  dot.addEventListener('click', () => {{
    document.querySelectorAll('.bg-dot').forEach(d => d.classList.remove('active'));
    dot.classList.add('active');
    scene.background = new THREE.Color(parseInt(dot.dataset.bg));
  }});
}});

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
        // In filament mode with slabs, hide GLB mesh; toggle slab instead
        if (showFilamentColors && entry.isTextured) {{
          entry.mesh.visible = false;
          const slab = filamentSlabs.find(s => s.idx === targetIdx);
          if (slab) slab.slab.visible = !isHidden;
        }} else {{
          entry.mesh.visible = !isHidden;
        }}
      }}
    }}
    // Also toggle slab if no layerMesh entry matched (shouldn't happen, but safe)
    if (showFilamentColors) {{
      const slab = filamentSlabs.find(s => s.idx === targetIdx);
      if (slab) slab.slab.visible = !isHidden;
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
