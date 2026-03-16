"""Pure color-science functions: Beer-Lambert, sRGB/linear, contrast, dithering.

All functions are stateless — no class, no self references. Parameters that
were previously read from STLGenerator attributes are now explicit arguments.
"""

import numpy as np
import logging
from skimage import color

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# sRGB ↔ linear (IEC 61966-2-1)
# ---------------------------------------------------------------------------

def srgb_to_linear(srgb):
    """IEC 61966-2-1 sRGB to linear light."""
    return np.where(srgb <= 0.04045, srgb / 12.92,
                    ((srgb + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(linear):
    """IEC 61966-2-1 linear light to sRGB."""
    linear = np.clip(linear, 0, 1)
    return np.where(linear <= 0.0031308, linear * 12.92,
                    1.055 * linear ** (1.0 / 2.4) - 0.055)


# ---------------------------------------------------------------------------
# Beer-Lambert simulation
# ---------------------------------------------------------------------------

def compute_effective_color(filament_rgb, filament_td, thickness):
    """Return the Beer-Lambert rendered LAB colour at a given thickness.

    Args:
        filament_rgb: (3,) sRGB values (0-1)
        filament_td: float, transmission distance (mm)
        thickness: float, material thickness (mm)

    Returns:
        (3,) LAB colour array
    """
    rgbs = np.array([filament_rgb])
    tds = np.array([filament_td])
    combos = np.array([[thickness]])
    result_rgb = vectorized_beer_lambert(rgbs, tds, combos)
    return color.rgb2lab(result_rgb.reshape(1, 1, 3))[0][0]


def compute_achievable_gamut_sample(filament_rgb, filament_td, min_thickness,
                                     max_thickness, num_samples=5):
    """Return LAB colours a filament can produce across its thickness range.

    Args:
        filament_rgb: (3,) sRGB values (0-1)
        filament_td: float, transmission distance (mm)
        min_thickness: float, minimum thickness (mm)
        max_thickness: float, maximum thickness (mm)
        num_samples: int, number of sample points

    Returns:
        (num_samples, 3) LAB colour array
    """
    thicknesses = np.linspace(min_thickness, max_thickness, num_samples)
    rgbs = np.array([filament_rgb])
    tds = np.array([filament_td])
    combos = thicknesses.reshape(-1, 1)
    result_rgb = vectorized_beer_lambert(rgbs, tds, combos)
    return color.rgb2lab(result_rgb.reshape(-1, 1, 3)).reshape(-1, 3)


def compute_achievable_gamut(filament_rgbs, filament_tds, layer_height, num_layers):
    """Build achievable LAB point cloud from all filament combos.

    Args:
        filament_rgbs: (N, 3) sRGB values (0-1)
        filament_tds: (N,) transmission distances (mm)
        layer_height: float, layer height (mm)
        num_layers: int, total layer budget

    Returns:
        (M, 3) LAB point cloud of achievable colours
    """
    from itertools import product

    N = len(filament_tds)
    caps = [min(num_layers, max(3, int(td * 2))) for td in filament_tds]
    from math import prod as math_prod
    while math_prod(c + 1 for c in caps) > 100000:
        max_idx = caps.index(max(caps))
        caps[max_idx] -= 1

    ranges = [range(0, c + 1) for c in caps]
    all_combos = np.array(list(product(*ranges)))
    valid = all_combos.sum(axis=1) <= num_layers
    combos = all_combos[valid]

    thicknesses = combos * layer_height
    combo_rgb = vectorized_beer_lambert(filament_rgbs, filament_tds, thicknesses)
    return color.rgb2lab(combo_rgb.reshape(-1, 1, 3)).reshape(-1, 3)


def vectorized_beer_lambert(filament_rgbs, filament_tds, thickness_combos):
    """Compute Beer-Lambert transmitted colors for all thickness combinations.

    Uses corrected transmissive filter model in linear light space.

    Args:
        filament_rgbs: (num_colors, 3) sRGB values (0-1)
        filament_tds: (num_colors,) transmission distances (mm)
        thickness_combos: (n_combos, num_colors) thicknesses (mm)

    Returns:
        (n_combos, 3) resulting sRGB colors (0-1)
    """
    n_combos = len(thickness_combos)
    light = np.ones((n_combos, 3))

    filament_rgbs_lin = srgb_to_linear(np.array(filament_rgbs))

    for k in range(len(filament_tds)):
        thickness = thickness_combos[:, k]
        td = max(filament_tds[k], 0.1)
        rgb_lin = filament_rgbs_lin[k]

        transmission = np.exp(-thickness / td)[:, None]
        light = light * transmission + rgb_lin[None, :] * (1.0 - transmission)
        light = np.clip(light, 0, 1)

    return linear_to_srgb(light)


# ---------------------------------------------------------------------------
# Filament sorting / layer allocation
# ---------------------------------------------------------------------------

def sort_filaments_by_luminosity(selected_filaments):
    """Sort filaments by LAB luminosity (dark to light).

    Returns:
        DataFrame of sorted filaments (reset index)
    """
    sorted_filaments = selected_filaments.copy()
    sorted_filaments['luminosity'] = sorted_filaments['lab'].apply(
        lambda x: float(np.asarray(x).flat[0]))
    sorted_filaments = sorted_filaments.sort_values('luminosity').reset_index(drop=True)
    return sorted_filaments


def allocate_layers_td_proportional(filament_tds, total_layers, layer_height):
    """Allocate layers proportionally to TD using log-dampened weighting.

    Args:
        filament_tds: 1D array of transmission distances
        total_layers: Total number of layers to allocate
        layer_height: Layer height in mm

    Returns:
        (layer_counts, layer_boundaries, z_boundaries)
    """
    num_colors = len(filament_tds)
    td_weights = np.log1p(np.maximum(filament_tds, 1.0))
    td_weights = td_weights / td_weights.sum()

    min_layers = 1
    available_layers = total_layers - min_layers * num_colors
    if available_layers < 0:
        available_layers = 0
        min_layers = max(1, total_layers // num_colors)

    layer_counts = np.round(td_weights * available_layers).astype(int) + min_layers
    layer_counts[-1] = total_layers - layer_counts[:-1].sum()

    layer_boundaries = np.concatenate([[0], np.cumsum(layer_counts)]).astype(int)
    z_boundaries = layer_boundaries * layer_height

    return layer_counts, layer_boundaries, z_boundaries


# ---------------------------------------------------------------------------
# Heightmap / contrast
# ---------------------------------------------------------------------------

def compute_heightmap(enhanced_grayscale, alpha_pixels, max_height, layer_height):
    """Compute per-pixel height from contrast-enhanced grayscale.

    Args:
        enhanced_grayscale: 2D array (0-1)
        alpha_pixels: 2D boolean mask
        max_height: Maximum height in mm
        layer_height: Layer height in mm

    Returns:
        2D array of per-pixel heights in mm
    """
    global_brightness_max = np.max(enhanced_grayscale[alpha_pixels]) if np.any(alpha_pixels) else 1.0
    min_height = 2 * layer_height
    normalized = enhanced_grayscale / max(global_brightness_max, 1e-6)

    pixel_height = min_height + normalized * (max_height - min_height)
    pixel_height = np.clip(pixel_height, min_height, max_height)
    pixel_height = np.where(alpha_pixels, pixel_height, 0)
    return pixel_height


def apply_contrast_enhancement(brightness, alpha_mask, contrast_strength):
    """Apply adaptive S-curve contrast boost.

    Args:
        brightness: 2D array (0-1)
        alpha_mask: 2D array (0-1)
        contrast_strength: S-curve strength (1.0=none, 2.0=moderate)

    Returns:
        Enhanced brightness values (0-1)
    """
    if contrast_strength == 1.0:
        return brightness

    mean_brightness = np.mean(brightness[alpha_mask >= 0.5])

    if mean_brightness > 0.65:
        gamma = 1.5
        enhanced = np.power(brightness, gamma)
        center = np.mean(enhanced)
        strength = 4.0
        enhanced = 1.0 / (1.0 + np.exp(-strength * (enhanced - center)))
        enhanced = (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min())
        logger.info(f"Contrast: bright image (mean={mean_brightness:.3f}), gamma={gamma:.2f} + S-curve")
    elif mean_brightness < 0.35:
        gamma = 0.7
        enhanced = np.power(brightness, gamma)
        logger.info(f"Contrast: dark image (mean={mean_brightness:.3f}), gamma={gamma:.2f}")
    else:
        center = 0.5
        enhanced = 1.0 / (1.0 + np.exp(-contrast_strength * (brightness - center)))
        enhanced = (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min())
        logger.info(f"Contrast: S-curve (mean={mean_brightness:.3f}, strength={contrast_strength:.1f})")

    return enhanced


def apply_unsharp_mask(grayscale, alpha_mask, strength=1.5, radius=1.5):
    """Apply unsharp mask with spike suppression.

    Args:
        grayscale: 2D array (0-1)
        alpha_mask: 2D array (0-1)
        strength: Sharpening strength
        radius: Gaussian blur radius in pixels

    Returns:
        Sharpened grayscale (0-1)
    """
    from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter

    blurred = gaussian_filter(grayscale, sigma=radius)
    sharpened = grayscale + strength * (grayscale - blurred)
    sharpened = np.clip(sharpened, 0, 1)

    local_max = maximum_filter(sharpened, size=3)
    local_min = minimum_filter(sharpened, size=3)

    orig_local_max = maximum_filter(grayscale, size=3)
    orig_local_min = minimum_filter(grayscale, size=3)

    tolerance = 0.15
    upper_bound = orig_local_max + tolerance
    lower_bound = orig_local_min - tolerance
    sharpened = np.clip(sharpened, lower_bound, upper_bound)
    sharpened = np.clip(sharpened, 0, 1)

    alpha_pixels = alpha_mask >= 0.5
    sharpened = np.where(alpha_pixels, sharpened, grayscale)

    spike_count = int(np.sum((sharpened != grayscale) & alpha_pixels))
    logger.info(f"Unsharp mask: strength={strength:.1f}, radius={radius:.1f}px, "
                 f"{spike_count:,} pixels sharpened")
    return sharpened


# ---------------------------------------------------------------------------
# Flat-mode Beer-Lambert optimization
# ---------------------------------------------------------------------------

def compute_flat_layer_counts(sorted_filaments, total_layers, image_rgb,
                               alpha_pixels, layer_height, dither_mode='none',
                               image_shape=None):
    """Compute per-pixel integer layer counts via Beer-Lambert optimization.

    Args:
        sorted_filaments: DataFrame of color filaments (sorted dark to light)
        total_layers: int, total layer budget per pixel
        image_rgb: HxWx3 RGB image (0-1 range)
        alpha_pixels: 2D boolean mask
        layer_height: Layer height in mm
        dither_mode: 'none', 'floyd-steinberg', or 'ordered'
        image_shape: (H, W) shape tuple for dithering

    Returns:
        (H, W, N) int32 array of per-pixel layer counts
    """
    from itertools import product
    from scipy.spatial import cKDTree

    N = len(sorted_filaments)
    H, W = alpha_pixels.shape

    filament_rgbs = np.array([f['rgb'] for _, f in sorted_filaments.iterrows()])
    filament_tds = np.array([f['transmission_distance'] for _, f in sorted_filaments.iterrows()])

    per_color_caps = []
    for td in filament_tds:
        if td < 0.5:
            cap = 3
        elif td < 2.0:
            cap = 4
        else:
            cap = min(total_layers, max(6, int(td * 2)))
        per_color_caps.append(min(cap, total_layers))

    from math import prod as math_prod
    max_raw_combos = 2_000_000
    while math_prod(c + 1 for c in per_color_caps) > max_raw_combos:
        max_idx = per_color_caps.index(max(per_color_caps))
        per_color_caps[max_idx] -= 1

    ranges = [range(0, cap + 1) for cap in per_color_caps]
    all_combos = np.array(list(product(*ranges)))

    combo_totals = all_combos.sum(axis=1)
    valid = combo_totals <= total_layers
    combos = all_combos[valid]

    while len(combos) > 500000:
        max_idx = per_color_caps.index(max(per_color_caps))
        per_color_caps[max_idx] -= 1
        ranges = [range(0, cap + 1) for cap in per_color_caps]
        all_combos = np.array(list(product(*ranges)))
        combo_totals = all_combos.sum(axis=1)
        combos = all_combos[combo_totals <= total_layers]

    n_combos = len(combos)

    thicknesses = combos * layer_height
    combo_colors_rgb = vectorized_beer_lambert(filament_rgbs, filament_tds, thicknesses)
    combo_colors_lab = color.rgb2lab(combo_colors_rgb.reshape(-1, 1, 3)).reshape(-1, 3)

    caps_str = ', '.join(f"{sorted_filaments.iloc[k]['name']}={per_color_caps[k]}"
                          for k in range(N))
    logger.info(f"Flat mode: {N} colors, budget={total_layers}, caps=[{caps_str}] "
                 f"-> {n_combos} combos")

    tree = cKDTree(combo_colors_lab)
    target_lab_flat = color.rgb2lab(image_rgb).reshape(-1, 3)

    if dither_mode == 'ordered':
        apply_ordered_dither(target_lab_flat, alpha_pixels.ravel(), H, W)

    if dither_mode == 'floyd-steinberg':
        target_lab_2d = color.rgb2lab(image_rgb)
        pixel_counts = _compute_flat_layer_counts_dithered(
            target_lab_2d, alpha_pixels, combos, combo_colors_lab, tree)

        achieved_lab = combo_colors_lab[tree.query(target_lab_flat, k=1)[1]]
        valid_mask = alpha_pixels.ravel()
        valid_distances = np.sqrt(np.sum((target_lab_flat[valid_mask] - achieved_lab[valid_mask]) ** 2, axis=1))
        logger.info(f"  Flat Beer-Lambert matching (dithered): mean deltaE={np.mean(valid_distances):.2f}")
    else:
        distances, indices = tree.query(target_lab_flat, k=1)

        valid_mask = alpha_pixels.ravel()
        valid_distances = distances[valid_mask]
        dither_label = " (ordered dither)" if dither_mode == 'ordered' else ""
        logger.info(f"  Flat Beer-Lambert matching{dither_label}: mean deltaE={np.mean(valid_distances):.2f}, "
                     f"median={np.median(valid_distances):.2f}, "
                     f"95th={np.percentile(valid_distances, 95):.2f}")

        pixel_counts = combos[indices].reshape(H, W, N).astype(np.int32)
        pixel_counts[~alpha_pixels] = 0

    for k in range(N):
        name = sorted_filaments.iloc[k]['name']
        active = pixel_counts[:, :, k][alpha_pixels]
        max_k = int(pixel_counts[:, :, k].max())
        mean_k = float(active.mean()) if len(active) > 0 else 0
        pixel_count = int((pixel_counts[:, :, k] > 0).sum())
        logger.info(f"  {name}: max {max_k} layers, mean {mean_k:.1f}, "
                     f"{pixel_count} pixels ({pixel_count / max(alpha_pixels.sum(), 1) * 100:.1f}%)")

    return pixel_counts


def compute_exploded_layer_counts(color_filaments, transparent_filament,
                                   max_layers_per_pixel, per_color_caps,
                                   image_grayscale_shape, image_rgb, alpha_mask,
                                   layer_height, sandwich_layers, use_fill,
                                   dither_mode='none'):
    """Compute optimal integer layer counts per pixel per color for exploded mode.

    Args:
        color_filaments: DataFrame of color filaments (excludes transparent)
        transparent_filament: Series for the transparent filament
        max_layers_per_pixel: int, total layer budget
        per_color_caps: list of int, max sandwiches per color
        image_grayscale_shape: (H, W) shape tuple
        image_rgb: HxWx3 RGB image or None
        alpha_mask: HxW alpha mask
        layer_height: float
        sandwich_layers: int
        use_fill: bool
        dither_mode: str

    Returns:
        layer_counts: ndarray (H, W, num_colors) int32
    """
    from itertools import product
    from scipy.spatial import cKDTree

    num_colors = len(color_filaments)
    H, W = image_grayscale_shape

    filament_rgbs = np.array([f['rgb'] for _, f in color_filaments.iterrows()])
    filament_tds = np.array([f['transmission_distance'] for _, f in color_filaments.iterrows()])
    trans_rgb = np.array(transparent_filament['rgb'])
    trans_td = transparent_filament['transmission_distance']

    all_rgbs = np.vstack([filament_rgbs, trans_rgb.reshape(1, 3)])
    all_tds = np.append(filament_tds, trans_td)

    caps_str = ', '.join(f"{color_filaments.iloc[i]['name']}={per_color_caps[i]}"
                         for i in range(num_colors))
    logger.info(f"Exploded optimization: {num_colors} colors, {max_layers_per_pixel} total sandwiches")
    logger.info(f"  Allocation: {caps_str}")

    ranges = [range(0, cap + 1) for cap in per_color_caps]
    all_combos = np.array(list(product(*ranges)))

    total_layers = all_combos.sum(axis=1)
    valid = total_layers <= max_layers_per_pixel
    combos = all_combos[valid]

    logger.info(f"  {len(combos)} valid combinations (from {len(all_combos)} total)")

    color_thickness = layer_height * sandwich_layers
    trans_layers = max_layers_per_pixel - combos.sum(axis=1)
    color_thicknesses = combos * color_thickness
    trans_per_sandwich = color_thickness if use_fill else layer_height
    trans_thicknesses = (trans_layers * trans_per_sandwich).reshape(-1, 1)
    thickness_combos = np.column_stack([color_thicknesses, trans_thicknesses])

    combo_colors_rgb = vectorized_beer_lambert(all_rgbs, all_tds, thickness_combos)
    combo_colors_lab = color.rgb2lab(combo_colors_rgb.reshape(-1, 1, 3)).reshape(-1, 3)

    tree = cKDTree(combo_colors_lab)

    if image_rgb is not None:
        target_lab_flat = color.rgb2lab(image_rgb).reshape(-1, 3)
    else:
        gray_rgb = np.stack([np.zeros((H, W))] * 3, axis=-1)  # placeholder
        target_lab_flat = color.rgb2lab(gray_rgb).reshape(-1, 3)

    alpha_pixels = alpha_mask >= 0.5

    if dither_mode == 'ordered':
        apply_ordered_dither(target_lab_flat, alpha_pixels.ravel(), H, W)

    if dither_mode == 'floyd-steinberg':
        if image_rgb is not None:
            target_lab_2d = color.rgb2lab(image_rgb)
        else:
            gray_rgb = np.stack([np.zeros((H, W))] * 3, axis=-1)
            target_lab_2d = color.rgb2lab(gray_rgb)

        layer_counts = _compute_exploded_layer_counts_dithered(
            target_lab_2d, alpha_pixels, combos, combo_colors_lab, tree, num_colors)

        valid_mask = alpha_pixels.ravel()
        logger.info(f"  Beer-Lambert matching (Floyd-Steinberg dithered)")
    else:
        distances, indices = tree.query(target_lab_flat, k=1)

        valid_mask = alpha_pixels.ravel()
        valid_distances = distances[valid_mask]
        dither_label = " (ordered dither)" if dither_mode == 'ordered' else ""
        logger.info(f"  Beer-Lambert matching{dither_label}: mean deltaE={np.mean(valid_distances):.2f}, "
                     f"median={np.median(valid_distances):.2f}, "
                     f"95th={np.percentile(valid_distances, 95):.2f}")

        pixel_combos = combos[indices]
        layer_counts = pixel_combos.reshape(H, W, num_colors).astype(np.int32)

        alpha_mask_3d = alpha_pixels[:, :, np.newaxis]
        layer_counts = np.where(alpha_mask_3d, layer_counts, 0)

    total_sandwiches = 0
    for c in range(num_colors):
        name = color_filaments.iloc[c]['name']
        active = layer_counts[:, :, c][alpha_mask >= 0.5]
        max_k = int(layer_counts[:, :, c].max())
        mean_k = float(active.mean()) if len(active) > 0 else 0
        total_sandwiches += max_k
        logger.info(f"  {name}: {max_k} sandwiches (allocated {per_color_caps[c]}), "
                     f"mean {mean_k:.1f} layers/pixel")

    logger.info(f"  Total sandwiches: {total_sandwiches} ({total_sandwiches * 2} STL files)")

    return layer_counts


# ---------------------------------------------------------------------------
# Render flat/standard preview
# ---------------------------------------------------------------------------

def render_flat_preview(pixel_counts, sorted_filaments, layer_height,
                         transparent_filament=None, cap_layers=0):
    """Render Beer-Lambert preview from per-pixel layer counts.

    Args:
        pixel_counts: (H, W, N) int32 array
        sorted_filaments: DataFrame of color filaments
        layer_height: float
        transparent_filament: Series (flat-cap only)
        cap_layers: int (flat-cap only)

    Returns:
        HxWx3 RGB preview image (0-1 range)
    """
    H, W = pixel_counts.shape[:2]
    light = np.ones((H, W, 3))

    for k in range(len(sorted_filaments)):
        filament = sorted_filaments.iloc[k]
        td = max(filament['transmission_distance'], 0.1)
        rgb_lin = srgb_to_linear(np.array(filament['rgb']))
        thickness = pixel_counts[:, :, k] * layer_height

        transmission = np.exp(-thickness / td)[:, :, np.newaxis]
        light = light * transmission + rgb_lin * (1.0 - transmission)

    if transparent_filament is not None and cap_layers > 0:
        trans_td = max(transparent_filament['transmission_distance'], 0.1)
        trans_rgb_lin = srgb_to_linear(np.array(transparent_filament['rgb']))
        cap_thickness = cap_layers * layer_height
        transmission = np.exp(-cap_thickness / trans_td)
        light = light * transmission + trans_rgb_lin * (1.0 - transmission)

    return linear_to_srgb(np.clip(light, 0, 1))


def render_standard_preview(pixel_height, z_boundaries, sorted_filaments):
    """Render Beer-Lambert preview for standard topographical mode.

    Args:
        pixel_height: 2D array of per-pixel heights (mm)
        z_boundaries: 1D array of color band boundaries (mm)
        sorted_filaments: DataFrame of filaments (dark to light)

    Returns:
        HxWx3 RGB preview image (0-1 range)
    """
    H, W = pixel_height.shape
    num_colors = len(sorted_filaments)

    filament_rgbs = np.array([f['rgb'] for _, f in sorted_filaments.iterrows()])
    filament_rgbs_lin = srgb_to_linear(filament_rgbs)
    filament_tds = np.array([f['transmission_distance'] for _, f in sorted_filaments.iterrows()])

    light = np.ones((H, W, 3))

    for i in range(num_colors):
        z_lo = z_boundaries[i]
        z_hi = z_boundaries[i + 1]

        thickness = np.clip(pixel_height, z_lo, z_hi) - z_lo
        td = max(filament_tds[i], 0.1)
        rgb_lin = filament_rgbs_lin[i]

        transmission = np.exp(-thickness / td)[:, :, np.newaxis]
        light = light * transmission + rgb_lin * (1.0 - transmission)

    return linear_to_srgb(np.clip(light, 0, 1))


# ---------------------------------------------------------------------------
# Preview contrast helpers
# ---------------------------------------------------------------------------

def compute_contrast_params(preview_rgb, valid_mask):
    """Compute Lab lightness stretch parameters.

    Returns:
        dict with 'p_lo' and 'rng' keys, or None if no adjustment needed
    """
    if not valid_mask.any():
        return None

    lab = color.rgb2lab(np.clip(preview_rgb, 0, 1))
    valid_L = lab[:, :, 0][valid_mask]
    if len(valid_L) == 0:
        return None
    p_lo = np.percentile(valid_L, 1)
    p_hi = np.percentile(valid_L, 99)
    rng = p_hi - p_lo
    if rng > 70:
        return None
    return {'p_lo': p_lo, 'rng': max(rng, 1.0)}


def apply_contrast_params(preview_rgb, valid_mask, params):
    """Stretch L channel in Lab space, leave a/b unchanged."""
    if params is None:
        return preview_rgb

    lab = color.rgb2lab(np.clip(preview_rgb, 0, 1))
    L = lab[:, :, 0]
    new_L = 5 + (L - params['p_lo']) / params['rng'] * 90

    chroma_boost = min(90.0 / max(params['rng'], 1.0), 5.0)
    lab[:, :, 1] *= chroma_boost
    lab[:, :, 2] *= chroma_boost

    lab[:, :, 0] = np.clip(new_L, 0, 100)
    return np.clip(color.lab2rgb(lab), 0, 1)


def auto_contrast_preview(preview_rgb, valid_mask):
    """Auto-contrast Beer-Lambert preview using Lab lightness stretch."""
    params = compute_contrast_params(preview_rgb, valid_mask)
    return apply_contrast_params(preview_rgb, valid_mask, params)


def apply_preview_face_colors(mesh, preview_rgb, width, height, pixel_size):
    """Map Beer-Lambert preview image onto mesh faces by centroid position.

    Args:
        mesh: trimesh.Trimesh to color
        preview_rgb: HxWx3 RGB preview image (0-1 range)
        width: preview image width (pixels)
        height: preview image height (pixels)
        pixel_size: pixel size in mm
    """
    centroids = mesh.triangles_center
    px = np.clip((centroids[:, 0] / pixel_size).astype(int), 0, width - 1)
    py = np.clip((centroids[:, 1] / pixel_size).astype(int), 0, height - 1)

    rgb_float = np.clip(preview_rgb[py, px], 0, 1)

    face_colors = np.zeros((len(mesh.faces), 4), dtype=np.uint8)
    face_colors[:, :3] = (rgb_float * 255).astype(np.uint8)
    face_colors[:, 3] = 255
    mesh.visual.face_colors = face_colors


# ---------------------------------------------------------------------------
# Dithering
# ---------------------------------------------------------------------------

def apply_ordered_dither(target_lab, alpha_mask_flat, H, W):
    """Apply ordered (Bayer 8x8) dithering to target LAB image (in-place).

    Args:
        target_lab: (H*W, 3) LAB array (modified in-place)
        alpha_mask_flat: (H*W,) boolean mask
        H: image height
        W: image width

    Returns:
        Modified target_lab (same array)
    """
    BAYER_8x8 = np.array([
        [ 0, 32,  8, 40,  2, 34, 10, 42],
        [48, 16, 56, 24, 50, 18, 58, 26],
        [12, 44,  4, 36, 14, 46,  6, 38],
        [60, 28, 52, 20, 62, 30, 54, 22],
        [ 3, 35, 11, 43,  1, 33,  9, 41],
        [51, 19, 59, 27, 49, 17, 57, 25],
        [15, 47,  7, 39, 13, 45,  5, 37],
        [63, 31, 55, 23, 61, 29, 53, 21],
    ], dtype=np.float64) / 64.0 - 0.5

    dither_strength = 5.0

    bayer_tiled = np.tile(BAYER_8x8, (H // 8 + 1, W // 8 + 1))[:H, :W]
    bayer_flat = bayer_tiled.ravel()

    target_lab[alpha_mask_flat, 0] += dither_strength * bayer_flat[alpha_mask_flat]

    return target_lab


def _compute_flat_layer_counts_dithered(target_lab, alpha_pixels, combos,
                                         combo_colors_lab, tree):
    """Floyd-Steinberg error diffusion for flat mode."""
    H, W = alpha_pixels.shape
    N = combos.shape[1]

    error_lab = target_lab.copy()
    pixel_counts = np.zeros((H, W, N), dtype=np.int32)

    for y in range(H):
        for x in range(W):
            if not alpha_pixels[y, x]:
                continue

            current_lab = error_lab[y, x].reshape(1, -1)
            _, idx = tree.query(current_lab, k=1)
            idx = int(idx)

            pixel_counts[y, x] = combos[idx]

            achieved_lab = combo_colors_lab[idx]
            err = error_lab[y, x] - achieved_lab

            if x + 1 < W and alpha_pixels[y, x + 1]:
                error_lab[y, x + 1] += err * (7.0 / 16.0)
            if y + 1 < H:
                if x - 1 >= 0 and alpha_pixels[y + 1, x - 1]:
                    error_lab[y + 1, x - 1] += err * (3.0 / 16.0)
                if alpha_pixels[y + 1, x]:
                    error_lab[y + 1, x] += err * (5.0 / 16.0)
                if x + 1 < W and alpha_pixels[y + 1, x + 1]:
                    error_lab[y + 1, x + 1] += err * (1.0 / 16.0)

    return pixel_counts


def _compute_exploded_layer_counts_dithered(target_lab, alpha_pixels, combos,
                                             combo_colors_lab, tree, num_colors):
    """Floyd-Steinberg error diffusion for exploded mode."""
    H, W = alpha_pixels.shape

    error_lab = target_lab.copy()
    pixel_counts = np.zeros((H, W, num_colors), dtype=np.int32)

    for y in range(H):
        for x in range(W):
            if not alpha_pixels[y, x]:
                continue

            current_lab = error_lab[y, x].reshape(1, -1)
            _, idx = tree.query(current_lab, k=1)
            idx = int(idx)

            pixel_counts[y, x] = combos[idx]

            achieved_lab = combo_colors_lab[idx]
            err = error_lab[y, x] - achieved_lab

            if x + 1 < W and alpha_pixels[y, x + 1]:
                error_lab[y, x + 1] += err * (7.0 / 16.0)
            if y + 1 < H:
                if x - 1 >= 0 and alpha_pixels[y + 1, x - 1]:
                    error_lab[y + 1, x - 1] += err * (3.0 / 16.0)
                if alpha_pixels[y + 1, x]:
                    error_lab[y + 1, x] += err * (5.0 / 16.0)
                if x + 1 < W and alpha_pixels[y + 1, x + 1]:
                    error_lab[y + 1, x + 1] += err * (1.0 / 16.0)

    return pixel_counts
