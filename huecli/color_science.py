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


def allocate_layers_standard(filament_tds, total_layers, layer_height):
    """Allocate near-uniform z-bands for standard mode.

    Standard mode detail comes from the heightmap, not band thickness.
    Equal z-bands give the most evenly-distributed color transitions.
    Only physical constraint: base needs >= 2 layers for bed adhesion.

    Args:
        filament_tds: 1D array of transmission distances (dark-to-light order)
        total_layers: Total number of layers to allocate
        layer_height: Layer height in mm

    Returns:
        (layer_counts, layer_boundaries, z_boundaries)
    """
    n = len(filament_tds)
    base_count = total_layers // n
    layer_counts = np.full(n, base_count, dtype=int)

    # Distribute remainder evenly from first color onward
    remainder = total_layers - int(layer_counts.sum())
    for i in range(remainder):
        layer_counts[i % n] += 1

    # Ensure base has at least 2 layers for bed adhesion
    if layer_counts[0] < 2 and n > 1:
        deficit = 2 - layer_counts[0]
        layer_counts[0] = 2
        # Steal from the last colors that have more than 1
        for i in range(n - 1, 0, -1):
            if deficit <= 0:
                break
            give = min(deficit, layer_counts[i] - 1)
            if give > 0:
                layer_counts[i] -= give
                deficit -= give

    layer_boundaries = np.concatenate([[0], np.cumsum(layer_counts)]).astype(int)
    z_boundaries = layer_boundaries * layer_height
    return layer_counts, layer_boundaries, z_boundaries


# ---------------------------------------------------------------------------
# Heightmap / contrast
# ---------------------------------------------------------------------------

def compute_heightmap(enhanced_grayscale, alpha_pixels, max_height, layer_height,
                      min_height=None):
    """Compute per-pixel height from contrast-enhanced grayscale.

    Args:
        enhanced_grayscale: 2D array (0-1)
        alpha_pixels: 2D boolean mask
        max_height: Maximum height in mm
        layer_height: Layer height in mm
        min_height: Minimum pixel height in mm. When set to z_boundaries[1],
            ensures the bottom filament band always has full coverage.
            Defaults to layer_height if not provided.

    Returns:
        2D array of per-pixel heights in mm
    """
    if min_height is None:
        min_height = layer_height
    global_brightness_max = np.max(enhanced_grayscale[alpha_pixels]) if np.any(alpha_pixels) else 1.0
    normalized = enhanced_grayscale / max(global_brightness_max, 1e-6)

    pixel_height = min_height + normalized * (max_height - min_height)
    pixel_height = np.clip(pixel_height, min_height, max_height)
    pixel_height = np.where(alpha_pixels, pixel_height, 0)
    return pixel_height


def apply_edge_inset(pixel_height, z_boundaries, nozzle_diameter, pixel_size):
    """Pull back upper layer edges to prevent filament squish from filling fine detail.

    For each layer boundary above the base, pixels at the edge of that layer's
    presence mask have their height lowered to just below the boundary. The inset
    grows cumulatively with each layer: layer k above base erodes by
    k * 0.25 * nozzle_diameter (in mm), converted to pixels.

    Args:
        pixel_height: 2D array of per-pixel heights (mm), modified in place
        z_boundaries: 1D array of color band z-boundaries (mm)
        nozzle_diameter: nozzle diameter in mm
        pixel_size: size of one pixel in mm

    Returns:
        Modified pixel_height array
    """
    from scipy.ndimage import binary_erosion

    inset_mm_per_layer = 0.1 * nozzle_diameter
    min_thickness = z_boundaries[1] - z_boundaries[0] if len(z_boundaries) > 1 else 0.1

    for k in range(1, len(z_boundaries) - 1):
        z_lo = z_boundaries[k]
        inset_mm = k * inset_mm_per_layer
        inset_px = inset_mm / pixel_size

        if inset_px < 0.5:
            continue

        # Build circular structuring element sized to the inset
        radius = int(np.ceil(inset_px))
        size = 2 * radius + 1
        y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        struct = (x * x + y * y) <= inset_px * inset_px

        # Pixels present in this layer
        present = pixel_height > z_lo + min_thickness * 0.5
        if not present.any():
            continue

        # Erode the presence mask
        eroded = binary_erosion(present, structure=struct)

        # Pixels that were present but got eroded — lower their height
        edge_pixels = present & ~eroded
        if edge_pixels.any():
            # Set height to just below this boundary so upper layers don't cover them
            pixel_height[edge_pixels] = np.minimum(
                pixel_height[edge_pixels], z_lo - min_thickness * 0.1)

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
        strength = 7.0
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


def median_smooth(grayscale, alpha_mask, radius=2):
    """Edge-preserving median filter for heightmap grayscale.

    Unifies locally-uniform regions (e.g. a card border that looks solid
    yellow) while preserving sharp edges and text detail. Applied after
    contrast enhancement and unsharp mask, before heightmap computation.

    Args:
        grayscale: 2D array (0-1)
        alpha_mask: 2D array (0-1)
        radius: Filter radius in pixels (window size = 2*radius + 1)

    Returns:
        Median-filtered grayscale (0-1)
    """
    from scipy.ndimage import median_filter

    alpha_pixels = alpha_mask >= 0.5
    size = 2 * radius + 1
    filtered = median_filter(grayscale, size=size)
    filtered = np.where(alpha_pixels, filtered, 0)

    logger.info(f"Median smooth: {size}x{size} window (radius={radius}px)")
    return filtered


def limit_height_gradient(pixel_height, alpha_pixels, max_slope_per_pixel):
    """Limit height gradient between adjacent pixels for printability.

    Each pixel's height cannot exceed any neighbor's height by more than
    max_slope_per_pixel. Iteratively pulls down spikes until convergence.
    Prevents unprintable vertical walls and travel-heavy isolated peaks.

    Args:
        pixel_height: 2D array of per-pixel heights (mm)
        alpha_pixels: 2D boolean mask
        max_slope_per_pixel: maximum height change per pixel step (mm)

    Returns:
        Gradient-limited pixel_height (copy)
    """
    h = pixel_height.copy()

    for iteration in range(50):
        max_from_neighbors = np.full_like(h, np.inf)

        # Right neighbor limits current pixel
        max_from_neighbors[:, :-1] = np.minimum(
            max_from_neighbors[:, :-1], h[:, 1:] + max_slope_per_pixel)
        # Left neighbor
        max_from_neighbors[:, 1:] = np.minimum(
            max_from_neighbors[:, 1:], h[:, :-1] + max_slope_per_pixel)
        # Down neighbor
        max_from_neighbors[:-1, :] = np.minimum(
            max_from_neighbors[:-1, :], h[1:, :] + max_slope_per_pixel)
        # Up neighbor
        max_from_neighbors[1:, :] = np.minimum(
            max_from_neighbors[1:, :], h[:-1, :] + max_slope_per_pixel)

        h_new = np.minimum(h, max_from_neighbors)
        h_new = np.where(alpha_pixels, h_new, 0)

        if np.max(np.abs(h - h_new)) < 1e-6:
            if iteration > 0:
                logger.debug(f"Gradient limiting converged in {iteration + 1} iterations")
            break
        h = h_new

    return h


def score_standard_filament_set(filaments_df, ds_enhanced, ds_alpha_pixels,
                                 ds_lab, model_height, layer_height, num_layers):
    """Score a filament set for standard mode using stack-aware rendering.

    Renders the front-lit LAB-interpolated preview and compares to the
    original image in LAB space. This scores the *cumulative appearance*
    of the filament stack, not individual filaments in isolation.

    Args:
        filaments_df: DataFrame of candidate filaments
        ds_enhanced: 2D downsampled enhanced grayscale (0-1)
        ds_alpha_pixels: 2D boolean mask (downsampled)
        ds_lab: HxWx3 LAB of original downsampled image
        model_height: total model height in mm
        layer_height: layer height in mm
        num_layers: total layer count

    Returns:
        float mean delta-E (Euclidean in LAB)
    """
    sorted_fils = sort_filaments_by_luminosity(filaments_df)
    filament_tds = np.array([f['transmission_distance']
                              for _, f in sorted_fils.iterrows()])
    _, _, z_bounds = allocate_layers_standard(
        filament_tds, num_layers, layer_height)

    heightmap = compute_heightmap(
        ds_enhanced, ds_alpha_pixels, model_height, layer_height,
        min_height=float(z_bounds[1]))

    preview_rgb = render_standard_preview(heightmap, z_bounds, sorted_fils)
    preview_lab = color.rgb2lab(np.clip(preview_rgb, 0, 1))

    diff = np.sqrt(np.sum((preview_lab - ds_lab) ** 2, axis=2))
    return float(np.mean(diff[ds_alpha_pixels]))


def recommend_model_height(filament_tds, color_count, layer_height):
    """Recommend model height for standard mode.

    Targets enough layers per band for smooth color transitions while
    keeping material usage reasonable. Each band gets at least 3 layers,
    with a practical floor of 1.5mm.

    Args:
        filament_tds: array of transmission distances
        color_count: number of filament colors
        layer_height: layer height in mm

    Returns:
        recommended height in mm (rounded to layer_height)
    """
    min_layers_per_band = 3
    ideal_height = color_count * min_layers_per_band * layer_height
    recommended = max(ideal_height, 1.5)
    recommended = min(recommended, 5.0)
    recommended = round(recommended / layer_height) * layer_height
    return recommended


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
    """Render front-lit preview for standard topographical mode.

    Models a front-lit view where pixel height maps to a smooth color gradient
    through the filament stack. Each band's midpoint is the pure filament color;
    between midpoints, colors are linearly interpolated in LAB space for
    perceptually smooth transitions. This avoids posterization with few colors.

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

    # Build a smooth color ramp: each band midpoint gets the pure filament color,
    # and we interpolate between midpoints in LAB for perceptual smoothness.
    # Anchor points: band midpoints + endpoints (clamp to first/last color)
    midpoints = np.array([(z_boundaries[i] + z_boundaries[i + 1]) / 2
                           for i in range(num_colors)])

    # Convert filament colors to LAB for perceptual interpolation
    filament_labs = color.rgb2lab(filament_rgbs.reshape(-1, 1, 3)).reshape(-1, 3)

    # Normalize pixel height to [0, 1] range over the full z span
    z_min = z_boundaries[0]
    z_max = z_boundaries[-1]
    z_range = max(z_max - z_min, 1e-6)

    # Normalize midpoints to same [0, 1] range
    mid_norm = (midpoints - z_min) / z_range
    height_norm = np.clip((pixel_height - z_min) / z_range, 0, 1)

    # Vectorized: find which two midpoints each pixel is between, then lerp in LAB
    h_flat = height_norm.ravel()

    # searchsorted gives the index of the midpoint AFTER each height
    idx_right = np.searchsorted(mid_norm, h_flat).clip(1, num_colors - 1)
    idx_left = idx_right - 1

    # Interpolation parameter: 0 at left midpoint, 1 at right midpoint
    left_z = mid_norm[idx_left]
    right_z = mid_norm[idx_right]
    t = np.clip((h_flat - left_z) / np.maximum(right_z - left_z, 1e-6), 0, 1)

    # Interpolate in LAB
    lab_left = filament_labs[idx_left]    # (N, 3)
    lab_right = filament_labs[idx_right]  # (N, 3)
    result_lab = lab_left * (1 - t[:, np.newaxis]) + lab_right * t[:, np.newaxis]

    return np.clip(color.lab2rgb(result_lab.reshape(H, W, 3)), 0, 1)


def render_standard_preview_layers(pixel_height, z_boundaries, sorted_filaments,
                                    layer_height):
    """Render per-layer cumulative front-lit alpha-over composites for standard mode.

    For each layer k (0=bottom/darkest to N-1=top/lightest), computes the
    cumulative front-lit appearance by alpha-over compositing all layers from
    the bottom up. Each layer's opacity is derived from its thickness and TD
    via Beer-Lambert: opacity = 1 - exp(-thickness / td).

    Args:
        pixel_height: 2D array of per-pixel heights (mm)
        z_boundaries: 1D array of color band boundaries (mm)
        sorted_filaments: DataFrame of filaments (dark to light)
        layer_height: layer height in mm

    Returns:
        List of N RGBA images (HxWx4, 0-1 float sRGB).
        Present pixels: RGB = cumulative composite, A = cumulative opacity.
        Absent pixels: A = 0.
    """
    H, W = pixel_height.shape
    num_colors = len(sorted_filaments)

    filament_rgbs_lin = np.array([srgb_to_linear(np.array(f['rgb']))
                                   for _, f in sorted_filaments.iterrows()])
    filament_tds = np.array([max(f['transmission_distance'], 0.1)
                              for _, f in sorted_filaments.iterrows()])

    # Accumulate composite in linear light: RGB premultiplied, alpha
    accum_rgb = np.zeros((H, W, 3))  # premultiplied linear RGB
    accum_alpha = np.zeros((H, W))

    layer_images = []
    for k in range(num_colors):
        z_lo = z_boundaries[k]
        z_hi = z_boundaries[k + 1]

        # Which pixels are present in this band
        present = (pixel_height > z_lo + layer_height * 0.5)

        # Thickness of this layer per pixel (clipped to band range)
        thickness = np.clip(pixel_height, z_lo, z_hi) - z_lo
        thickness[~present] = 0.0

        # Front-lit opacity: multiply thickness by 4x to account for
        # double-pass (light in + out) and pigment scattering in real FDM layers
        opacity = np.where(present, 1.0 - np.exp(-4.0 * thickness / filament_tds[k]), 0.0)

        # Alpha-over composite (premultiplied): over existing accumulation
        layer_rgb_lin = filament_rgbs_lin[k]  # (3,)
        src_rgb = opacity[:, :, np.newaxis] * layer_rgb_lin  # premultiplied
        src_a = opacity

        # out = src + dst * (1 - src_a)
        accum_rgb = src_rgb + accum_rgb * (1.0 - src_a[:, :, np.newaxis])
        accum_alpha = src_a + accum_alpha * (1.0 - src_a)

        # Convert accumulated premultiplied linear to sRGB for this layer's image
        safe_alpha = np.maximum(accum_alpha, 1e-6)
        straight_rgb = accum_rgb / safe_alpha[:, :, np.newaxis]
        straight_rgb = np.clip(straight_rgb, 0, 1)
        srgb = linear_to_srgb(straight_rgb)

        rgba = np.zeros((H, W, 4))
        rgba[:, :, :3] = srgb
        rgba[:, :, 3] = accum_alpha
        # Zero out absent pixels (no contribution from any layer yet)
        no_coverage = accum_alpha < 1e-6
        rgba[no_coverage] = 0.0

        layer_images.append(rgba)

    return layer_images


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
