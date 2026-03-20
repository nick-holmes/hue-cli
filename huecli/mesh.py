"""Pure geometry functions: topographical height-field, greedy-meshed flat layers, box mesh.

All functions are stateless. pixel_size is an explicit parameter instead of self.pixel_size.
"""

import numpy as np
import trimesh
import logging

logger = logging.getLogger(__name__)


def greedy_mesh_rects(mask):
    """Find maximal rectangles covering all True pixels using greedy meshing.

    Args:
        mask: 2D boolean array

    Returns:
        List of (y_start, x_start, y_end, x_end) rectangles (exclusive end)
    """
    if not mask.any():
        return []

    height, width = mask.shape
    consumed = np.zeros_like(mask, dtype=bool)
    rects = []

    for y in range(height):
        row_avail = mask[y] & ~consumed[y]
        if not row_avail.any():
            continue

        padded = np.concatenate([[False], row_avail, [False]])
        diff = np.diff(padded.astype(np.int8))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        for s, e in zip(starts, ends):
            if consumed[y, s]:
                continue

            y_end = y + 1
            while y_end < height:
                if not (mask[y_end, s:e] & ~consumed[y_end, s:e]).all():
                    break
                y_end += 1

            consumed[y:y_end, s:e] = True
            rects.append((y, s, y_end, e))

    return rects


def build_box_mesh(rects, z_bottom_val, z_top_val, pixel_size):
    """Build mesh geometry for a list of rectangles at the same height (vectorized).

    Args:
        rects: List of (y0, x0, y1, x1) rectangles
        z_bottom_val: Z height for bottom face
        z_top_val: Z height for top face
        pixel_size: Size of each pixel in mm

    Returns:
        (vertices_array, faces_array) as numpy arrays
    """
    if not rects:
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3).astype(int)

    n = len(rects)
    rects_arr = np.array(rects)

    px0 = rects_arr[:, 1] * pixel_size
    px1 = rects_arr[:, 3] * pixel_size
    py0 = rects_arr[:, 0] * pixel_size
    py1 = rects_arr[:, 2] * pixel_size
    zb = np.full(n, z_bottom_val)
    zt = np.full(n, z_top_val)

    vertices = np.empty((n * 8, 3))
    vertices[0::8] = np.column_stack([px0, py0, zb])
    vertices[1::8] = np.column_stack([px1, py0, zb])
    vertices[2::8] = np.column_stack([px1, py1, zb])
    vertices[3::8] = np.column_stack([px0, py1, zb])
    vertices[4::8] = np.column_stack([px0, py0, zt])
    vertices[5::8] = np.column_stack([px1, py0, zt])
    vertices[6::8] = np.column_stack([px1, py1, zt])
    vertices[7::8] = np.column_stack([px0, py1, zt])

    base_faces = np.array([
        [0, 2, 1], [0, 3, 2],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [2, 3, 7], [2, 7, 6],
        [0, 4, 7], [0, 7, 3],
        [1, 2, 6], [1, 6, 5],
    ], dtype=np.int64)

    offsets = (np.arange(n) * 8)[:, None, None]
    faces = (base_faces[None, :, :] + offsets).reshape(-1, 3)

    return vertices, faces


def generate_preview_surface(z_top, pixel_mask, pixel_size, preview_rgb=None,
                              z_bottom=None):
    """Generate preview mesh with shared grid-corner vertices.

    Creates the top surface (2 triangles per pixel). When z_bottom is provided,
    also creates a bottom surface and side walls for visible layer thickness.

    Two modes:
    - preview_rgb provided: bakes vertex colors (for flat/exploded modes)
    - preview_rgb=None: sets UV coordinates for texture mapping (standard mode)

    Args:
        z_top: 2D array of top z-heights (mm)
        pixel_mask: 2D boolean array of active pixels
        pixel_size: pixel size in mm
        preview_rgb: HxWx3 RGB preview image (0-1 range), or None for UV mode
        z_bottom: scalar or 2D array of bottom z-heights (mm), or None for
            top-surface-only mode

    Returns:
        trimesh.Trimesh with vertex colors or UV coordinates
    """
    if not pixel_mask.any():
        return trimesh.Trimesh()

    H, W = pixel_mask.shape
    ps = pixel_size
    mask_f = pixel_mask.astype(np.float64)

    # Average z_top at grid corners from adjacent active pixels
    count = np.zeros((H + 1, W + 1))
    zt_sum = np.zeros((H + 1, W + 1))
    for dy in range(2):
        for dx in range(2):
            count[dy:H + dy, dx:W + dx] += mask_f
            zt_sum[dy:H + dy, dx:W + dx] += z_top * mask_f

    corner_active = count > 0
    zt_avg = np.divide(zt_sum, count, where=corner_active,
                        out=np.zeros_like(zt_sum))

    n_active = int(corner_active.sum())
    if n_active == 0:
        return trimesh.Trimesh()

    vertex_idx = np.full((H + 1, W + 1), -1, dtype=np.int64)
    active_cy, active_cx = np.where(corner_active)
    vertex_idx[active_cy, active_cx] = np.arange(n_active)

    # Top surface vertices
    top_verts = np.column_stack([
        active_cx * ps, active_cy * ps,
        zt_avg[active_cy, active_cx]
    ])

    py, px = np.where(pixel_mask)
    v00 = vertex_idx[py, px]
    v01 = vertex_idx[py, px + 1]
    v10 = vertex_idx[py + 1, px]
    v11 = vertex_idx[py + 1, px + 1]

    all_faces = [
        np.column_stack([v00, v01, v10]),
        np.column_stack([v01, v11, v10]),
    ]

    if z_bottom is not None:
        # Bottom vertices at z_bottom (scalar or array)
        if np.ndim(z_bottom) == 0:
            zb_vals = np.full(n_active, float(z_bottom))
        else:
            zb_sum = np.zeros((H + 1, W + 1))
            for dy in range(2):
                for dx in range(2):
                    zb_sum[dy:H + dy, dx:W + dx] += z_bottom * mask_f
            zb_vals = np.divide(zb_sum, count, where=corner_active,
                                 out=np.zeros_like(zb_sum))[active_cy, active_cx]

        bottom_verts = np.column_stack([
            active_cx * ps, active_cy * ps, zb_vals
        ])
        vertices = np.vstack([top_verts, bottom_verts])

        # Bottom faces (reversed winding), offset by n_active
        b00 = v00 + n_active
        b01 = v01 + n_active
        b10 = v10 + n_active
        b11 = v11 + n_active
        all_faces.append(np.column_stack([b00, b10, b01]))
        all_faces.append(np.column_stack([b01, b10, b11]))

        # Side walls at boundary edges
        def get_boundary(dy, dx):
            ny, nx = py + dy, px + dx
            oob = (ny < 0) | (ny >= H) | (nx < 0) | (nx >= W)
            inactive = oob | ~pixel_mask[np.clip(ny, 0, H - 1),
                                          np.clip(nx, 0, W - 1)]
            return inactive

        left = get_boundary(0, -1)
        if left.any():
            tl0 = vertex_idx[py[left], px[left]]
            tl1 = vertex_idx[py[left] + 1, px[left]]
            bl0 = tl0 + n_active
            bl1 = tl1 + n_active
            all_faces.append(np.column_stack([bl0, tl0, bl1]))
            all_faces.append(np.column_stack([bl1, tl0, tl1]))

        right = get_boundary(0, 1)
        if right.any():
            tr0 = vertex_idx[py[right], px[right] + 1]
            tr1 = vertex_idx[py[right] + 1, px[right] + 1]
            br0 = tr0 + n_active
            br1 = tr1 + n_active
            all_faces.append(np.column_stack([br0, br1, tr0]))
            all_faces.append(np.column_stack([br1, tr1, tr0]))

        front = get_boundary(-1, 0)
        if front.any():
            tf0 = vertex_idx[py[front], px[front]]
            tf1 = vertex_idx[py[front], px[front] + 1]
            bf0 = tf0 + n_active
            bf1 = tf1 + n_active
            all_faces.append(np.column_stack([bf0, bf1, tf0]))
            all_faces.append(np.column_stack([bf1, tf1, tf0]))

        back = get_boundary(1, 0)
        if back.any():
            tb0 = vertex_idx[py[back] + 1, px[back]]
            tb1 = vertex_idx[py[back] + 1, px[back] + 1]
            bb0 = tb0 + n_active
            bb1 = tb1 + n_active
            all_faces.append(np.column_stack([bb0, tb0, bb1]))
            all_faces.append(np.column_stack([bb1, tb0, tb1]))
    else:
        vertices = top_verts

    faces = np.vstack(all_faces)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    n_verts = len(vertices)

    if preview_rgb is not None:
        # Vertex colors: average preview_rgb at corners from adjacent active pixels
        rgb_weighted = preview_rgb * mask_f[:, :, np.newaxis]
        color_sum = np.zeros((H + 1, W + 1, 3))
        for dy in range(2):
            for dx in range(2):
                color_sum[dy:H + dy, dx:W + dx] += rgb_weighted

        count_3d = count[:, :, np.newaxis]
        corner_active_3d = corner_active[:, :, np.newaxis]
        color_avg = np.divide(color_sum, count_3d, where=corner_active_3d,
                               out=np.zeros_like(color_sum))

        vertex_rgb = color_avg[active_cy, active_cx]
        vertex_colors = np.zeros((n_verts, 4), dtype=np.uint8)
        vertex_colors[:n_active, :3] = (np.clip(vertex_rgb, 0, 1) * 255).astype(np.uint8)
        vertex_colors[:n_active, 3] = 255
        if z_bottom is not None:
            # Bottom + side wall vertices get same colors as corresponding top
            vertex_colors[n_active:] = vertex_colors[:n_active]
        mesh.visual.vertex_colors = vertex_colors
    else:
        # UV coordinates for texture mapping: map grid corners to [0,1] range
        top_uv = np.column_stack([active_cx / W, 1.0 - active_cy / H])
        if z_bottom is not None:
            # Bottom + side wall vertices share same UV as top
            uv = np.vstack([top_uv, top_uv])
        else:
            uv = top_uv
        mesh.visual = trimesh.visual.TextureVisuals(uv=uv)

    logger.info(f"  Preview surface: {pixel_mask.sum():,} pixels -> "
                 f"{n_verts} vertices, {len(faces):,} faces")

    return mesh


def generate_topographical_stl(z_bottom, z_top, pixel_mask, pixel_size,
                                layer_height, preview=False):
    """Generate topographical STL using shared-vertex grid.

    Args:
        z_bottom: 2D array of bottom z-heights (mm)
        z_top: 2D array of top z-heights (mm)
        pixel_mask: 2D boolean array
        pixel_size: pixel size in mm
        layer_height: layer height in mm
        preview: if True, skip edge-preserving corners

    Returns:
        trimesh.Trimesh object
    """
    if not pixel_mask.any():
        return trimesh.Trimesh()

    H, W = pixel_mask.shape
    ps = pixel_size

    effective_mask = pixel_mask & (z_top > z_bottom + 1e-6)
    if not effective_mask.any():
        return trimesh.Trimesh()

    mask_f = effective_mask.astype(np.float64)

    zt_active = z_top[effective_mask]
    height_range = zt_active.max() - zt_active.min()
    edge_threshold = max(height_range * 0.15, layer_height)

    count = np.zeros((H + 1, W + 1))
    zb_sum = np.zeros((H + 1, W + 1))
    zt_sum = np.zeros((H + 1, W + 1))

    for dy in range(2):
        for dx in range(2):
            count[dy:H + dy, dx:W + dx] += mask_f
            zt_sum[dy:H + dy, dx:W + dx] += z_top * mask_f
            zb_sum[dy:H + dy, dx:W + dx] += z_bottom * mask_f

    corner_active = count > 0
    zt_naive = np.divide(zt_sum, count, where=corner_active, out=np.zeros_like(zt_sum))

    max_dev = np.zeros((H, W))
    for dy in range(2):
        for dx in range(2):
            dev = np.abs(z_top - zt_naive[dy:H + dy, dx:W + dx])
            max_dev = np.maximum(max_dev, dev)

    is_edge_pixel = effective_mask & (max_dev > edge_threshold)
    edge_count = int(is_edge_pixel.sum())

    if edge_count > 0 and not preview:
        smooth_f = effective_mask.astype(np.float64)
        smooth_f[is_edge_pixel] = 0

        count[:] = 0
        zb_sum[:] = 0
        zt_sum[:] = 0

        smooth_zt = z_top * smooth_f
        smooth_zb = z_bottom * smooth_f
        for dy in range(2):
            for dx in range(2):
                count[dy:H + dy, dx:W + dx] += smooth_f
                zt_sum[dy:H + dy, dx:W + dx] += smooth_zt
                zb_sum[dy:H + dy, dx:W + dx] += smooth_zb

        edge_py, edge_px = np.where(is_edge_pixel)
        edge_zt = z_top[edge_py, edge_px]
        edge_zb = z_bottom[edge_py, edge_px]

        for dy in range(2):
            for dx in range(2):
                cy = edge_py + dy
                cx = edge_px + dx
                no_smooth = count[cy, cx] == 0
                if no_smooth.any():
                    count[cy[no_smooth], cx[no_smooth]] += 1
                    zt_sum[cy[no_smooth], cx[no_smooth]] += edge_zt[no_smooth]
                    zb_sum[cy[no_smooth], cx[no_smooth]] += edge_zb[no_smooth]

        corner_active = count > 0
        logger.info(f"  Edge-preserving corners: {edge_count:,} edge pixels "
                     f"({edge_count / effective_mask.sum() * 100:.1f}%) preserved sharp")

    zb_avg = np.divide(zb_sum, count, where=corner_active, out=np.zeros_like(zb_sum))
    zt_avg = np.divide(zt_sum, count, where=corner_active, out=np.zeros_like(zt_sum))

    n_active = int(corner_active.sum())
    vertex_idx = np.full((H + 1, W + 1), -1, dtype=np.int64)
    active_cy, active_cx = np.where(corner_active)
    vertex_idx[active_cy, active_cx] = np.arange(n_active)

    bottom_verts = np.column_stack([active_cx * ps, active_cy * ps, zb_avg[active_cy, active_cx]])
    top_verts = np.column_stack([active_cx * ps, active_cy * ps, zt_avg[active_cy, active_cx]])
    vertices = np.vstack([bottom_verts, top_verts])

    py, px = np.where(effective_mask)

    b00 = vertex_idx[py, px]
    b01 = vertex_idx[py, px + 1]
    b10 = vertex_idx[py + 1, px]
    b11 = vertex_idx[py + 1, px + 1]

    t00 = b00 + n_active
    t01 = b01 + n_active
    t10 = b10 + n_active
    t11 = b11 + n_active

    all_faces = []

    all_faces.append(np.column_stack([t00, t01, t10]))
    all_faces.append(np.column_stack([t01, t11, t10]))

    all_faces.append(np.column_stack([b00, b10, b01]))
    all_faces.append(np.column_stack([b01, b10, b11]))

    def get_boundary(dy, dx):
        ny, nx = py + dy, px + dx
        oob = (ny < 0) | (ny >= H) | (nx < 0) | (nx >= W)
        inactive = oob | ~effective_mask[np.clip(ny, 0, H - 1), np.clip(nx, 0, W - 1)]
        return inactive

    left = get_boundary(0, -1)
    if left.any():
        bl0 = vertex_idx[py[left], px[left]]
        bl1 = vertex_idx[py[left] + 1, px[left]]
        tl0 = bl0 + n_active
        tl1 = bl1 + n_active
        all_faces.append(np.column_stack([bl0, bl1, tl0]))
        all_faces.append(np.column_stack([bl1, tl1, tl0]))

    right = get_boundary(0, 1)
    if right.any():
        br0 = vertex_idx[py[right], px[right] + 1]
        br1 = vertex_idx[py[right] + 1, px[right] + 1]
        tr0 = br0 + n_active
        tr1 = br1 + n_active
        all_faces.append(np.column_stack([br0, tr0, br1]))
        all_faces.append(np.column_stack([br1, tr0, tr1]))

    front = get_boundary(-1, 0)
    if front.any():
        bf0 = vertex_idx[py[front], px[front]]
        bf1 = vertex_idx[py[front], px[front] + 1]
        tf0 = bf0 + n_active
        tf1 = bf1 + n_active
        all_faces.append(np.column_stack([bf0, tf0, bf1]))
        all_faces.append(np.column_stack([bf1, tf0, tf1]))

    back = get_boundary(1, 0)
    if back.any():
        bb0 = vertex_idx[py[back] + 1, px[back]]
        bb1 = vertex_idx[py[back] + 1, px[back] + 1]
        tb0 = bb0 + n_active
        tb1 = bb1 + n_active
        all_faces.append(np.column_stack([bb0, bb1, tb0]))
        all_faces.append(np.column_stack([bb1, tb1, tb0]))

    combined_faces = np.vstack(all_faces)
    mesh = trimesh.Trimesh(vertices=vertices, faces=combined_faces, process=False)

    logger.info(f"  Shared-vertex grid: {effective_mask.sum():,} pixels, "
                 f"{n_active} corners -> {len(mesh.faces):,} faces")

    return mesh


def generate_quantized_stl(z_bottom, z_top, pixel_mask, pixel_size, layer_height):
    """Generate STL with height quantization + incremental slab greedy meshing.

    Args:
        z_bottom: 2D array of bottom z-heights (mm)
        z_top: 2D array of top z-heights (mm)
        pixel_mask: 2D boolean array
        pixel_size: pixel size in mm
        layer_height: layer height in mm

    Returns:
        trimesh.Trimesh object
    """
    if not pixel_mask.any():
        return trimesh.Trimesh()

    lh = layer_height

    q_z_top = np.round(z_top / lh) * lh
    q_z_bottom = np.round(z_bottom / lh) * lh

    effective_mask = pixel_mask & (q_z_top > q_z_bottom + 1e-6)
    if not effective_mask.any():
        return trimesh.Trimesh()

    active_zb = q_z_bottom[effective_mask]
    active_zt = q_z_top[effective_mask]
    all_z_values = np.unique(np.concatenate([active_zb, active_zt]))
    z_layers = np.round(all_z_values / lh).astype(np.int64)
    z_layers = np.unique(z_layers)

    q_zb_layers = np.round(q_z_bottom / lh).astype(np.int64)
    q_zt_layers = np.round(q_z_top / lh).astype(np.int64)

    all_verts = []
    all_faces = []
    vertex_offset = 0
    total_rects = 0

    for s in range(len(z_layers) - 1):
        slab_bottom_layer = z_layers[s]
        slab_top_layer = z_layers[s + 1]

        slab_mask = effective_mask & (q_zb_layers <= slab_bottom_layer) & (q_zt_layers >= slab_top_layer)

        if not slab_mask.any():
            continue

        rects = greedy_mesh_rects(slab_mask)
        if rects:
            zb_val = slab_bottom_layer * lh
            zt_val = slab_top_layer * lh
            verts, faces = build_box_mesh(rects, zb_val, zt_val, pixel_size)
            if len(verts) > 0:
                all_verts.append(verts)
                all_faces.append(faces + vertex_offset)
                vertex_offset += len(verts)
                total_rects += len(rects)

    if not all_verts:
        return trimesh.Trimesh()

    combined_verts = np.vstack(all_verts)
    combined_faces = np.vstack(all_faces)
    mesh = trimesh.Trimesh(vertices=combined_verts, faces=combined_faces, process=False)

    total_pixels = effective_mask.sum()
    num_slabs = len(z_layers) - 1
    logger.info(f"  Quantized mesh: {total_pixels:,} pixels, {num_slabs} slabs, "
                 f"{total_rects} rects -> {len(combined_faces):,} faces "
                 f"(vs ~{total_pixels * 6:,} topographical)")

    return mesh


def generate_flat_layer_stl(pixel_mask, layer_start, layer_end, pixel_size, layer_height):
    """Generate flat STL for given pixel mask at specific layer range.

    Args:
        pixel_mask: 2D boolean array
        layer_start: Starting layer number
        layer_end: Ending layer number
        pixel_size: pixel size in mm
        layer_height: layer height in mm

    Returns:
        trimesh.Trimesh object
    """
    z_bottom = layer_start * layer_height
    z_top = layer_end * layer_height

    rects = greedy_mesh_rects(pixel_mask)
    verts, faces = build_box_mesh(rects, z_bottom, z_top, pixel_size)

    if len(verts) == 0:
        return trimesh.Trimesh()

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    original_pixels = pixel_mask.sum()
    logger.info(f"    Flat layer: {original_pixels:,} pixels -> {len(rects)} rects, {len(faces)} faces")

    return mesh


def generate_color_band_stls(sorted_filaments, pixel_height, z_boundaries,
                              layer_boundaries, alpha_pixels, output_base_path,
                              pixel_size, layer_height, collect_meshes=False):
    """Generate one quantized STL per color band.

    Args:
        sorted_filaments: DataFrame of filaments (sorted)
        pixel_height: 2D array of per-pixel heights (mm)
        z_boundaries: 1D float array of z-heights at band boundaries
        layer_boundaries: 1D int array of layer indices at boundaries
        alpha_pixels: 2D boolean mask
        output_base_path: Path for output files
        pixel_size: pixel size in mm
        layer_height: layer height in mm
        collect_meshes: If True, return list of (mesh, path, name)

    Returns:
        List of (path, name, layer_start, layer_end) tuples, or
        list of (mesh, path, name) tuples if collect_meshes=True
    """
    from scipy import ndimage

    generated_files = []
    mesh_outputs = []
    min_thickness = layer_height * 0.5

    for i in range(len(sorted_filaments)):
        filament = sorted_filaments.iloc[i]
        color_name = filament['name'].replace(' ', '_').replace('/', '_')

        layer_start = int(layer_boundaries[i])
        layer_end = int(layer_boundaries[i + 1])

        z_bottom_flat = float(z_boundaries[i])
        z_top_boundary = float(z_boundaries[i + 1])

        pixel_mask = (pixel_height > z_bottom_flat + min_thickness) & alpha_pixels
        pixel_count = int(np.sum(pixel_mask))

        labeled, num_regions = ndimage.label(pixel_mask)
        # Minimum printable region: ~0.8mm² (roughly 4x4 nozzle widths for typical 0.2mm nozzle)
        min_region_size = max(8, int(0.8 / (pixel_size * pixel_size)))

        if num_regions > 1 and i > 0:
            region_sizes = np.bincount(labeled.ravel())
            small_regions = region_sizes < min_region_size
            small_regions[0] = False
            pixel_mask[small_regions[labeled]] = False

            pixel_count_filtered = int(np.sum(pixel_mask))
            removed_pixels = pixel_count - pixel_count_filtered
            if removed_pixels > 0:
                logger.debug(f"  Filtered {removed_pixels} pixels in "
                              f"{int(np.sum(small_regions)) - 1} small regions")
            pixel_count = pixel_count_filtered

        logger.debug(f"  {color_name}: layers {layer_start}-{layer_end} "
                      f"({layer_end - layer_start} layers, {pixel_count} px)")

        if pixel_count > 0:
            z_top_color = np.clip(pixel_height, z_bottom_flat, z_top_boundary)
            z_bottom_color = np.full_like(pixel_height, z_bottom_flat)

            thickness = z_top_color - z_bottom_flat
            color_effective_mask = pixel_mask & (thickness >= min_thickness)

            if color_effective_mask.any():
                z_top_max = np.max(z_top_color[color_effective_mask])
                logger.debug(f"    z: {z_bottom_flat:.2f} - {z_top_max:.2f}mm")

                output_path = output_base_path.parent / f"{output_base_path.stem}_{color_name}.stl"
                mesh = generate_topographical_stl(z_bottom_color, z_top_color,
                                                   color_effective_mask, pixel_size, layer_height)

                if len(mesh.vertices) > 0:
                    if collect_meshes:
                        mesh_outputs.append((mesh, output_path, filament['name']))
                    else:
                        mesh.export(str(output_path))
                        max_z_top = np.max(z_top_color[color_effective_mask])
                        layer_start_actual = int(np.floor(z_bottom_flat / layer_height))
                        layer_end_actual = int(np.ceil(max_z_top / layer_height))
                        generated_files.append((output_path, filament['name'], layer_start_actual, layer_end_actual))
                    logger.debug(f"  Generated: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
                else:
                    logger.debug(f"  Empty mesh for {color_name} - skipping")
        else:
            logger.debug(f"  No pixels for {color_name} - skipping")

    if collect_meshes:
        return mesh_outputs
    return generated_files
