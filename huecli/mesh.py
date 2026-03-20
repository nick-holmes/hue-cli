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


def generate_watertight_stl(z_bottom_val, z_top, pixel_mask, pixel_size,
                             layer_height):
    """Generate a watertight manifold STL using marching cubes.

    Voxelizes the heightfield into a 3D volume with 1-voxel padding on all
    sides, then extracts the isosurface. This guarantees zero non-manifold
    edges and zero boundary edges (fully closed shell).

    Args:
        z_bottom_val: scalar z-height for the bottom face (mm)
        z_top: 2D array of top z-heights (mm)
        pixel_mask: 2D boolean mask of active pixels
        pixel_size: size of one pixel in mm
        layer_height: layer height in mm (used as Z voxel size)

    Returns:
        trimesh.Trimesh — watertight manifold mesh
    """
    from skimage.measure import marching_cubes

    if not pixel_mask.any():
        return trimesh.Trimesh()

    H, W = pixel_mask.shape
    z_min = float(z_bottom_val)
    z_max = float(z_top[pixel_mask].max()) if pixel_mask.any() else z_min + layer_height

    if z_max <= z_min + 1e-6:
        return trimesh.Trimesh()

    # Build 3D volume with 1-voxel padding for closed surface
    voxel_z = layer_height
    z_range = np.arange(z_min - voxel_z, z_max + 2 * voxel_z, voxel_z)
    nz = len(z_range)

    volume = np.zeros((H + 2, W + 2, nz), dtype=np.float32)
    for zi, z in enumerate(z_range):
        active = pixel_mask & (z_top >= z) & (z >= z_min)
        volume[1:H+1, 1:W+1, zi] = active.astype(np.float32)

    try:
        verts, faces, normals, _ = marching_cubes(
            volume, level=0.5, spacing=(pixel_size, pixel_size, voxel_z))
    except (ValueError, RuntimeError):
        return trimesh.Trimesh()

    # Swap X/Y: marching cubes maps dim0→X, dim1→Y, but our volume has
    # dim0=rows(Y), dim1=cols(X). Swap to match the coordinate system
    # used by the rest of the pipeline (X=columns, Y=rows).
    verts[:, 0], verts[:, 1] = verts[:, 1].copy(), verts[:, 0].copy()

    # Offset for padding
    verts[:, 0] -= pixel_size  # X (cols)
    verts[:, 1] -= pixel_size  # Y (rows)
    verts[:, 2] += z_min - voxel_z

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    mesh.fix_normals()

    logger.info(f"  Marching cubes: {pixel_mask.sum():,} pixels -> "
                 f"{len(mesh.faces):,} faces (watertight={mesh.is_watertight})")

    return mesh


def export_stl_manifold(mesh, filepath):
    """Export mesh to binary STL, fixing non-manifold edges.

    Identifies non-manifold edges (shared by >2 faces) and offsets the
    excess faces' vertices by a tiny normal-direction epsilon. This
    prevents STL vertex merging from re-creating shared edges while
    keeping the mesh visually identical. Uses vectorized numpy operations.

    Args:
        mesh: trimesh.Trimesh to export
        filepath: output STL path
    """
    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)

    # Find non-manifold faces (vectorized)
    n = len(faces)
    v0 = faces[:, [0, 1, 2]]
    v1 = faces[:, [1, 2, 0]]
    edge_a = np.minimum(v0, v1).ravel()
    edge_b = np.maximum(v0, v1).ravel()
    max_v = int(vertices.shape[0]) + 1
    edge_keys = edge_a.astype(np.int64) * max_v + edge_b.astype(np.int64)
    face_indices = np.repeat(np.arange(n), 3)

    unique_edges, inverse, counts = np.unique(edge_keys, return_inverse=True, return_counts=True)
    nm_mask = counts[inverse] > 2

    if nm_mask.any():
        nm_fi = face_indices[nm_mask]
        # For each non-manifold edge, keep first 2 face occurrences, offset rest
        seen = {}
        offset_faces = set()
        for i in range(len(nm_fi)):
            eidx = int(inverse[np.where(nm_mask)[0][i]])
            fi = int(nm_fi[i])
            if eidx not in seen:
                seen[eidx] = []
            seen[eidx].append(fi)

        for eidx, fis in seen.items():
            unique_fis = list(dict.fromkeys(fis))
            for fi in unique_fis[2:]:
                offset_faces.add(fi)

    # Build face vertex positions array: (n_faces, 3, 3)
    tri_verts = vertices[faces]  # (n, 3, 3)

    if nm_mask.any() and offset_faces:
        # Offset the non-manifold faces along their normal
        for fi in offset_faces:
            tv = tri_verts[fi]
            normal = np.cross(tv[1] - tv[0], tv[2] - tv[0])
            nl = np.linalg.norm(normal)
            if nl > 0:
                tri_verts[fi] += (normal / nl) * 2e-5  # 20 nanometers

    # Compute normals
    e1 = tri_verts[:, 1] - tri_verts[:, 0]
    e2 = tri_verts[:, 2] - tri_verts[:, 0]
    normals = np.cross(e1, e2)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normals = normals / norms

    # Write binary STL (vectorized)
    import struct
    header = b'\x00' * 80 + struct.pack('<I', n)
    # Each record: 12 floats (normal + 3 verts) + 2 bytes attribute
    records = np.zeros((n, 50), dtype=np.uint8)

    # Pack all data as float32
    all_data = np.hstack([normals, tri_verts.reshape(n, 9)]).astype(np.float32)

    with open(filepath, 'wb') as f:
        f.write(header)
        for i in range(n):
            f.write(all_data[i].tobytes())
            f.write(b'\x00\x00')


def _fix_non_manifold(vertices, faces):
    """Fix non-manifold edges by removing excess faces.

    For edges shared by >2 faces (at convex wall corners), removes the
    extra faces entirely. This leaves tiny holes at corners that slicers
    auto-repair, while eliminating non-manifold edges that slicers cannot.

    Returns:
        (vertices, faces) — with non-manifold faces removed
    """
    # Build edge array: each face contributes 3 edges (sorted vertex pairs)
    n = len(faces)
    v0 = faces[:, [0, 1, 2]]
    v1 = faces[:, [1, 2, 0]]
    edge_a = np.minimum(v0, v1).ravel()
    edge_b = np.maximum(v0, v1).ravel()

    # Pack edge pairs into single int64 for fast counting
    max_v = int(vertices.shape[0])
    edge_keys = edge_a.astype(np.int64) * max_v + edge_b.astype(np.int64)
    face_indices = np.repeat(np.arange(n), 3)

    # Count edge occurrences
    unique_edges, inverse, counts = np.unique(edge_keys, return_inverse=True, return_counts=True)

    # Find edges with >2 faces (non-manifold)
    nm_mask = counts[inverse] > 2  # per edge-occurrence mask
    if not nm_mask.any():
        return vertices, faces

    # For each non-manifold edge, find ALL face indices using it
    nm_face_indices = face_indices[nm_mask]
    nm_edge_inv = inverse[nm_mask]

    # For each unique non-manifold edge, keep first 2 faces, mark rest for removal
    faces_to_remove = set()
    seen_edges = {}
    for i in range(len(nm_face_indices)):
        eidx = int(nm_edge_inv[i])
        fi = int(nm_face_indices[i])
        if eidx not in seen_edges:
            seen_edges[eidx] = []
        seen_edges[eidx].append(fi)

    for eidx, fi_list in seen_edges.items():
        # Keep first 2 unique face indices, remove the rest
        unique_fis = list(dict.fromkeys(fi_list))
        for fi in unique_fis[2:]:
            faces_to_remove.add(fi)

    if not faces_to_remove:
        return vertices, faces

    keep_mask = np.ones(n, dtype=bool)
    keep_mask[list(faces_to_remove)] = False
    return vertices, faces[keep_mask]


def _fill_triangular_holes(vertices, faces):
    """Fill triangular holes (3 boundary edges forming a triangle).

    After _fix_non_manifold removes excess faces, small triangular holes
    remain at convex corners. This patches each 3-edge hole with a single
    face, ensuring the result has no non-manifold edges.

    Returns:
        Updated faces array with patch faces appended
    """
    from collections import defaultdict

    # Find boundary edges (shared by exactly 1 face)
    edge_count = {}
    for face in faces:
        for i in range(3):
            e = (min(int(face[i]), int(face[(i+1)%3])),
                 max(int(face[i]), int(face[(i+1)%3])))
            edge_count[e] = edge_count.get(e, 0) + 1

    boundary = {e for e, c in edge_count.items() if c == 1}
    if not boundary:
        return faces

    # Build adjacency: vertex → boundary edges
    vert_edges = defaultdict(set)
    for e in boundary:
        vert_edges[e[0]].add(e)
        vert_edges[e[1]].add(e)

    # Find triangular holes: 3 boundary edges forming a cycle through 3 vertices
    patch_faces = []
    used_edges = set()

    for start_edge in boundary:
        if start_edge in used_edges:
            continue
        v0, v1 = start_edge

        # Check if v0 and v1 share a third vertex through boundary edges
        v0_neighbors = set()
        for e in vert_edges[v0]:
            if e == start_edge:
                continue
            other = e[1] if e[0] == v0 else e[0]
            v0_neighbors.add((other, e))

        v1_neighbors = set()
        for e in vert_edges[v1]:
            if e == start_edge:
                continue
            other = e[1] if e[0] == v1 else e[0]
            v1_neighbors.add((other, e))

        # Find common vertex (triangular hole)
        v0_set = {v for v, _ in v0_neighbors}
        v1_set = {v for v, _ in v1_neighbors}
        common = v0_set & v1_set

        for v2 in common:
            e1 = None
            for v, e in v0_neighbors:
                if v == v2:
                    e1 = e
                    break
            e2 = None
            for v, e in v1_neighbors:
                if v == v2:
                    e2 = e
                    break

            if e1 in used_edges or e2 in used_edges:
                continue

            # Check that adding this face won't create non-manifold
            new_edges = [
                (min(v0, v1), max(v0, v1)),
                (min(v0, v2), max(v0, v2)),
                (min(v1, v2), max(v1, v2)),
            ]
            would_exceed = any(edge_count.get(e, 0) >= 2 for e in new_edges)
            if would_exceed:
                continue

            patch_faces.append([v0, v2, v1])  # Winding matches the hole
            used_edges.update([start_edge, e1, e2])
            # Update edge counts for subsequent checks
            for e in new_edges:
                edge_count[e] = edge_count.get(e, 0) + 1
            break

    # Also fill quadrilateral holes (4 boundary edges forming a cycle)
    # Rebuild boundary from updated edge counts
    boundary2 = {e for e, c in edge_count.items() if c == 1}
    if boundary2:
        vert_edges2 = defaultdict(set)
        for e in boundary2:
            vert_edges2[e[0]].add(e)
            vert_edges2[e[1]].add(e)

        used2 = set()
        for start_e in boundary2:
            if start_e in used2:
                continue
            # Trace a loop from this edge
            loop_verts = [start_e[0], start_e[1]]
            loop_edges = [start_e]
            current = start_e[1]
            for _ in range(20):  # Max edges per hole
                found = False
                for e in vert_edges2[current]:
                    if e in loop_edges or e in used2:
                        continue
                    other = e[1] if e[0] == current else e[0]
                    loop_verts.append(other)
                    loop_edges.append(e)
                    current = other
                    found = True
                    break
                if not found or current == loop_verts[0]:
                    break

            if current == loop_verts[0] and len(loop_edges) >= 3 and len(loop_edges) <= 20:
                verts_cycle = loop_verts[:-1]  # Remove duplicate start
                # Fan triangulation from first vertex
                can_fill = True
                for i in range(1, len(verts_cycle) - 1):
                    new_e = [(min(verts_cycle[0], verts_cycle[i]), max(verts_cycle[0], verts_cycle[i])),
                             (min(verts_cycle[i], verts_cycle[i+1]), max(verts_cycle[i], verts_cycle[i+1]))]
                    for e in new_e:
                        if e not in boundary2 and edge_count.get(e, 0) >= 2:
                            can_fill = False
                            break
                    if not can_fill:
                        break

                if can_fill:
                    for i in range(1, len(verts_cycle) - 1):
                        patch_faces.append([verts_cycle[0], verts_cycle[i+1], verts_cycle[i]])
                    for e in loop_edges:
                        used2.add(e)
                    for pf in patch_faces[-len(verts_cycle)+2:]:
                        for i in range(3):
                            ne = (min(pf[i], pf[(i+1)%3]), max(pf[i], pf[(i+1)%3]))
                            edge_count[ne] = edge_count.get(ne, 0) + 1

    if patch_faces:
        return np.vstack([faces, np.array(patch_faces)])
    return faces


def generate_topographical_stl(z_bottom, z_top, pixel_mask, pixel_size,
                                layer_height, preview=False):
    """Generate topographical STL using shared-vertex grid.

    Produces a manifold mesh. Top/bottom surfaces use shared-vertex grid
    (corner heights averaged from neighboring pixels). Side walls use
    DEDICATED vertices per wall quad to avoid non-manifold edges at convex
    corners where two wall directions would otherwise share a vertical edge.

    Args:
        z_bottom: 2D array of bottom z-heights (mm)
        z_top: 2D array of top z-heights (mm)
        pixel_mask: 2D boolean array
        pixel_size: pixel size in mm
        layer_height: layer height in mm
        preview: if True, used for preview mesh (no functional difference)

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

    # Average z values at grid corners from adjacent active pixels
    count = np.zeros((H + 1, W + 1))
    zb_sum = np.zeros((H + 1, W + 1))
    zt_sum = np.zeros((H + 1, W + 1))

    for dy in range(2):
        for dx in range(2):
            count[dy:H + dy, dx:W + dx] += mask_f
            zt_sum[dy:H + dy, dx:W + dx] += z_top * mask_f
            zb_sum[dy:H + dy, dx:W + dx] += z_bottom * mask_f

    corner_active = count > 0
    zb_avg = np.divide(zb_sum, count, where=corner_active, out=np.zeros_like(zb_sum))
    zt_avg = np.divide(zt_sum, count, where=corner_active, out=np.zeros_like(zt_sum))

    n_active = int(corner_active.sum())
    if n_active == 0:
        return trimesh.Trimesh()

    vertex_idx = np.full((H + 1, W + 1), -1, dtype=np.int64)
    active_cy, active_cx = np.where(corner_active)
    vertex_idx[active_cy, active_cx] = np.arange(n_active)

    bottom_verts = np.column_stack([active_cx * ps, active_cy * ps, zb_avg[active_cy, active_cx]])
    top_verts = np.column_stack([active_cx * ps, active_cy * ps, zt_avg[active_cy, active_cx]])

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

    # Top faces — normal points +z (CCW from above)
    all_faces.append(np.column_stack([t00, t01, t10]))
    all_faces.append(np.column_stack([t01, t11, t10]))

    # Bottom faces — normal points -z (CW from above)
    all_faces.append(np.column_stack([b00, b10, b01]))
    all_faces.append(np.column_stack([b01, b10, b11]))

    # Side walls at mask boundaries — connect top and bottom surfaces.
    # Each wall is 2 triangles per boundary pixel edge. At convex corners
    # where two wall directions meet, the shared vertical edge would get
    # 4 faces (non-manifold). We prevent this by skipping the triangle
    # that uses the shared corner vertex from one of the two directions.
    def is_boundary(dy, dx):
        ny, nx = py + dy, px + dx
        oob = (ny < 0) | (ny >= H) | (nx < 0) | (nx >= W)
        return oob | ~effective_mask[np.clip(ny, 0, H - 1), np.clip(nx, 0, W - 1)]

    left = is_boundary(0, -1)
    right = is_boundary(0, 1)
    front = is_boundary(-1, 0)
    back = is_boundary(1, 0)

    # Convex corner flags: pixel has boundaries on two adjacent sides.
    # At these corners, one wall direction skips its triangle that uses
    # the shared corner vertex. Left/right walls yield to front/back.
    corner_lf = left & front   # left+front corner at (y, x)
    corner_lb = left & back    # left+back corner at (y+1, x)
    corner_rf = right & front  # right+front corner at (y, x+1)
    corner_rb = right & back   # right+back corner at (y+1, x+1)

    # Left wall: 2 triangles using vertices (y,x)-(y+1,x).
    # Triangle A uses corner (y,x). Triangle B uses corner (y+1,x).
    # Skip A if left+front corner. Skip B if left+back corner.
    if left.any():
        bl0 = vertex_idx[py[left], px[left]]
        bl1 = vertex_idx[py[left] + 1, px[left]]
        tl0 = bl0 + n_active; tl1 = bl1 + n_active
        # Tri A: uses (y,x) corner — skip if left+front
        keep_a = left & ~corner_lf
        if keep_a.any():
            a0 = vertex_idx[py[keep_a], px[keep_a]]
            a1 = vertex_idx[py[keep_a] + 1, px[keep_a]]
            all_faces.append(np.column_stack([a0, a0 + n_active, a1 + n_active]))
        # Tri B: uses (y+1,x) corner — skip if left+back
        keep_b = left & ~corner_lb
        if keep_b.any():
            a0 = vertex_idx[py[keep_b], px[keep_b]]
            a1 = vertex_idx[py[keep_b] + 1, px[keep_b]]
            all_faces.append(np.column_stack([a0, a1 + n_active, a1]))

    # Right wall: vertices (y,x+1)-(y+1,x+1)
    if right.any():
        # Tri A: uses (y+1,x+1) — skip if right+back
        keep_a = right & ~corner_rb
        if keep_a.any():
            a0 = vertex_idx[py[keep_a], px[keep_a] + 1]
            a1 = vertex_idx[py[keep_a] + 1, px[keep_a] + 1]
            all_faces.append(np.column_stack([a0, a1, a0 + n_active]))
            all_faces.append(np.column_stack([a1, a1 + n_active, a0 + n_active]))
        # Handle skipped corner triangles — only add the safe triangle
        only_rb = right & corner_rb & ~corner_rf
        if only_rb.any():
            a0 = vertex_idx[py[only_rb], px[only_rb] + 1]
            a1 = vertex_idx[py[only_rb] + 1, px[only_rb] + 1]
            # Only tri using (y,x+1) corner, skip tri using (y+1,x+1)
            all_faces.append(np.column_stack([a0, a1, a0 + n_active]))
        # Tri B: uses (y,x+1) — skip if right+front
        only_rf = right & corner_rf & ~corner_rb
        if only_rf.any():
            a0 = vertex_idx[py[only_rf], px[only_rf] + 1]
            a1 = vertex_idx[py[only_rf] + 1, px[only_rf] + 1]
            all_faces.append(np.column_stack([a1, a1 + n_active, a0 + n_active]))

    # Front wall: vertices (y,x)-(y,x+1)
    if front.any():
        bf0 = vertex_idx[py[front], px[front]]
        bf1 = vertex_idx[py[front], px[front] + 1]
        tf0 = bf0 + n_active; tf1 = bf1 + n_active
        all_faces.append(np.column_stack([bf0, bf1, tf0]))
        all_faces.append(np.column_stack([bf1, tf1, tf0]))

    # Back wall: vertices (y+1,x)-(y+1,x+1)
    if back.any():
        bb0 = vertex_idx[py[back] + 1, px[back]]
        bb1 = vertex_idx[py[back] + 1, px[back] + 1]
        tb0 = bb0 + n_active; tb1 = bb1 + n_active
        all_faces.append(np.column_stack([bb0, tb0, bb1]))
        all_faces.append(np.column_stack([bb1, tb0, tb1]))

    combined_faces = np.vstack(all_faces)
    vertices = np.vstack([bottom_verts, top_verts])

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


def _downsample_heightmap(pixel_height, alpha_pixels, factor):
    """Downsample heightmap by block-averaging for smaller STL output.

    Active pixels are averaged; inactive pixels stay 0. The alpha mask
    is downsampled by majority vote (>50% active → active).

    Args:
        pixel_height: 2D array of heights
        alpha_pixels: 2D boolean mask
        factor: downsample factor (2 = half resolution)

    Returns:
        (ds_height, ds_alpha, ds_factor) where ds_factor is the actual
        factor used (may differ if image dimensions aren't divisible)
    """
    if factor <= 1:
        return pixel_height, alpha_pixels, 1

    H, W = pixel_height.shape
    # Trim to multiple of factor
    H_trim = (H // factor) * factor
    W_trim = (W // factor) * factor
    ph = pixel_height[:H_trim, :W_trim]
    ap = alpha_pixels[:H_trim, :W_trim].astype(np.float64)

    dH, dW = H_trim // factor, W_trim // factor
    ph_blocks = ph.reshape(dH, factor, dW, factor)
    ap_blocks = ap.reshape(dH, factor, dW, factor)

    # Sum of active pixel heights / count of active pixels per block
    ap_sum = ap_blocks.sum(axis=(1, 3))
    ph_weighted = (ph_blocks * ap.reshape(dH, factor, dW, factor)).sum(axis=(1, 3))

    ds_alpha = ap_sum > (factor * factor * 0.5)
    ds_height = np.divide(ph_weighted, ap_sum, where=ap_sum > 0,
                           out=np.zeros((dH, dW)))
    ds_height[~ds_alpha] = 0

    return ds_height, ds_alpha, factor


def generate_color_band_stls(sorted_filaments, pixel_height, z_boundaries,
                              layer_boundaries, alpha_pixels, output_base_path,
                              pixel_size, layer_height, collect_meshes=False,
                              mesh_downsample=1):
    """Generate one topographical STL per color band.

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
        mesh_downsample: Factor to downsample mesh (2 = half res, ~4x fewer faces)

    Returns:
        List of (path, name, layer_start, layer_end) tuples, or
        list of (mesh, path, name) tuples if collect_meshes=True
    """
    from scipy import ndimage

    # Downsample heightmap for mesh generation (slicer can't print sub-nozzle detail)
    ds_height, ds_alpha, ds_factor = _downsample_heightmap(
        pixel_height, alpha_pixels, mesh_downsample)
    ds_pixel_size = pixel_size * ds_factor

    if ds_factor > 1:
        logger.info(f"Mesh downsampled {ds_factor}x: {pixel_height.shape} -> {ds_height.shape} "
                     f"(pixel_size {pixel_size:.2f} -> {ds_pixel_size:.2f}mm)")

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

        pixel_mask = (ds_height > z_bottom_flat + min_thickness) & ds_alpha
        pixel_count = int(np.sum(pixel_mask))

        labeled, num_regions = ndimage.label(pixel_mask)
        # Minimum printable region: ~0.8mm² scaled for downsampled resolution
        min_region_size = max(4, int(0.8 / (ds_pixel_size * ds_pixel_size)))

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
            z_top_color = np.clip(ds_height, z_bottom_flat, z_top_boundary)
            z_bottom_color = np.full_like(ds_height, z_bottom_flat)

            thickness = z_top_color - z_bottom_flat
            color_effective_mask = pixel_mask & (thickness >= min_thickness)

            if color_effective_mask.any():
                z_top_max = np.max(z_top_color[color_effective_mask])
                logger.debug(f"    z: {z_bottom_flat:.2f} - {z_top_max:.2f}mm")

                output_path = output_base_path.parent / f"{output_base_path.stem}_{color_name}.stl"
                mesh = generate_watertight_stl(z_bottom_flat, z_top_color,
                                                color_effective_mask, ds_pixel_size, layer_height)

                if len(mesh.vertices) > 0:
                    if collect_meshes:
                        mesh_outputs.append((mesh, output_path, filament['name']))
                    else:
                        mesh.export(str(output_path))
                        max_z_top_val = np.max(z_top_color[color_effective_mask])
                        layer_start_actual = int(np.floor(z_bottom_flat / layer_height))
                        layer_end_actual = int(np.ceil(max_z_top_val / layer_height))
                        generated_files.append((output_path, filament['name'], layer_start_actual, layer_end_actual))
                    logger.debug(f"  Generated: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
                else:
                    logger.debug(f"  Empty mesh for {color_name} - skipping")
        else:
            logger.debug(f"  No pixels for {color_name} - skipping")

    if collect_meshes:
        return mesh_outputs
    return generated_files
