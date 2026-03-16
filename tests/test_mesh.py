import numpy as np
import pytest
from mesh import (
    greedy_mesh_rects, build_box_mesh, generate_topographical_stl,
    generate_flat_layer_stl, generate_quantized_stl,
)


class TestGreedyMesh:
    def test_all_true_single_rect(self):
        mask = np.ones((5, 5), dtype=bool)
        rects = greedy_mesh_rects(mask)
        assert len(rects) == 1
        assert rects[0] == (0, 0, 5, 5)

    def test_empty_mask(self):
        mask = np.zeros((5, 5), dtype=bool)
        rects = greedy_mesh_rects(mask)
        assert len(rects) == 0

    def test_diagonal_pixels(self):
        mask = np.eye(4, dtype=bool)
        rects = greedy_mesh_rects(mask)
        assert len(rects) == 4  # Each pixel is its own rect

    def test_covers_all_pixels(self):
        mask = np.random.rand(10, 10) > 0.5
        rects = greedy_mesh_rects(mask)
        covered = np.zeros_like(mask)
        for y0, x0, y1, x1 in rects:
            covered[y0:y1, x0:x1] = True
        np.testing.assert_array_equal(covered & mask, mask)


class TestBuildBoxMesh:
    def test_single_rect(self):
        rects = [(0, 0, 1, 1)]
        verts, faces = build_box_mesh(rects, 0.0, 1.0, 0.5)
        assert verts.shape == (8, 3)
        assert faces.shape == (12, 3)

    def test_n_rects(self):
        rects = [(0, 0, 1, 1), (2, 2, 3, 3), (4, 4, 5, 5)]
        verts, faces = build_box_mesh(rects, 0.0, 1.0, 0.5)
        assert verts.shape == (24, 3)  # 3 * 8
        assert faces.shape == (36, 3)  # 3 * 12

    def test_empty_rects(self):
        verts, faces = build_box_mesh([], 0.0, 1.0, 0.5)
        assert len(verts) == 0
        assert len(faces) == 0


class TestTopographicalSTL:
    def test_single_pixel(self):
        mask = np.array([[True]])
        z_bottom = np.array([[0.0]])
        z_top = np.array([[1.0]])
        mesh = generate_topographical_stl(z_bottom, z_top, mask, 0.5, 0.08)
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0

    def test_empty_mask(self):
        mask = np.zeros((3, 3), dtype=bool)
        z_bottom = np.zeros((3, 3))
        z_top = np.ones((3, 3))
        mesh = generate_topographical_stl(z_bottom, z_top, mask, 0.5, 0.08)
        assert len(mesh.vertices) == 0


class TestFlatLayerSTL:
    def test_produces_mesh(self):
        mask = np.ones((5, 5), dtype=bool)
        mesh = generate_flat_layer_stl(mask, 0, 2, 0.5, 0.08)
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0

    def test_empty_mask(self):
        mask = np.zeros((5, 5), dtype=bool)
        mesh = generate_flat_layer_stl(mask, 0, 2, 0.5, 0.08)
        assert len(mesh.vertices) == 0


class TestQuantizedSTL:
    def test_produces_mesh(self):
        mask = np.ones((5, 5), dtype=bool)
        z_bottom = np.zeros((5, 5))
        z_top = np.full((5, 5), 0.24)  # 3 layers at 0.08
        mesh = generate_quantized_stl(z_bottom, z_top, mask, 0.5, 0.08)
        assert len(mesh.vertices) > 0

    def test_equal_heights_empty(self):
        mask = np.ones((3, 3), dtype=bool)
        z = np.ones((3, 3)) * 0.5
        mesh = generate_quantized_stl(z, z, mask, 0.5, 0.08)
        assert len(mesh.vertices) == 0
