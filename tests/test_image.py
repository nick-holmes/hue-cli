import numpy as np
import pytest
from pathlib import Path
from huecli.image import ImageProcessor


@pytest.fixture
def test_image(tmp_path):
    from PIL import Image
    img = Image.new('RGB', (100, 80), color=(128, 64, 32))
    path = tmp_path / 'test.png'
    img.save(str(path))
    return str(path)


@pytest.fixture
def test_rgba_image(tmp_path):
    from PIL import Image
    img = Image.new('RGBA', (100, 80), color=(128, 64, 32, 200))
    path = tmp_path / 'test_alpha.png'
    img.save(str(path))
    return str(path)


class TestLoadAndPrepare:
    def test_correct_dimensions(self, test_image):
        proc = ImageProcessor(test_image, 20.0, 3)
        proc.load_and_prepare(nozzle_diameter=0.4)
        # 20mm / 0.4mm = 50 pixels wide
        assert proc.image.shape[1] == 50
        assert proc.image.shape[2] == 3

    def test_alpha_extraction(self, test_rgba_image):
        proc = ImageProcessor(test_rgba_image, 20.0, 3)
        proc.load_and_prepare(nozzle_diameter=0.4)
        assert proc.alpha_mask is not None
        assert proc.alpha_mask.shape == proc.image.shape[:2]

    def test_aspect_ratio(self, test_image):
        proc = ImageProcessor(test_image, 20.0, 3)
        proc.load_and_prepare(nozzle_diameter=0.4)
        H, W = proc.image.shape[:2]
        # Original is 100x80, aspect = 0.8
        assert abs(H / W - 0.8) < 0.1


class TestQuantizeColors:
    def test_returns_correct_count(self, test_image):
        proc = ImageProcessor(test_image, 20.0, 3)
        proc.load_and_prepare(nozzle_diameter=0.4)
        colors, kmeans, indices = proc.quantize_colors()
        assert len(colors) == 3
