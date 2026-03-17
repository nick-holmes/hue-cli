import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from skimage import color as skcolor
from huecli.config import PipelineConfig, ProcessedImage


def _make_test_data():
    """Create synthetic test image and filaments matching test_regression.py"""
    H, W = 60, 80
    img = np.zeros((H, W, 3), dtype=np.float64)
    img[:H//2, :W//2] = [0.8, 0.2, 0.1]
    img[:H//2, W//2:] = [0.1, 0.6, 0.2]
    img[H//2:, :W//2] = [0.1, 0.2, 0.7]
    img[H//2:, W//2:] = [0.9, 0.8, 0.3]
    gy = np.linspace(0.3, 1.0, H)[:, None]
    img *= gy[:, :, None]

    grayscale = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    alpha = np.ones((H, W))

    colors = [
        ('Dark Red', (0.3, 0.05, 0.05), 1.5),
        ('Forest Green', (0.05, 0.25, 0.05), 2.0),
        ('Navy Blue', (0.05, 0.05, 0.3), 1.8),
        ('Golden Yellow', (0.8, 0.7, 0.1), 3.0),
        ('Clear', (0.95, 0.95, 0.95), 15.0),
    ]
    df = pd.DataFrame({
        'name': [c[0] for c in colors],
        'rgb': [c[1] for c in colors],
        'transmission_distance': [c[2] for c in colors],
        'Brand': ['Test'] * len(colors),
    })
    df['lab'] = df['rgb'].apply(lambda rgb: skcolor.rgb2lab([[rgb]])[0][0])

    return img, grayscale, alpha, df


@pytest.fixture
def test_data():
    return _make_test_data()


@pytest.fixture
def output_dir(tmp_path):
    return tmp_path


class TestStandardMode:
    def test_generates_files(self, test_data, output_dir):
        from huecli import modes
        img, gray, alpha, filaments = test_data
        config = PipelineConfig(
            layer_height=0.08, model_height=1.5, width_mm=40.0,
            mode='standard', contrast_strength=2.0
        )
        proc = ProcessedImage(
            image_rgb=img, image_lab=skcolor.rgb2lab(img),
            grayscale=gray, alpha_mask=alpha
        )
        result = modes.generate('standard', output_dir / 'test.stl',
                                 config, proc, filaments.iloc[:4])
        assert len(result) > 0


class TestFlatMode:
    def test_generates_files(self, test_data, output_dir):
        from huecli import modes
        img, gray, alpha, filaments = test_data
        config = PipelineConfig(
            layer_height=0.08, model_height=1.5, width_mm=40.0,
            mode='flat', dither_mode='none', base_layers=1
        )
        proc = ProcessedImage(
            image_rgb=img, image_lab=skcolor.rgb2lab(img),
            grayscale=gray, alpha_mask=alpha
        )
        result = modes.generate('flat', output_dir / 'test.stl',
                                 config, proc, filaments.iloc[:4])
        assert len(result) > 0


class TestExplodedMode:
    def test_generates_files(self, test_data, output_dir):
        from huecli import modes
        img, gray, alpha, filaments = test_data
        config = PipelineConfig(
            layer_height=0.08, model_height=1.5, width_mm=40.0,
            mode='exploded', sandwich_layers=1, use_fill=False,
            base_layers=2, max_color_sandwiches=3, dither_mode='none'
        )
        proc = ProcessedImage(
            image_rgb=img, image_lab=skcolor.rgb2lab(img),
            grayscale=gray, alpha_mask=alpha
        )
        result = modes.generate('exploded', output_dir / 'test.stl',
                                 config, proc, filaments)
        assert len(result) > 0


class TestExplodedMultiMode:
    def test_generates_files(self, test_data, output_dir):
        from huecli import modes
        img, gray, alpha, filaments = test_data
        config = PipelineConfig(
            layer_height=0.08, model_height=1.5, width_mm=40.0,
            mode='exploded-multi', sandwich_layers=1, use_fill=False,
            base_layers=2, max_color_sandwiches=5, dither_mode='none'
        )
        proc = ProcessedImage(
            image_rgb=img, image_lab=skcolor.rgb2lab(img),
            grayscale=gray, alpha_mask=alpha
        )
        result = modes.generate('exploded-multi', output_dir / 'test.stl',
                                 config, proc, filaments)
        assert len(result) > 0
