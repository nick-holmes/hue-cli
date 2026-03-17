import pytest
from huecli.config import PipelineConfig, ProcessedImage, GeneratedFile
import numpy as np
from pathlib import Path


class TestPipelineConfig:
    def test_mode_properties(self):
        cfg = PipelineConfig(mode='flat')
        assert cfg.use_flat is True
        assert cfg.use_flat_cap is False
        assert cfg.use_exploded is False

    def test_flat_cap_mode(self):
        cfg = PipelineConfig(mode='flat-cap')
        assert cfg.use_flat_cap is True
        assert cfg.use_flat is False

    def test_exploded_modes(self):
        for mode in ['exploded', 'exploded-multi', 'exploded-cmyk']:
            cfg = PipelineConfig(mode=mode)
            assert cfg.exploded_any is True

    def test_standard_mode(self):
        cfg = PipelineConfig(mode='standard')
        assert cfg.use_flat is False
        assert cfg.exploded_any is False

    def test_num_layers(self):
        cfg = PipelineConfig(model_height=2.0, layer_height=0.08)
        assert cfg.num_layers == 25

    def test_num_layers_none(self):
        cfg = PipelineConfig()
        assert cfg.num_layers is None


class TestProcessedImage:
    def test_auto_dimensions(self):
        img = ProcessedImage(
            image_rgb=np.zeros((60, 80, 3)),
            image_lab=np.zeros((60, 80, 3)),
            grayscale=np.zeros((60, 80)),
            alpha_mask=np.ones((60, 80)),
        )
        assert img.width_px == 80
        assert img.height_px == 60


class TestGeneratedFile:
    def test_fields(self):
        gf = GeneratedFile(
            path=Path('/tmp/test.stl'),
            color_name='Red',
            layer_start=0,
            layer_end=5,
        )
        assert gf.color_name == 'Red'
        assert gf.layer_start == 0
