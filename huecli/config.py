"""Typed dataclasses and constants shared across all HueCLI modules."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd


COLOR_SCHEMES = {
    'greyscale': [
        '#000000', '#404040', '#808080', '#B0B0B0', '#FFFFFF',
    ],
    'cyberpunk': [
        '#0D0221', '#FF00FF', '#00FFFF', '#FF6600', '#FFFF00',
        '#8B00FF',
    ],
    'sepia': [
        '#2B1700', '#6B3A1F', '#A0724A', '#C4A47A', '#F5E6C8',
    ],
    'sunset': [
        '#1A0533', '#8B1A4A', '#E94E3D', '#F49D37', '#FFD662',
    ],
    'ocean': [
        '#001B2E', '#014F6B', '#0496A8', '#5CC8D4', '#D1F0F0',
    ],
    'vaporwave': [
        '#2B0A3D', '#FF71CE', '#B967FF', '#01CDFE', '#05FFA1',
    ],
    'autumn': [
        '#2D1B00', '#8B2500', '#CC5500', '#E09540', '#FFD700',
        '#556B2F',
    ],
    'nordic': [
        '#1C2833', '#4A6A7A', '#8EB8C4', '#C8DDE0', '#F0F5F5',
    ],
}


@dataclass
class PipelineConfig:
    """All CLI parameters for a HueCLI run. Fields are Optional until interactive fill."""
    image_path: str = ''
    filaments_csv: Optional[str] = None
    color_count: Optional[int] = None
    nozzle_diameter: Optional[float] = None
    layer_height: Optional[float] = None
    model_height: Optional[float] = None
    width_mm: Optional[float] = None
    height_mm: Optional[float] = None
    output_name: Optional[str] = None
    min_color_difference: Optional[float] = None
    mode: Optional[str] = None
    sandwich_layers: Optional[int] = None
    use_fill: bool = False
    base_layers: Optional[int] = None
    max_color_sandwiches: Optional[int] = None
    scheme: Optional[str] = None
    flip: Optional[str] = None
    dither_mode: str = 'none'
    contrast_strength: float = 2.0

    @property
    def use_flat(self) -> bool:
        return self.mode == 'flat'

    @property
    def use_flat_cap(self) -> bool:
        return self.mode == 'flat-cap'

    @property
    def use_exploded(self) -> bool:
        return self.mode == 'exploded'

    @property
    def use_exploded_multi(self) -> bool:
        return self.mode == 'exploded-multi'

    @property
    def use_exploded_cmyk(self) -> bool:
        return self.mode == 'exploded-cmyk'

    @property
    def exploded_any(self) -> bool:
        return self.mode in ('exploded', 'exploded-multi', 'exploded-cmyk')

    @property
    def num_layers(self) -> Optional[int]:
        if self.model_height is not None and self.layer_height is not None:
            return int(self.model_height / self.layer_height)
        return None


@dataclass
class ProcessedImage:
    """Result of image loading and preprocessing."""
    image_rgb: np.ndarray       # HxWx3, float 0-1
    image_lab: np.ndarray       # HxWx3, LAB
    grayscale: np.ndarray       # HxW, float 0-1
    alpha_mask: np.ndarray      # HxW, float 0-1
    width_px: int = 0
    height_px: int = 0

    def __post_init__(self):
        if self.image_rgb is not None and self.width_px == 0:
            self.height_px, self.width_px = self.image_rgb.shape[:2]


@dataclass
class FilamentSet:
    """Selected filaments for a generation run."""
    filaments_df: pd.DataFrame
    dominant_colors_lab: Optional[np.ndarray] = None


@dataclass
class GeneratedFile:
    """Metadata for a generated STL file."""
    path: Path
    color_name: str
    layer_start: int
    layer_end: int
