#!/usr/bin/env python3
"""
HueCLI STL Generator — clean facade over modular pipeline.

Accepts PipelineConfig + ProcessedImage + filaments DataFrame,
delegates to modes.generate() for STL output.
"""

import logging

from .config import PipelineConfig, ProcessedImage
from . import modes

logger = logging.getLogger(__name__)


class STLGenerator:
    """Facade: config + image + filaments -> STL files."""

    def __init__(self, config: PipelineConfig, processed_image: ProcessedImage,
                 selected_filaments):
        self.config = config
        self.processed_image = processed_image
        self.selected_filaments = selected_filaments

    def generate(self, output_base_path):
        """Generate STL files — delegates to modes.generate()."""
        num_colors = len(self.selected_filaments)
        num_layers = self.config.num_layers
        logger.info(f"Generating {num_colors}-color model: {self.config.model_height}mm tall, {num_layers} layers")

        return modes.generate(self.config.mode, output_base_path, self.config,
                              self.processed_image, self.selected_filaments)
