# ============================================================================== 
# rasterpy: legacy DIC scalar-field rasterisers
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================== 
"""Small camera utilities required by the NumPy rasteriser."""

import numpy as np


class CameraTools:
    """Utilities retained from the original rasteriser support code."""

    @staticmethod
    def average_subpixel_image(image: np.ndarray, sub_samp: int) -> np.ndarray:
        """Average a square subpixel buffer into an image buffer."""
        height, width = image.shape[0:2]
        output_shape = (height // sub_samp, sub_samp, width // sub_samp, sub_samp)
        if image.ndim == 2:
            return image.reshape(output_shape).mean(axis=(1, 3))
        return image.reshape(*output_shape, image.shape[2]).mean(axis=(1, 3))


__all__ = ["CameraTools"]
