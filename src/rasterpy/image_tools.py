# ============================================================================== 
# rasterpy: legacy DIC scalar-field rasterisers
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================== 
"""Image saving utilities used by the legacy rasterisers."""

from enum import Enum
from pathlib import Path
import warnings

import numpy as np
from PIL import Image


class EImageType(Enum):
    """Supported image output types."""

    TIFF = ".tiff"
    BMP = ".bmp"


class ImageTools:
    """Image formatting and file-name helpers."""

    @staticmethod
    def scale(image: np.ndarray, min_frac: float = 0.0,
              max_frac: float = 1.0) -> np.ndarray:
        """Scale finite image values to the requested intensity interval."""
        image_min = np.nanmin(image)
        image_max = np.nanmax(image)
        if image_max == image_min:
            return np.full_like(image, min_frac, dtype=np.float64)
        image_scaled = (image - image_min) / (image_max - image_min)
        return image_scaled * (max_frac - min_frac) + min_frac

    @staticmethod
    def digitise(image: np.ndarray, bits: int = 16,
                 min_frac: float = 0.0, max_frac: float = 1.0,
                 background_frac: float = 0.5) -> np.ndarray:
        """Scale a floating image and convert it to an unsigned image type."""
        image_digitised = ImageTools.scale(image, min_frac, max_frac)
        image_digitised[np.isnan(image)] = background_frac
        return _image_to_uint(np.round((2 ** bits - 1) * image_digitised), bits)

    @staticmethod
    def scale_digitise_save(save_file: Path, image: np.ndarray,
                            image_type: EImageType, bits: int = 16,
                            min_frac: float = 0.0, max_frac: float = 1.0,
                            background: float = 0.5) -> None:
        """Scale, digitise, and save an image in legacy raster orientation."""
        image_save = ImageTools.digitise(
            image, bits, min_frac, max_frac, background,
        )[::-1, :]
        Image.fromarray(image_save).save(save_file.with_suffix(image_type.value))

    @staticmethod
    def get_save_name(cam_num: int, frame_num: int, field_num: int | None,
                      cam_width: int = 2, frame_width: int = 4,
                      field_width: int = 2) -> str:
        """Return the historic render output file stem."""
        camera = str(cam_num).zfill(cam_width)
        frame = str(frame_num).zfill(frame_width)
        if field_num is None:
            return f"cam{camera}_frame{frame}"
        field = str(field_num).zfill(field_width)
        return f"cam{camera}_frame{frame}_field{field}"


def _image_to_uint(image: np.ndarray, bits: int) -> np.ndarray:
    if 0 < bits <= 8:
        return image.astype(np.uint8)
    if bits <= 16:
        return image.astype(np.uint16)
    if bits <= 32:
        return image.astype(np.uint32)
    warnings.warn("Image bit depth must be between 1 and 32; using uint16.")
    return image.astype(np.uint16)


def image_save(path: Path | str, image: np.ndarray) -> None:
    """Save an image array to disk in standard orientation."""
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    Image.fromarray(arr[::-1, :]).save(save_path)


__all__ = ["EImageType", "ImageTools", "image_save"]
