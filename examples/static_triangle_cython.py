# ============================================================================
# rasterpy: legacy DIC scalar-field rasterisers
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Render a small scalar triangle with the legacy Cython rasteriser."""

from pathlib import Path

import rasterpy
from static_triangle_numpy import make_scene


def main() -> None:
    """Render and save the Cython example image."""
    scene = make_scene()
    images, _, _ = rasterpy.RasterCY.raster_static_mesh(
        scene.cameras[0], scene.meshes[0],
    )
    output_path = Path.cwd() / "rasterpy-cython-triangle"
    rasterpy.ImageTools.scale_digitise_save(
        output_path, images[:, :, 0, 0], rasterpy.EImageType.TIFF, bits=8,
    )
    print(f"Saved {output_path.with_suffix('.tiff')}")


if __name__ == "__main__":
    main()
