# ============================================================================
# rasterpy: legacy DIC scalar-field rasterisers
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Render a small scalar triangle with the legacy NumPy rasteriser."""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import rasterpy


def make_scene() -> rasterpy.Scene:
    """Create a self-contained static triangle scene."""
    camera = rasterpy.Camera(
        pixels_num=np.array((128, 96)),
        pixels_size=np.array((0.0001, 0.0001)),
        pos_world=np.array((0.0, 0.0, 2.0)),
        rot_world=Rotation.identity(),
        roi_cent_world=np.zeros((3,)),
        focal_length=0.05,
        sub_samp=2,
    )
    mesh = rasterpy.Mesh(
        coords=np.array(
            ((-0.22, -0.16, 0.0, 1.0), (0.0, 0.16, 0.0, 1.0),
             (0.22, -0.16, 0.0, 1.0)),
            dtype=np.float64,
        ),
        connectivity=np.array(((0, 2, 1),), dtype=np.uintp),
        fields_render=np.array(((1.0,), (0.5,), (0.0,)), dtype=np.float64)
        .reshape(3, 1, 1),
    )
    return rasterpy.Scene([camera], [mesh])


def main() -> None:
    """Render and save the example image."""
    image = rasterpy.RasterNumpy(
        rasterpy.RasterOpts(parallel=None),
    ).render(make_scene())
    output_path = Path.cwd() / "rasterpy-numpy-triangle"
    rasterpy.ImageTools.scale_digitise_save(
        output_path, image, rasterpy.EImageType.TIFF, bits=8,
    )
    print(f"Saved {output_path.with_suffix('.tiff')}")


if __name__ == "__main__":
    main()
