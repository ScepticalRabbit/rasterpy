# ============================================================================
# rasterpy: legacy DIC scalar-field rasterisers
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
import numpy as np

import rasterpy


def test_cython_static_triangle_matches_numpy(triangle_scene) -> None:
    """The Cython static renderer matches NumPy for one frame."""
    camera, mesh, scene = triangle_scene
    numpy_image = rasterpy.RasterNumpy(
        rasterpy.RasterOpts(parallel=None),
    ).render(scene)
    images, depths, elements = rasterpy.RasterCY.raster_static_mesh(camera, mesh)

    foreground = np.isfinite(numpy_image)
    np.testing.assert_allclose(images[:, :, 0, 0][foreground],
                               numpy_image[foreground])
    assert np.all(images[:, :, 0, 0][~foreground] == 0.0)
    assert depths.shape == (24, 32, 1)
    np.testing.assert_array_equal(elements, np.array((1.0,)))
