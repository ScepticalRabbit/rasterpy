# ============================================================================
# rasterpy: legacy DIC scalar-field rasterisers
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
import numpy as np
import pytest

import rasterpy


def test_numpy_static_triangle_regression(triangle_scene) -> None:
    """A static triangle renders a stable non-empty scalar image."""
    _, _, scene = triangle_scene
    renderer = rasterpy.RasterNumpy(rasterpy.RasterOpts(parallel=None))
    image = renderer.render(scene)

    assert image.shape == (24, 32)
    assert np.count_nonzero(np.isfinite(image)) == 216
    assert np.nanmax(image) == pytest.approx(0.9791666666666666)
    assert np.nanmin(image) == pytest.approx(0.020833333333333332)


def test_numpy_scene_returns_one_image_stack_per_camera(triangle_scene) -> None:
    """The all-frames API retains its camera/frame/field layout."""
    _, _, scene = triangle_scene
    renderer = rasterpy.RasterNumpy(rasterpy.RasterOpts(parallel=None))
    images = renderer.render_all(scene)

    assert images is not None
    assert len(images) == 1
    assert images[0].shape == (24, 32, 1, 1)
    assert np.nanmax(images[0]) == pytest.approx(0.9791666666666666)
