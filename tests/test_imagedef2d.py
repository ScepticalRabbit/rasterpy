# ============================================================================== 
# rasterpy: legacy DIC scalar-field rasterisers
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================== 
"""Regression tests for ImageDef2D planar deformation."""

from pathlib import Path
import numpy as np
import pytest
import rasterpy as render

GOLD = Path(__file__).parent / "gold_imagedef2d"


def test_imagedef2d_basic_warp(tmp_path: Path) -> None:
    coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    connect = np.array([[0, 1, 2, 3]], dtype=np.uintp)
    displacements = np.zeros((2, 4, 2), dtype=np.float64)
    displacements[1, :, 0] = 0.1

    mesh = render.Mesh2D(
        render.EElementType.QUAD4,
        coords,
        connect,
        displacements,
    )
    camera = render.Camera2D(pixels_num=np.array([64, 64]), pixels_size=1.0)
    source_image = np.ones((64, 64), dtype=np.float64) * 128.0
    scene = render.Scene2D(mesh, camera, source_image=source_image)

    renderer = render.ImageDef2D(render.ImageDefOpts(save_path=tmp_path))
    result = renderer.render(scene)
    assert result.images.shape == (2, 1, 64, 64, 1)
