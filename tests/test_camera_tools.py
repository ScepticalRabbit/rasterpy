# ============================================================================
# rasterpy: legacy DIC scalar-field rasterisers
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Regression tests for automatic perspective-camera placement."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import rasterpy


def test_fov_from_camera_rotation_uses_mesh_bounds() -> None:
    """An identity camera reports the mesh x/y bounding-box lengths."""
    coords = np.array(
        ((-2.0, -1.0, 0.0), (2.0, -1.0, 0.0),
         (2.0, 1.0, 0.0), (-2.0, 1.0, 0.0)),
    )

    fov = rasterpy.CameraTools.fov_from_cam_rot_3d(
        Rotation.identity(), coords,
    )

    np.testing.assert_allclose(fov, (4.0, 2.0))


def test_pos_fill_frame_centres_and_fits_mesh() -> None:
    """Automatic placement centres a mesh and retains the requested margin."""
    coords = np.array(
        ((-2.0, -1.0, 0.0, 1.0), (2.0, -1.0, 0.0, 1.0),
         (2.0, 1.0, 0.0, 1.0), (-2.0, 1.0, 0.0, 1.0)),
    )

    roi, position = rasterpy.CameraTools.pos_fill_frame(
        coords,
        pixel_num=np.array((400, 200)),
        pixel_size=np.array((0.01, 0.01)),
        focal_leng=0.05,
        cam_rot=Rotation.identity(),
        frame_fill=1.1,
    )

    np.testing.assert_allclose(roi, (0.0, 0.0, 0.0))
    np.testing.assert_allclose(position, (0.0, 0.0, 0.055))


def test_pos_fill_frame_rejects_zooming_past_mesh_bounds() -> None:
    """A fitting margin cannot crop the requested mesh bounds."""
    with pytest.raises(ValueError, match="frame_fill"):
        rasterpy.CameraTools.pos_fill_frame(
            np.array(((0.0, 0.0, 0.0),)),
            np.array((10, 10)),
            np.array((1.0, 1.0)),
            1.0,
            Rotation.identity(),
            frame_fill=0.9,
        )
