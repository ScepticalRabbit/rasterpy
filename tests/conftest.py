# ============================================================================
# rasterpy: legacy DIC scalar-field rasterisers
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import rasterpy


@pytest.fixture
def triangle_scene() -> tuple[rasterpy.Camera, rasterpy.Mesh, rasterpy.Scene]:
    """Create a small front-facing static triangle scene."""
    camera = rasterpy.Camera(
        pixels_num=np.array((32, 24)),
        pixels_size=np.array((0.05, 0.05)),
        pos_world=np.array((0.0, 0.0, 2.0)),
        rot_world=Rotation.identity(),
        roi_cent_world=np.zeros((3,)),
        focal_length=1.0,
        sub_samp=1,
    )
    mesh = rasterpy.Mesh(
        coords=np.array(
            ((-1.2, -0.9, 0.0, 1.0), (0.0, 0.9, 0.0, 1.0),
             (1.2, -0.9, 0.0, 1.0)),
            dtype=np.float64,
        ),
        connectivity=np.array(((0, 2, 1),), dtype=np.uintp),
        fields_render=np.array(((1.0,), (0.5,), (0.0,)), dtype=np.float64)
        .reshape(3, 1, 1),
    )
    return camera, mesh, rasterpy.Scene([camera], [mesh])
