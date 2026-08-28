# ============================================================================== 
# rasterpy: legacy DIC scalar-field rasterisers
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================== 
"""Perspective camera data used by the legacy rasterisers."""

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(slots=True)
class Camera:
    """Camera data required by the legacy perspective rasterisers."""

    pixels_num: np.ndarray
    pixels_size: np.ndarray
    pos_world: np.ndarray
    rot_world: Rotation
    roi_cent_world: np.ndarray
    focal_length: float = 50.0
    sub_samp: int = 2
    bits: int = 16
    back_face_removal: bool = True
    sensor_size: np.ndarray = field(init=False)
    image_dims: np.ndarray = field(init=False)
    image_dist: float = field(init=False)
    cam_to_world_mat: np.ndarray = field(init=False)
    world_to_cam_mat: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.pixels_num = np.asarray(self.pixels_num, dtype=np.int32)
        self.pixels_size = np.asarray(self.pixels_size, dtype=np.float64)
        self.pos_world = np.asarray(self.pos_world, dtype=np.float64)
        self.roi_cent_world = np.asarray(self.roi_cent_world, dtype=np.float64)
        self.image_dist = float(np.linalg.norm(self.pos_world - self.roi_cent_world))
        self.sensor_size = self.pixels_num * self.pixels_size
        self.image_dims = self.image_dist * self.sensor_size / self.focal_length
        self.cam_to_world_mat = np.zeros((4, 4), dtype=np.float64)
        self.cam_to_world_mat[0:3, 0:3] = self.rot_world.as_matrix()
        self.cam_to_world_mat[-1, -1] = 1.0
        self.cam_to_world_mat[0:3, -1] = self.pos_world
        self.world_to_cam_mat = np.linalg.inv(self.cam_to_world_mat)


CameraData = Camera


@dataclass(slots=True)
class Camera2D:
    """An orthographic camera for planar image-warp renderers."""

    pixels_num: np.ndarray = field(
        default_factory=lambda: np.array((1000, 1000), dtype=np.int32),
    )
    pixels_size: float = 1.0e-3
    bits: int = 8
    roi_cent_world: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64),
    )
    background: float = 0.5
    sample_times: np.ndarray | None = None
    angle: Rotation | None = None
    subsample: int = 1
    field_of_view: np.ndarray = field(init=False)
    dynamic_range: int = field(init=False)
    background_code: float = field(init=False)
    world_to_cam: np.ndarray = field(init=False)
    cam_to_world: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        """Normalise arrays and calculate derived camera quantities."""
        self.pixels_num = np.asarray(self.pixels_num, dtype=np.int32)
        self.roi_cent_world = np.asarray(self.roi_cent_world, dtype=np.float64)
        self.field_of_view = self.pixels_size * self.pixels_num.astype(
            np.float64
        )
        self.dynamic_range = 2**self.bits
        self.background_code = self.background * float(self.dynamic_range)
        self.world_to_cam = self.field_of_view / 2.0 - self.roi_cent_world[:2]
        self.cam_to_world = -self.world_to_cam


__all__ = ["Camera", "Camera2D", "CameraData"]
