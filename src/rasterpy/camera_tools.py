# ============================================================================== 
# rasterpy: legacy DIC scalar-field rasterisers
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================== 
"""Camera placement and image-processing utilities for rasterpy."""

import numpy as np
from scipy.spatial.transform import Rotation


class CameraTools:
    """Utilities retained from the original rasteriser support code."""

    @staticmethod
    def fov_from_cam_rot_3d(
        cam_rot: Rotation,
        coords_world: np.ndarray,
    ) -> np.ndarray:
        """Return the x/y extents of a mesh in the camera coordinate system.

        The mesh axis-aligned bounding box is transformed into camera
        coordinates, so the result includes the effect of camera orientation.
        ``coords_world`` can contain either three Cartesian columns or a
        trailing homogeneous-coordinate column.
        """
        coords = np.asarray(coords_world, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[0] == 0 or coords.shape[1] < 3:
            raise ValueError(
                "coords_world must be a non-empty array with at least three "
                "coordinate columns."
            )

        coords = coords[:, :3]
        bounds_min = np.min(coords, axis=0)
        bounds_max = np.max(coords, axis=0)
        corners_world = np.array(
            np.meshgrid(
                (bounds_min[0], bounds_max[0]),
                (bounds_min[1], bounds_max[1]),
                (bounds_min[2], bounds_max[2]),
                indexing="ij",
            )
        ).reshape(3, -1).T
        corners_cam = cam_rot.inv().apply(corners_world)
        return np.ptp(corners_cam, axis=0)[:2]

    @staticmethod
    def image_dist_from_fov_3d(
        pixel_num: np.ndarray,
        pixel_size: np.ndarray,
        focal_leng: float,
        fov_leng: np.ndarray,
    ) -> np.ndarray:
        """Return camera distances required to cover each requested FOV size."""
        sensor_dims = np.asarray(pixel_num, dtype=np.float64) * np.asarray(
            pixel_size, dtype=np.float64,
        )
        fov_leng = np.asarray(fov_leng, dtype=np.float64)
        if np.any(sensor_dims <= 0.0):
            raise ValueError("pixel_num and pixel_size must define a positive sensor.")
        if focal_leng <= 0.0:
            raise ValueError("focal_leng must be positive.")
        if np.any(fov_leng < 0.0):
            raise ValueError("fov_leng must not contain negative values.")

        return fov_leng * focal_leng / sensor_dims

    @staticmethod
    def pos_fill_frame(
        coords_world: np.ndarray,
        pixel_num: np.ndarray,
        pixel_size: np.ndarray,
        focal_leng: float,
        cam_rot: Rotation,
        frame_fill: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ROI centre and camera position that fit a mesh in frame.

        ``frame_fill`` is a multiplicative margin: values greater than one
        leave a border around the mesh.  The returned camera lies on the
        positive local-z axis defined by ``cam_rot``.
        """
        if frame_fill < 1.0:
            raise ValueError("frame_fill must be at least 1.0.")

        coords = np.asarray(coords_world, dtype=np.float64)
        fov_leng = frame_fill * CameraTools.fov_from_cam_rot_3d(
            cam_rot, coords,
        )
        image_dist = CameraTools.image_dist_from_fov_3d(
            pixel_num, pixel_size, focal_leng, fov_leng,
        )
        roi_cent_world = (
            np.max(coords[:, :3], axis=0) + np.min(coords[:, :3], axis=0)
        ) / 2.0
        cam_pos_world = roi_cent_world + np.max(image_dist) * cam_rot.apply(
            np.array((0.0, 0.0, 1.0)),
        )
        return roi_cent_world, cam_pos_world

    @staticmethod
    def average_subpixel_image(image: np.ndarray, sub_samp: int) -> np.ndarray:
        """Average a square subpixel buffer into an image buffer."""
        height, width = image.shape[0:2]
        output_shape = (height // sub_samp, sub_samp, width // sub_samp, sub_samp)
        if image.ndim == 2:
            return image.reshape(output_shape).mean(axis=(1, 3))
        return image.reshape(*output_shape, image.shape[2]).mean(axis=(1, 3))


__all__ = ["CameraTools"]
