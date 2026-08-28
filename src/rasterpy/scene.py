
#===============================================================================
# rasterpy: legacy DIC scalar-field rasterisers
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
#===============================================================================
from dataclasses import dataclass
import numpy as np
from .camera import Camera2D, CameraData
from .mesh import Mesh2D, RenderMesh


@dataclass(slots=True)
class Scene2D:
    """Complete scene description for planar 2D image-warp renderers."""

    mesh: Mesh2D
    camera: Camera2D
    source_image: np.ndarray | None = None
    mask: np.ndarray | None = None


class RenderScene:
    __slots__ = ("cameras", "meshes")

    def __init__(
        self,
        cameras: list[CameraData] | None = None,
        meshes: list[RenderMesh] | None = None,
    ) -> None:
        if cameras is None:
            self.cameras = []
        else:
            self.cameras = cameras

        if meshes is None:
            self.meshes = []
        else:
            self.meshes = meshes

    def is_deformable(self) -> bool:
        for mm in self.meshes:
            if mm.fields_disp is not None:
                return True

        return False


def get_all_coords_world(meshes: list[RenderMesh]) -> np.ndarray:
    coords_all = []
    for mm in meshes:
        coords_all.append(np.matmul(mm.coords, mm.mesh_to_world_mat.T))

    return np.vstack(coords_all)


Scene = RenderScene

__all__ = ["RenderScene", "Scene", "Scene2D", "get_all_coords_world"]
