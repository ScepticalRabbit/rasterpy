# ============================================================================== 
# rasterpy: legacy DIC scalar-field rasterisers
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================== 
"""Mesh data used by the legacy rasterisers."""

import numpy as np
from scipy.spatial.transform import Rotation


class Mesh:
    """A scalar-field render mesh with optional nodal displacement history."""

    __slots__ = (
        "coords", "connectivity", "fields_render", "fields_disp",
        "pos_world", "rot_world", "node_count", "elem_count",
        "nodes_per_elem", "mesh_to_world_mat", "world_to_mesh_mat",
    )

    def __init__(
        self,
        coords: np.ndarray,
        connectivity: np.ndarray,
        fields_render: np.ndarray,
        fields_disp: np.ndarray | None = None,
        pos_world: np.ndarray | None = None,
        rot_world: Rotation | None = None,
    ) -> None:
        self.coords = np.ascontiguousarray(coords, dtype=np.float64)
        self.connectivity = np.ascontiguousarray(connectivity, dtype=np.uintp)
        self.fields_render = np.ascontiguousarray(fields_render, dtype=np.float64)
        self.fields_disp = None
        if fields_disp is not None:
            self.fields_disp = np.ascontiguousarray(fields_disp, dtype=np.float64)
        self.node_count = self.coords.shape[0]
        self.elem_count = self.connectivity.shape[0]
        self.nodes_per_elem = self.connectivity.shape[1]
        self.pos_world = np.zeros((3,), dtype=np.float64)
        if pos_world is not None:
            self.pos_world = np.asarray(pos_world, dtype=np.float64)
        self.rot_world = Rotation.identity() if rot_world is None else rot_world
        self.mesh_to_world_mat = np.zeros((4, 4), dtype=np.float64)
        self.world_to_mesh_mat = np.zeros((4, 4), dtype=np.float64)
        self._build_transform_mats()

    def _build_transform_mats(self) -> None:
        self.mesh_to_world_mat.fill(0.0)
        self.mesh_to_world_mat[0:3, 0:3] = self.rot_world.as_matrix()
        self.mesh_to_world_mat[-1, -1] = 1.0
        self.mesh_to_world_mat[0:3, -1] = self.pos_world
        self.world_to_mesh_mat = np.linalg.inv(self.mesh_to_world_mat)

    def set_pos(self, pos_world: np.ndarray) -> None:
        self.pos_world = np.asarray(pos_world, dtype=np.float64)
        self._build_transform_mats()

    def set_rot(self, rot_world: Rotation) -> None:
        self.rot_world = rot_world
        self._build_transform_mats()


from enum import Enum


class EElementType(str, Enum):
    """Finite-element topology identifiers."""

    TRI3 = "tri3"
    TRI6 = "tri6"
    QUAD4 = "quad4"
    QUAD8 = "quad8"
    QUAD9 = "quad9"


class Mesh2D:
    """A planar 2D finite-element mesh representation."""

    __slots__ = ("element_type", "coords", "connectivity", "_displacements")

    def __init__(
        self,
        element_type: EElementType,
        coords: np.ndarray,
        connectivity: np.ndarray,
        displacements: np.ndarray | None = None,
        displacement: np.ndarray | None = None,
    ) -> None:
        self.element_type = element_type
        self.coords = np.ascontiguousarray(coords, dtype=np.float64)
        self.connectivity = np.ascontiguousarray(connectivity, dtype=np.uintp)
        if displacements is None and displacement is not None:
            displacements = displacement
        self._displacements = (
            None
            if displacements is None
            else np.ascontiguousarray(displacements, dtype=np.float64)
        )

    @property
    def displacements(self) -> np.ndarray | None:
        return self._displacements

    @displacements.setter
    def displacements(self, val: np.ndarray | None) -> None:
        self._displacements = (
            None if val is None else np.ascontiguousarray(val, dtype=np.float64)
        )

    @property
    def displacement(self) -> np.ndarray | None:
        return self._displacements

    @displacement.setter
    def displacement(self, val: np.ndarray | None) -> None:
        self._displacements = (
            None if val is None else np.ascontiguousarray(val, dtype=np.float64)
        )


RenderMesh = Mesh
Mesh3D = Mesh

__all__ = ["EElementType", "Mesh", "Mesh2D", "Mesh3D", "RenderMesh"]
