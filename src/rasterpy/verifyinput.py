# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Cheap, non-rendering validation helpers."""

import numpy as np

from pyvale.dataio import SimData, check_mesh_convention

from .errors import ValidationIssue
from .mesh import EElementType

_NODES_PER_ELEMENT = {
    EElementType.TRI3: 3,
    EElementType.TRI6: 6,
    EElementType.QUAD4: 4,
    EElementType.QUAD8: 8,
    EElementType.QUAD9: 9,
}


def mesh_convention_issues(
    coords: np.ndarray,
    connectivity: np.ndarray,
    path: str,
) -> tuple[ValidationIssue, ...]:
    """Return shared-convention issues for a render surface mesh.

    Two-dimensional coordinates are padded onto the XY plane before invoking
    the common DataIO checker. This keeps planar renderers aligned with the
    same counter-clockwise, right-handed convention as 3D renderers.
    """
    coords_array = np.asarray(coords)
    connectivity_array = np.asarray(connectivity)
    if coords_array.ndim != 2 or coords_array.shape[1] not in (2, 3):
        return ()
    if connectivity_array.ndim != 2 or connectivity_array.size == 0:
        return ()

    if coords_array.shape[1] == 2:
        coords_array = np.pad(coords_array, ((0, 0), (0, 1)))

    try:
        report = check_mesh_convention(
            SimData(
                coords=coords_array,
                connect={"connect1": connectivity_array},
            )
        )
    except (IndexError, NotImplementedError, TypeError, ValueError) as error:
        return (
            ValidationIssue(
                path + ".connectivity",
                "CONVENTION",
                f"Could not check the shared mesh convention: {error}",
            ),
        )

    if hasattr(report, "failed_checks"):
        if not report.failed_checks:
            return ()
        failed = "; ".join(report.failed_checks)
    elif hasattr(report, "passed"):
        if report.passed:
            return ()
        failed = getattr(report, "message", str(report))
    elif hasattr(report, "items"):
        if not report:
            return ()
        failed = "; ".join(
            f"{table}: {', '.join(code.value for code in codes)}"
            for table, codes in report.items()
        )
    else:
        if not report:
            return ()
        failed = str(report)

    return (
        ValidationIssue(
            path + ".connectivity",
            "CONVENTION",
            "Mesh must follow the shared Riley/VTK convention: " + str(failed),
        ),
    )

__all__ = ["mesh_convention_issues"]
