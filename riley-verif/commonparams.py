# ==============================================================================
# rasterpy: legacy DIC scalar-field rasterisers
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ==============================================================================
"""Common parameters and shared utilities for Riley verification cases."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

# ------------------------------------------------------------------------------
# Image and Camera Configuration
# ------------------------------------------------------------------------------
NUM_PIXELS_X: int = 400
NUM_PIXELS_Y: int = 250
PIXEL_SIZE: float = 1.0
FOCAL_LENGTH: float = 1000.0
BIT_DEPTH: int = 8
NUM_THREADS: int = 4

# ------------------------------------------------------------------------------
# Sampling and Quadrature Configuration
# ------------------------------------------------------------------------------
RILEY_SSAA: int = 32
GRIDINT2D_GAUSS_ORDER: int = 4

# ------------------------------------------------------------------------------
# Eggbox Grid Texture Parameters
# ------------------------------------------------------------------------------
GRID_PITCH: float = 5.0
GRID_MEAN: float = 0.5
GRID_CONTRAST: float = 0.4
GRID_PHASE: tuple[float, float] = (0.0, 0.0)

# Tolerance for floating-point difference between Gauss 4 and 32x32 SSAA box
TOLERANCE_FLOAT: float = 1.0e-4

# ------------------------------------------------------------------------------
# Directory Paths
# ------------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
OUTPUT_ROOT: Path = PROJECT_ROOT / "out"


def create_planar_mesh_quad4(
    margin: float = 150.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a planar Quad4 mesh enclosing the camera field of view.

    Parameters
    ----------
    margin : float, optional
        Extra extent beyond camera view limits in world units (pixels).
        Default is 150.0.

    Returns
    -------
    coords : np.ndarray
        Array of node coordinates of shape (4, 3) and dtype float64.
    connect : np.ndarray
        Array of element connectivity of shape (1, 4) and dtype int64.
    """
    half_width: float = 0.5 * NUM_PIXELS_X * PIXEL_SIZE + margin
    half_height: float = 0.5 * NUM_PIXELS_Y * PIXEL_SIZE + margin

    coords = np.array(
        [
            [-half_width, -half_height, 0.0],
            [half_width, -half_height, 0.0],
            [half_width, half_height, 0.0],
            [-half_width, half_height, 0.0],
        ],
        dtype=np.float64,
    )
    connect = np.array([[0, 1, 2, 3]], dtype=np.int64)
    return coords, connect


def plot_comparison(
    img_gridint2d: np.ndarray,
    img_riley: np.ndarray,
    diff: np.ndarray,
    title_prefix: str,
    output_path: Path,
    is_float: bool = True,
) -> None:
    """Generate and save a 3-panel comparison figure with clear spacing."""
    max_diff: float = float(np.max(np.abs(diff)))
    mean_diff: float = float(np.mean(np.abs(diff)))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    if is_float:
        vmin, vmax = 0.0, 1.0
        diff_cmap = "inferno"
        diff_label = "Abs Diff (Float)"
        title = (
            f"{title_prefix} Comparison (Float: [0, 1])\n"
            f"Max Diff = {max_diff:.3e}, Mean Diff = {mean_diff:.3e}"
        )
    else:
        vmin, vmax = 0, 255
        diff_cmap = "magma"
        diff_label = "Abs Diff (Grey Level)"
        title = (
            f"{title_prefix} Comparison (8-bit UInt8)\n"
            f"Max Diff = {int(round(max_diff))} GL, "
            f"Mean Diff = {mean_diff:.3e} GL"
        )

    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.98)

    im0 = axes[0].imshow(
        img_gridint2d, cmap="gray", vmin=vmin, vmax=vmax, origin="lower"
    )
    axes[0].set_title("GridInt2D (PxInt2D)", fontsize=11, pad=8)
    axes[0].set_xlabel("Pixel X", fontsize=10)
    axes[0].set_ylabel("Pixel Y", fontsize=10)
    divider0 = make_axes_locatable(axes[0])
    cax0 = divider0.append_axes("right", size="5%", pad=0.10)
    fig.colorbar(im0, cax=cax0)

    im1 = axes[1].imshow(
        img_riley, cmap="gray", vmin=vmin, vmax=vmax, origin="lower"
    )
    axes[1].set_title("Riley Reference", fontsize=11, pad=8)
    axes[1].set_xlabel("Pixel X", fontsize=10)
    axes[1].set_ylabel("Pixel Y", fontsize=10)
    divider1 = make_axes_locatable(axes[1])
    cax1 = divider1.append_axes("right", size="5%", pad=0.10)
    fig.colorbar(im1, cax=cax1)

    im2 = axes[2].imshow(diff, cmap=diff_cmap, origin="lower")
    axes[2].set_title("Absolute Difference", fontsize=11, pad=8)
    axes[2].set_xlabel("Pixel X", fontsize=10)
    axes[2].set_ylabel("Pixel Y", fontsize=10)
    divider2 = make_axes_locatable(axes[2])
    cax2 = divider2.append_axes("right", size="5%", pad=0.10)
    cbar2 = fig.colorbar(im2, cax=cax2)
    cbar2.set_label(diff_label, fontsize=10, labelpad=8)

    fig.subplots_adjust(
        left=0.06, right=0.94, bottom=0.12, top=0.85, wspace=0.35
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
