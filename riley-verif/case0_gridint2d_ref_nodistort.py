# ==============================================================================
# rasterpy: legacy DIC scalar-field rasterisers
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ==============================================================================
"""Render undistorted eggbox reference using GridInt2D (PxInt2D)."""

from __future__ import annotations

import numpy as np
from PIL import Image

from case0params import (
    GRIDINT2D_FLOAT_PATH,
    GRIDINT2D_TIFF_PATH,
    OUT_GRIDINT2D_DIR,
)
from commonparams import (
    BIT_DEPTH,
    FOCAL_LENGTH,
    GRID_CONTRAST,
    GRID_MEAN,
    GRID_PHASE,
    GRID_PITCH,
    GRIDINT2D_GAUSS_ORDER,
    NUM_PIXELS_X,
    NUM_PIXELS_Y,
    PIXEL_SIZE,
    create_planar_mesh_quad4,
)
import rasterpy as rp


def render_gridint2d_eggbox() -> tuple[np.ndarray, np.ndarray]:
    """Render an undistorted eggbox image with GridInt2D.

    Returns
    -------
    image_float : np.ndarray
        Floating-point image with shape (NUM_PIXELS_Y, NUM_PIXELS_X).
    image_uint8 : np.ndarray
        Quantised 8-bit unsigned integer image with the same shape.
    """
    coords, connect = create_planar_mesh_quad4()

    # Planar 2D mesh for PixIntGrid2D
    mesh_2d = rp.Mesh2D(
        rp.EElementType.QUAD4,
        coords[:, :2],
        connect,
    )

    # Orthographic camera positioned over origin
    camera_2d = rp.Camera2D(
        pixels_num=np.array([NUM_PIXELS_X, NUM_PIXELS_Y], dtype=np.int32),
        pixels_size=PIXEL_SIZE,
        roi_cent_world=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        focal_length=FOCAL_LENGTH,
        bits=BIT_DEPTH,
        background=0.5,
    )

    # Periodic eggbox texture and Gauss integration options
    texture = rp.Eggbox(
        mean=GRID_MEAN,
        contrast=GRID_CONTRAST,
        period=(GRID_PITCH, GRID_PITCH),
        phase=GRID_PHASE,
    )
    options = rp.PxInt2DOpts(
        mapping=rp.EPxIntMapping.AFFINE,
        integration=rp.GaussRule(GRIDINT2D_GAUSS_ORDER),
    )

    renderer = rp.PixIntGrid2D(texture=texture, options=options)
    scene = rp.Scene2D(mesh=mesh_2d, camera=camera_2d)

    # Render frame
    result = renderer.render(scene)
    image_float = result.images[0, 0, :, :, 0]

    # Convert to 8-bit unsigned integer
    image_uint8 = rp.pxint2d.quantise_image(image_float, bits=BIT_DEPTH)
    return image_float, image_uint8


def save_renders(image_float: np.ndarray, image_uint8: np.ndarray) -> None:
    """Save raw floating point numpy array and 8-bit TIFF image."""
    OUT_GRIDINT2D_DIR.mkdir(parents=True, exist_ok=True)

    np.save(GRIDINT2D_FLOAT_PATH, image_float)
    print(f"Saved GridInt2D float array to: {GRIDINT2D_FLOAT_PATH}")

    tiff_image = Image.fromarray(image_uint8, mode="L")
    tiff_image.save(GRIDINT2D_TIFF_PATH)
    print(f"Saved GridInt2D 8-bit TIFF to:  {GRIDINT2D_TIFF_PATH}")


def main() -> None:
    """Execute undistorted GridInt2D render and save outputs."""
    print("=" * 70)
    print("Running GridInt2D (PxInt2D) Undistorted Eggbox Reference Render")
    print("=" * 70)

    image_float, image_uint8 = render_gridint2d_eggbox()

    print(
        f"Rendered image shape: {image_float.shape}, dtype: {image_float.dtype}"
    )
    print(
        f"Float range: [{image_float.min():.4f}, {image_float.max():.4f}], "
        f"mean: {image_float.mean():.4f}"
    )
    print(
        f"8-bit range: [{image_uint8.min()}, {image_uint8.max()}], "
        f"mean: {image_uint8.mean():.2f}"
    )

    save_renders(image_float, image_uint8)
    print("GridInt2D render complete.")


if __name__ == "__main__":
    main()
