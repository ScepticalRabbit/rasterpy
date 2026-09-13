# ==============================================================================
# rasterpy: legacy DIC scalar-field rasterisers
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ==============================================================================
"""Render undistorted eggbox reference using Riley rasteriser."""

from __future__ import annotations

import numpy as np
from PIL import Image

from case0params import (
    OUT_RILEY_DIR,
    RILEY_FLOAT_PATH,
    RILEY_TIFF_PATH,
)
from commonparams import (
    BIT_DEPTH,
    FOCAL_LENGTH,
    GRID_CONTRAST,
    GRID_MEAN,
    GRID_PHASE,
    GRID_PITCH,
    NUM_PIXELS_X,
    NUM_PIXELS_Y,
    NUM_THREADS,
    PIXEL_SIZE,
    RILEY_SSAA,
    create_planar_mesh_quad4,
)
import riley


def render_riley_eggbox() -> tuple[np.ndarray, np.ndarray]:
    """Render an undistorted eggbox image with Riley.

    Returns
    -------
    image_float : np.ndarray
        Floating-point normalised image in [0, 1] with shape
        (NUM_PIXELS_Y, NUM_PIXELS_X).
    image_uint8 : np.ndarray
        Quantised 8-bit unsigned integer image with the same shape.
    """
    coords, connect = create_planar_mesh_quad4()

    # Function shader configured for eggbox grid
    func_params = riley.FuncShaderParams(
        eggbox_mean=GRID_MEAN,
        eggbox_contrast=GRID_CONTRAST,
        eggbox_pitch=(GRID_PITCH, GRID_PITCH),
        eggbox_phase=GRID_PHASE,
    )

    shader = riley.FunctionShader(
        builtin=riley.FuncShaderBuiltin.eggbox,
        coord_mode=riley.FuncCoordMode.world_reference,
        params=func_params,
        bits=BIT_DEPTH,
        scaling_type=riley.ScaleStrategy.fixed,
        scaling_min=0.0,
        scaling_max=1.0,
    )

    convention = riley.ConnectConvention(
        riley.EElemType.QUAD4,
        riley.EConnectAxis.ROW,
        0,
        riley.ENodeOrder.RILEY,
    )

    mesh = riley.create_mesh(
        convention,
        riley.MeshType.quad4,
        coords,
        connect,
        shader,
    )

    # Perspective camera positioned normal to the mesh at z = focal_length
    camera = riley.Camera(
        pixels_num=(NUM_PIXELS_X, NUM_PIXELS_Y),
        pixels_size=(PIXEL_SIZE, PIXEL_SIZE),
        pos_world=(0.0, 0.0, FOCAL_LENGTH),
        rot_world=(0.0, 0.0, 0.0),
        roi_cent_world=(0.0, 0.0, 0.0),
        focal_length=FOCAL_LENGTH,
        sub_sample=RILEY_SSAA,
        coord_sys=riley.CameraCoordSys.opengl,
    )

    config = riley.create_raster_config(
        1,
        total_threads=NUM_THREADS,
        save_strategy=riley.SaveStrategy.memory,
    )

    # Render frame using in-memory buffer
    rendered_frames = riley.raster(mesh, camera, config)
    raw_array = rendered_frames[0, 0, 0]

    max_val = float((1 << BIT_DEPTH) - 1)
    # OpenGL raster buffer has row 0 at bottom; flip to top-down image
    # orientation to match standard matrix and TIFF image layout
    raw_flipped = np.flipud(raw_array)
    image_float = np.asarray(raw_flipped, dtype=np.float64) / max_val
    image_uint8 = np.rint(np.clip(raw_flipped, 0.0, max_val)).astype(np.uint8)

    return image_float, image_uint8


def save_renders(image_float: np.ndarray, image_uint8: np.ndarray) -> None:
    """Save raw floating point numpy array and 8-bit TIFF image."""
    OUT_RILEY_DIR.mkdir(parents=True, exist_ok=True)

    np.save(RILEY_FLOAT_PATH, image_float)
    print(f"Saved Riley float array to: {RILEY_FLOAT_PATH}")

    tiff_image = Image.fromarray(image_uint8, mode="L")
    tiff_image.save(RILEY_TIFF_PATH)
    print(f"Saved Riley 8-bit TIFF to:  {RILEY_TIFF_PATH}")


def main() -> None:
    """Execute undistorted Riley render and save outputs."""
    print("=" * 70)
    print("Running Riley Undistorted Eggbox Reference Render")
    print("=" * 70)

    image_float, image_uint8 = render_riley_eggbox()

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
    print("Riley render complete.")


if __name__ == "__main__":
    main()
