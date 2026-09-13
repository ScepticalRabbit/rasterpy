# ==============================================================================
# rasterpy: legacy DIC scalar-field rasterisers
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ==============================================================================
"""Render Brown-Conrady distorted grid references using GridInt2D."""

from __future__ import annotations

import numpy as np
from PIL import Image

from case1params import (
    BC_PRESETS,
    BCPreset,
    OUT_GRIDINT2D_DIR,
    get_gridint2d_paths,
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
from rasterpy.pxint2d.distortion import BrownConrady


def render_preset(preset: BCPreset) -> tuple[np.ndarray, np.ndarray]:
    """Render a single Brown-Conrady preset using GridInt2D."""
    coords, connect = create_planar_mesh_quad4()

    mesh_2d = rp.Mesh2D(
        rp.EElementType.QUAD4,
        coords[:, :2],
        connect,
    )

    distortion = BrownConrady(
        k1=preset.k1,
        k2=preset.k2,
        k3=preset.k3,
        p1=preset.p1,
        p2=preset.p2,
    )

    camera_2d = rp.Camera2D(
        pixels_num=np.array([NUM_PIXELS_X, NUM_PIXELS_Y], dtype=np.int32),
        pixels_size=PIXEL_SIZE,
        roi_cent_world=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        focal_length=FOCAL_LENGTH,
        distortion=distortion,
        bits=BIT_DEPTH,
        background=0.5,
    )

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

    result = renderer.render(scene)
    image_float = result.images[0, 0, :, :, 0]
    image_uint8 = rp.pxint2d.quantise_image(image_float, bits=BIT_DEPTH)

    return image_float, image_uint8


def main() -> None:
    """Execute all Case 1 GridInt2D renders."""
    print("=" * 70)
    print("Running Case 1: GridInt2D Brown-Conrady Distorted Grid Renders")
    print("=" * 70)

    OUT_GRIDINT2D_DIR.mkdir(parents=True, exist_ok=True)

    for preset in BC_PRESETS:
        print(f"\nRendering preset: {preset.name} ({preset.description})...")
        image_float, image_uint8 = render_preset(preset)

        float_path, tiff_path = get_gridint2d_paths(preset.name)
        np.save(float_path, image_float)
        tiff_img = Image.fromarray(image_uint8, mode="L")
        tiff_img.save(tiff_path)

        print(
            f"  Float range: [{image_float.min():.4f}, "
            f"{image_float.max():.4f}], mean: {image_float.mean():.4f}"
        )
        print(f"  Saved float array to: {float_path}")
        print(f"  Saved 8-bit TIFF to:  {tiff_path}")

    print("\nCase 1 GridInt2D renders completed successfully.")


if __name__ == "__main__":
    main()
