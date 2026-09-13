# ==============================================================================
# rasterpy: legacy DIC scalar-field rasterisers
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ==============================================================================
"""Render Brown-Conrady distorted grid references using Riley rasteriser."""

from __future__ import annotations

import numpy as np
from PIL import Image

from case1params import (
    BC_PRESETS,
    BCPreset,
    OUT_RILEY_DIR,
    get_riley_paths,
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


def render_preset(preset: BCPreset) -> tuple[np.ndarray, np.ndarray]:
    """Render a single Brown-Conrady preset using Riley."""
    coords, connect = create_planar_mesh_quad4()

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

    camera = riley.Camera(
        pixels_num=(NUM_PIXELS_X, NUM_PIXELS_Y),
        pixels_size=(PIXEL_SIZE, PIXEL_SIZE),
        pos_world=(0.0, 0.0, FOCAL_LENGTH),
        rot_world=(0.0, 0.0, 0.0),
        roi_cent_world=(0.0, 0.0, 0.0),
        focal_length=FOCAL_LENGTH,
        sub_sample=RILEY_SSAA,
        coord_sys=riley.CameraCoordSys.opengl,
        distortion_model=1,  # brown_conrady
        distortion_k1=preset.k1,
        distortion_k2=preset.k2,
        distortion_k3=preset.k3,
        distortion_p1=preset.p1,
        distortion_p2=preset.p2,
    )

    config = riley.create_raster_config(
        1,
        total_threads=NUM_THREADS,
        save_strategy=riley.SaveStrategy.memory,
    )

    rendered_frames = riley.raster(mesh, camera, config)
    raw_array = rendered_frames[0, 0, 0]

    max_val = float((1 << BIT_DEPTH) - 1)
    # OpenGL raster buffer has row 0 at bottom; flip to top-down image
    # orientation to match standard matrix and TIFF image layout
    raw_flipped = np.flipud(raw_array)
    image_float = np.asarray(raw_flipped, dtype=np.float64) / max_val
    image_uint8 = np.rint(np.clip(raw_flipped, 0.0, max_val)).astype(np.uint8)

    return image_float, image_uint8


def main() -> None:
    """Execute all Case 1 Riley renders."""
    print("=" * 70)
    print("Running Case 1: Riley Brown-Conrady Distorted Grid Renders")
    print("=" * 70)

    OUT_RILEY_DIR.mkdir(parents=True, exist_ok=True)

    for preset in BC_PRESETS:
        print(f"\nRendering preset: {preset.name} ({preset.description})...")
        image_float, image_uint8 = render_preset(preset)

        float_path, tiff_path = get_riley_paths(preset.name)
        np.save(float_path, image_float)
        tiff_img = Image.fromarray(image_uint8, mode="L")
        tiff_img.save(tiff_path)

        print(
            f"  Float range: [{image_float.min():.4f}, "
            f"{image_float.max():.4f}], mean: {image_float.mean():.4f}"
        )
        print(f"  Saved float array to: {float_path}")
        print(f"  Saved 8-bit TIFF to:  {tiff_path}")

    print("\nCase 1 Riley renders completed successfully.")


if __name__ == "__main__":
    main()
