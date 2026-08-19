# ============================================================================
# rasterpy: legacy DIC scalar-field rasterisers
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Render the bundled MOOSE Exodus case through the NumPy rasteriser.

This historical example requires ``pip install rasterpy[examples]``. It loads
an Exodus result but does not run MOOSE.
"""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import pyvale.mooseherder as mooseherder
import pyvale.sensorsim as sensorsim
import rasterpy
from rasterpy import data


def main() -> None:
    """Load the bundled Exodus file and save its first scalar raster image."""
    sim_data = mooseherder.ExodusLoader(
        data.render_mechanical_3d_path(),
    ).load_all_sim_data()
    pyvale_mesh = sensorsim.create_render_mesh(
        sim_data,
        ("disp_x",),
        sim_spat_dim=sensorsim.EDim.THREED,
        field_disp_keys=("disp_x", "disp_y", "disp_z"),
    )
    mesh = rasterpy.Mesh(
        pyvale_mesh.coords,
        pyvale_mesh.connectivity,
        pyvale_mesh.fields_render,
        pyvale_mesh.fields_disp,
    )
    camera = rasterpy.Camera(
        pixels_num=np.array((512, 512)),
        pixels_size=np.array((0.00345, 0.00345)),
        pos_world=np.array((0.0, 0.0, 500.0)),
        rot_world=Rotation.identity(),
        roi_cent_world=np.zeros((3,)),
        focal_length=15.0,
        sub_samp=1,
    )
    image = rasterpy.RasterNumpy(
        rasterpy.RasterOpts(parallel=None),
    ).render(rasterpy.Scene([camera], [mesh]))
    output_path = Path.cwd() / "rasterpy-case26"
    rasterpy.ImageTools.scale_digitise_save(
        output_path, image, rasterpy.EImageType.TIFF, bits=8,
    )
    print(f"Saved {output_path.with_suffix('.tiff')}")


if __name__ == "__main__":
    main()
