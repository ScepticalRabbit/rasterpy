# ==============================================================================
# rasterpy: legacy DIC scalar-field rasterisers
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ==============================================================================
"""Verify exact match between GridInt2D and Riley undistorted renders."""

from __future__ import annotations

import numpy as np
from PIL import Image

from case0params import (
    GRIDINT2D_FLOAT_PATH,
    GRIDINT2D_TIFF_PATH,
    OUT_VERIFY_DIR,
    RILEY_FLOAT_PATH,
    RILEY_TIFF_PATH,
    VERIFY_FLOAT_PLOT_PATH,
    VERIFY_TIFF_PLOT_PATH,
)
from commonparams import (
    TOLERANCE_FLOAT,
    plot_comparison,
)


def verify_float_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and compare floating point output arrays."""
    if not GRIDINT2D_FLOAT_PATH.exists():
        raise FileNotFoundError(
            f"GridInt2D float array not found: {GRIDINT2D_FLOAT_PATH}"
        )
    if not RILEY_FLOAT_PATH.exists():
        raise FileNotFoundError(
            f"Riley float array not found: {RILEY_FLOAT_PATH}"
        )

    img_gridint2d = np.load(GRIDINT2D_FLOAT_PATH)
    img_riley = np.load(RILEY_FLOAT_PATH)
    diff = np.abs(img_gridint2d - img_riley)

    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))

    print(
        f"Float comparison: max diff = {max_diff:.6e}, "
        f"mean diff = {mean_diff:.6e}"
    )
    if max_diff > TOLERANCE_FLOAT:
        print(
            f"WARNING: Max float diff {max_diff:.6e} exceeds tolerance "
            f"{TOLERANCE_FLOAT:.1e}"
        )
    else:
        print(
            f"PASS: Float difference is within tolerance {TOLERANCE_FLOAT:.1e}"
        )

    return img_gridint2d, img_riley, diff


def verify_tiff_images() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and compare 8-bit TIFF images."""
    if not GRIDINT2D_TIFF_PATH.exists():
        raise FileNotFoundError(
            f"GridInt2D TIFF not found: {GRIDINT2D_TIFF_PATH}"
        )
    if not RILEY_TIFF_PATH.exists():
        raise FileNotFoundError(f"Riley TIFF not found: {RILEY_TIFF_PATH}")

    img_gridint2d = np.array(Image.open(GRIDINT2D_TIFF_PATH), dtype=np.int32)
    img_riley = np.array(Image.open(RILEY_TIFF_PATH), dtype=np.int32)
    diff = np.abs(img_gridint2d - img_riley)

    max_diff = int(np.max(diff))
    mean_diff = float(np.mean(diff))
    diff_count = int(np.count_nonzero(diff))

    print(
        f"8-bit TIFF comparison: max diff = {max_diff} GL, "
        f"mean diff = {mean_diff:.6f} GL, "
        f"differing pixels = {diff_count} / {diff.size}"
    )

    if max_diff == 0:
        print("PASS: Exact bit-for-bit match (0 GL difference)!")
    elif max_diff <= 1:
        print("PASS: Exact within 1 GL rounding tolerance.")
    else:
        print(f"WARNING: Max 8-bit diff is {max_diff} GL.")

    return img_gridint2d, img_riley, diff


def main() -> None:
    """Run verification and create comparison figures."""
    print("=" * 70)
    print("Verifying GridInt2D vs Riley Undistorted Eggbox Reference Renders")
    print("=" * 70)

    OUT_VERIFY_DIR.mkdir(parents=True, exist_ok=True)

    img_g_flt, img_r_flt, diff_flt = verify_float_arrays()
    plot_comparison(
        img_g_flt,
        img_r_flt,
        diff_flt,
        title_prefix="Case 0 (Undistorted)",
        output_path=VERIFY_FLOAT_PLOT_PATH,
        is_float=True,
    )
    print(f"Saved float comparison plot: {VERIFY_FLOAT_PLOT_PATH}")

    img_g_u8, img_r_u8, diff_u8 = verify_tiff_images()
    plot_comparison(
        img_g_u8,
        img_r_u8,
        diff_u8,
        title_prefix="Case 0 (Undistorted)",
        output_path=VERIFY_TIFF_PLOT_PATH,
        is_float=False,
    )
    print(f"Saved 8-bit comparison plot:  {VERIFY_TIFF_PLOT_PATH}")
    print("Verification complete.")


if __name__ == "__main__":
    main()
