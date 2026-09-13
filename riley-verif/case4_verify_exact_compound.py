# ==============================================================================
# rasterpy: legacy DIC scalar-field rasterisers
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ==============================================================================
"""Verify exact match between GridInt2D and Riley Case 4 renders."""

from __future__ import annotations

import numpy as np
from PIL import Image

from case4params import (
    COMPOUND_PRESETS,
    OUT_VERIFY_DIR,
    get_gridint2d_paths,
    get_riley_paths,
    get_verify_paths,
)
from commonparams import (
    TOLERANCE_FLOAT,
    plot_comparison,
)


def verify_preset(preset_name: str) -> tuple[float, int]:
    """Compare GridInt2D and Riley renders for a given preset."""
    g_float_path, g_tiff_path = get_gridint2d_paths(preset_name)
    r_float_path, r_tiff_path = get_riley_paths(preset_name)
    plot_float_path, plot_tiff_path = get_verify_paths(preset_name)

    if not g_float_path.exists() or not r_float_path.exists():
        raise FileNotFoundError(
            f"Missing float output files for preset {preset_name}"
        )
    if not g_tiff_path.exists() or not r_tiff_path.exists():
        raise FileNotFoundError(
            f"Missing TIFF output files for preset {preset_name}"
        )

    # 1. Floating-point comparison
    img_g_flt = np.load(g_float_path)
    img_r_flt = np.load(r_float_path)
    diff_flt = np.abs(img_g_flt - img_r_flt)
    max_diff_flt = float(np.max(diff_flt))
    mean_diff_flt = float(np.mean(diff_flt))

    plot_comparison(
        img_g_flt,
        img_r_flt,
        diff_flt,
        title_prefix=f"Case 4 ({preset_name})",
        output_path=plot_float_path,
        is_float=True,
    )

    # 2. 8-bit integer comparison
    img_g_u8 = np.array(Image.open(g_tiff_path), dtype=np.int32)
    img_r_u8 = np.array(Image.open(r_tiff_path), dtype=np.int32)
    diff_u8 = np.abs(img_g_u8 - img_r_u8)
    max_diff_u8 = int(np.max(diff_u8))
    mean_diff_u8 = float(np.mean(diff_u8))

    plot_comparison(
        img_g_u8,
        img_r_u8,
        diff_u8,
        title_prefix=f"Case 4 ({preset_name})",
        output_path=plot_tiff_path,
        is_float=False,
    )

    print(f"\n--- Preset: {preset_name} ---")
    print(
        f"  Float diff: max = {max_diff_flt:.6e}, mean = {mean_diff_flt:.6e}"
    )
    print(
        f"  8-bit diff: max = {max_diff_u8} GL, mean = {mean_diff_u8:.6f} GL"
    )

    if max_diff_flt <= TOLERANCE_FLOAT:
        print(f"  [PASS] Float diff within tolerance ({TOLERANCE_FLOAT:.1e})")
    else:
        print(
            f"  [WARN] Float diff {max_diff_flt:.6e} > {TOLERANCE_FLOAT:.1e}"
        )

    if max_diff_u8 <= 1:
        print(f"  [PASS] 8-bit difference is {max_diff_u8} GL (within 1 GL)")
    else:
        print(f"  [WARN] 8-bit diff is {max_diff_u8} GL")

    return max_diff_flt, max_diff_u8


def main() -> None:
    """Run verification across all Case 4 presets."""
    print("=" * 70)
    print("Verifying Case 4: GridInt2D vs Riley Compound Distortion Renders")
    print("=" * 70)

    OUT_VERIFY_DIR.mkdir(parents=True, exist_ok=True)

    summary: list[tuple[str, float, int]] = []
    for preset in COMPOUND_PRESETS:
        max_flt, max_u8 = verify_preset(preset.name)
        summary.append((preset.name, max_flt, max_u8))

    print("\n" + "=" * 70)
    print("Case 4 Verification Summary")
    print("=" * 70)
    for name, max_flt, max_u8 in summary:
        is_pass = max_flt <= TOLERANCE_FLOAT and max_u8 <= 1
        status = "PASS" if is_pass else "WARN"
        print(
            f"  {name:<24}: max float diff = {max_flt:.3e}, "
            f"max 8-bit diff = {max_u8} GL [{status}]"
        )


if __name__ == "__main__":
    main()
