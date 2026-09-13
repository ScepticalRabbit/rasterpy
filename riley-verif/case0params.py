# ==============================================================================
# rasterpy: legacy DIC scalar-field rasterisers
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ==============================================================================
"""Parameters and paths specific to Case 0 (Undistorted Reference)."""

from __future__ import annotations

from pathlib import Path

from commonparams import OUTPUT_ROOT

CASE_NAME: str = "case0_nodistort"

OUT_GRIDINT2D_DIR: Path = OUTPUT_ROOT / "case0_gridint2d"
OUT_RILEY_DIR: Path = OUTPUT_ROOT / "case0_riley"
OUT_VERIFY_DIR: Path = OUTPUT_ROOT / "case0_verify_exact"

GRIDINT2D_FLOAT_PATH: Path = (
    OUT_GRIDINT2D_DIR / f"{CASE_NAME}_gridint2d_float.npy"
)
GRIDINT2D_TIFF_PATH: Path = (
    OUT_GRIDINT2D_DIR / f"{CASE_NAME}_gridint2d_8bit.tiff"
)

RILEY_FLOAT_PATH: Path = OUT_RILEY_DIR / f"{CASE_NAME}_riley_float.npy"
RILEY_TIFF_PATH: Path = OUT_RILEY_DIR / f"{CASE_NAME}_riley_8bit.tiff"

VERIFY_FLOAT_PLOT_PATH: Path = (
    OUT_VERIFY_DIR / f"{CASE_NAME}_float_comparison.png"
)
VERIFY_TIFF_PLOT_PATH: Path = (
    OUT_VERIFY_DIR / f"{CASE_NAME}_8bit_comparison.png"
)
