# ==============================================================================
# rasterpy: legacy DIC scalar-field rasterisers
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ==============================================================================
"""Parameters, presets, and paths for Case 1: Standard Brown-Conrady."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from commonparams import OUTPUT_ROOT

CASE_NAME: str = "case1_brownconrady"

OUT_GRIDINT2D_DIR: Path = OUTPUT_ROOT / "case1_gridint2d"
OUT_RILEY_DIR: Path = OUTPUT_ROOT / "case1_riley"
OUT_VERIFY_DIR: Path = OUTPUT_ROOT / "case1_verify_exact"


@dataclass(frozen=True, slots=True)
class BCPreset:
    """Preset configuration for a Brown-Conrady distortion test."""

    name: str
    description: str
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    p1: float = 0.0
    p2: float = 0.0


BC_PRESETS: list[BCPreset] = [
    BCPreset(
        name="light_barrel",
        description="Light barrel radial distortion",
        k1=-0.3,
        k2=0.5,
    ),
    BCPreset(
        name="extreme_barrel",
        description="Extreme barrel radial distortion",
        k1=-2.5,
        k2=8.0,
    ),
    BCPreset(
        name="light_pincushion",
        description="Light pincushion radial distortion",
        k1=0.3,
        k2=-0.5,
    ),
    BCPreset(
        name="extreme_pincushion",
        description="Extreme pincushion radial distortion",
        k1=2.5,
        k2=-5.0,
    ),
    BCPreset(
        name="mixed_asymmetric",
        description="Radial barrel with asymmetric tangential distortion",
        k1=-0.8,
        k2=1.5,
        p1=3.0e-3,
        p2=-3.0e-3,
    ),
]


def get_gridint2d_paths(preset_name: str) -> tuple[Path, Path]:
    """Return (float_path, tiff_path) for GridInt2D render."""
    float_path = OUT_GRIDINT2D_DIR / f"{preset_name}_gridint2d_float.npy"
    tiff_path = OUT_GRIDINT2D_DIR / f"{preset_name}_gridint2d_8bit.tiff"
    return float_path, tiff_path


def get_riley_paths(preset_name: str) -> tuple[Path, Path]:
    """Return (float_path, tiff_path) for Riley render."""
    float_path = OUT_RILEY_DIR / f"{preset_name}_riley_float.npy"
    tiff_path = OUT_RILEY_DIR / f"{preset_name}_riley_8bit.tiff"
    return float_path, tiff_path


def get_verify_paths(preset_name: str) -> tuple[Path, Path]:
    """Return (float_plot_path, tiff_plot_path) for verification."""
    float_plot = OUT_VERIFY_DIR / f"{preset_name}_float_comparison.png"
    tiff_plot = OUT_VERIFY_DIR / f"{preset_name}_8bit_comparison.png"
    return float_plot, tiff_plot
