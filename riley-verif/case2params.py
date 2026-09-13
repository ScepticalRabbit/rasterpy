# ==============================================================================
# rasterpy: legacy DIC scalar-field rasterisers
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ==============================================================================
"""Parameters, presets, and paths for Case 2: Extended Brown-Conrady."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from commonparams import OUTPUT_ROOT

CASE_NAME: str = "case2_bcext"

OUT_GRIDINT2D_DIR: Path = OUTPUT_ROOT / "case2_gridint2d"
OUT_RILEY_DIR: Path = OUTPUT_ROOT / "case2_riley"
OUT_VERIFY_DIR: Path = OUTPUT_ROOT / "case2_verify_exact"


@dataclass(frozen=True, slots=True)
class BCExtPreset:
    """Preset configuration for an Extended Brown-Conrady distortion test."""

    name: str
    description: str
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    k4: float = 0.0
    k5: float = 0.0
    k6: float = 0.0
    p1: float = 0.0
    p2: float = 0.0


BCEXT_PRESETS: list[BCExtPreset] = [
    BCExtPreset(
        name="mild_ext",
        description="Extended rational model with numerator & denominator",
        k1=-0.05,
        k2=0.01,
        k3=0.0,
        k4=0.02,
        k5=-0.005,
        k6=0.0,
    ),
    BCExtPreset(
        name="asymmetric_ext",
        description="Extended rational model with tangential distortion",
        k1=-0.05,
        k2=0.01,
        k3=0.0,
        k4=0.02,
        k5=-0.005,
        k6=0.0,
        p1=1.0e-3,
        p2=-1.0e-3,
    ),
    BCExtPreset(
        name="rational_higher_order",
        description="Full 8-parameter rational model with k1..k6 and p1..p2",
        k1=-0.08,
        k2=0.02,
        k3=-0.001,
        k4=0.03,
        k5=-0.01,
        k6=0.001,
        p1=5.0e-4,
        p2=-5.0e-4,
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
