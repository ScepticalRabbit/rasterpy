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
    s1: float = 0.0
    s2: float = 0.0
    s3: float = 0.0
    s4: float = 0.0
    tau_x: float = 0.0
    tau_y: float = 0.0


BCEXT_PRESETS: list[BCExtPreset] = [
    BCExtPreset(
        name="light_barrel_ext",
        description="Light barrel distortion with rational terms",
        k1=-0.3,
        k2=0.5,
        k4=0.1,
        k5=-0.05,
    ),
    BCExtPreset(
        name="extreme_barrel_ext",
        description="Extreme barrel distortion with rational terms",
        k1=-2.5,
        k2=8.0,
        k4=0.5,
        k5=-0.2,
    ),
    BCExtPreset(
        name="light_pincushion_ext",
        description="Light pincushion distortion with rational terms",
        k1=0.3,
        k2=-0.5,
        k4=-0.1,
        k5=0.05,
    ),
    BCExtPreset(
        name="extreme_pincushion_ext",
        description="Extreme pincushion distortion with rational terms",
        k1=2.5,
        k2=-5.0,
        k4=-0.5,
        k5=0.2,
    ),
    BCExtPreset(
        name="light_prism",
        description="Light thin prism distortion",
        s1=0.05,
        s2=-0.08,
        s3=0.03,
        s4=-0.05,
    ),
    BCExtPreset(
        name="extreme_prism",
        description="Extreme thin prism distortion",
        s1=0.4,
        s2=-0.8,
        s3=0.3,
        s4=-0.5,
    ),
    BCExtPreset(
        name="light_tilt",
        description="Light sensor tilt distortion",
        tau_x=0.05,
        tau_y=-0.04,
    ),
    BCExtPreset(
        name="extreme_tilt",
        description="Extreme sensor tilt distortion",
        tau_x=0.25,
        tau_y=-0.20,
    ),
    BCExtPreset(
        name="combined_extreme_all",
        description="Combined extreme rational, tangential, prism, and tilt",
        k1=-2.5,
        k2=8.0,
        k4=0.5,
        k5=-0.2,
        p1=3.0e-3,
        p2=-3.0e-3,
        s1=0.4,
        s2=-0.8,
        s3=0.3,
        s4=-0.5,
        tau_x=0.25,
        tau_y=-0.20,
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
