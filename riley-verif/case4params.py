# ==============================================================================
# rasterpy: legacy DIC scalar-field rasterisers
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ==============================================================================
"""Parameters, presets, and paths for Case 4: Compound Distortion Models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from commonparams import OUTPUT_ROOT
from rasterpy.pxint2d.distortion import EPolynomialOrder

CASE_NAME: str = "case4_compound"

OUT_GRIDINT2D_DIR: Path = OUTPUT_ROOT / "case4_gridint2d"
OUT_RILEY_DIR: Path = OUTPUT_ROOT / "case4_riley"
OUT_VERIFY_DIR: Path = OUTPUT_ROOT / "case4_verify_exact"


@dataclass(frozen=True, slots=True)
class CompoundPreset:
    """Preset configuration for a compound distortion test."""

    name: str
    description: str
    is_ext: bool
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
    poly_order: EPolynomialOrder = EPolynomialOrder.QUADRATIC
    poly_coeffs_u: tuple[float, ...] = (0.0,) * 10
    poly_coeffs_v: tuple[float, ...] = (0.0,) * 10


COMPOUND_PRESETS: list[CompoundPreset] = [
    CompoundPreset(
        name="bc_poly_quadratic",
        description="Brown-Conrady cascaded with quadratic polynomial",
        is_ext=False,
        k1=-0.3,
        k2=0.5,
        p1=2.0e-3,
        p2=-2.0e-3,
        poly_order=EPolynomialOrder.QUADRATIC,
        poly_coeffs_u=(0.0, 0.01, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0),
        poly_coeffs_v=(0.0, 0.0, -0.01, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ),
    CompoundPreset(
        name="bcext_poly_quadratic",
        description="Extended Brown-Conrady cascaded with quadratic polynomial",
        is_ext=True,
        k1=-0.3,
        k2=0.5,
        k4=0.1,
        k5=-0.05,
        p1=2.0e-3,
        p2=-2.0e-3,
        poly_order=EPolynomialOrder.QUADRATIC,
        poly_coeffs_u=(0.0, 0.01, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0),
        poly_coeffs_v=(0.0, 0.0, -0.01, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ),
    CompoundPreset(
        name="bc_poly_cubic",
        description="Brown-Conrady cascaded with cubic polynomial",
        is_ext=False,
        k1=-0.3,
        k2=0.5,
        p1=2.0e-3,
        p2=-2.0e-3,
        poly_order=EPolynomialOrder.CUBIC,
        poly_coeffs_u=(0.0, 0.01, 0.0, 0.0, 0.5, 0.0, 2.0, 0.0, 0.0, 0.0),
        poly_coeffs_v=(0.0, 0.0, -0.01, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0),
    ),
    CompoundPreset(
        name="bcext_poly_extreme",
        description=(
            "Extended Brown-Conrady (with prism & tilt) and cubic polynomial"
        ),
        is_ext=True,
        k1=-1.5,
        k2=4.0,
        k4=0.3,
        k5=-0.1,
        p1=2.0e-3,
        p2=-2.0e-3,
        s1=0.2,
        s2=-0.4,
        s3=0.15,
        s4=-0.25,
        tau_x=0.15,
        tau_y=-0.12,
        poly_order=EPolynomialOrder.CUBIC,
        poly_coeffs_u=(0.0, 0.01, 0.0, 0.0, 0.5, 0.0, 2.0, 0.0, 0.0, 0.0),
        poly_coeffs_v=(0.0, 0.0, -0.01, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0),
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
