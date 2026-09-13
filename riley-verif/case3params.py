# ==============================================================================
# rasterpy: legacy DIC scalar-field rasterisers
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ==============================================================================
"""Parameters, presets, and paths for Case 3: Standalone Polynomials."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from commonparams import OUTPUT_ROOT
from rasterpy.pxint2d.distortion import EPolynomialOrder

CASE_NAME: str = "case3_polynomial"

OUT_GRIDINT2D_DIR: Path = OUTPUT_ROOT / "case3_gridint2d"
OUT_RILEY_DIR: Path = OUTPUT_ROOT / "case3_riley"
OUT_VERIFY_DIR: Path = OUTPUT_ROOT / "case3_verify_exact"


@dataclass(frozen=True, slots=True)
class PolyPreset:
    """Preset configuration for a polynomial distortion test."""

    name: str
    description: str
    order: EPolynomialOrder
    coeffs_u: tuple[float, ...]
    coeffs_v: tuple[float, ...]


POLY_PRESETS: list[PolyPreset] = [
    PolyPreset(
        name="poly_linear",
        description="Linear polynomial shear and skew distortion",
        order=EPolynomialOrder.LINEAR,
        coeffs_u=(0.0, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        coeffs_v=(0.0, -0.01, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ),
    PolyPreset(
        name="poly_quadratic",
        description="Quadratic polynomial distortion matching Riley preset",
        order=EPolynomialOrder.QUADRATIC,
        coeffs_u=(0.0, 0.01, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0),
        coeffs_v=(0.0, 0.0, -0.01, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ),
    PolyPreset(
        name="poly_cubic",
        description="Cubic polynomial distortion matching Riley preset",
        order=EPolynomialOrder.CUBIC,
        coeffs_u=(0.0, 0.01, 0.0, 0.0, 0.5, 0.0, 2.0, 0.0, 0.0, 0.0),
        coeffs_v=(0.0, 0.0, -0.01, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0),
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
