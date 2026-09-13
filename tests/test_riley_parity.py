# ==============================================================================
# rasterpy: legacy DIC scalar-field rasterisers
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ==============================================================================
"""Verification tests replicating Riley's test_verif.zig distortion suite."""

from __future__ import annotations

import numpy as np
import pytest

from rasterpy.pxint2d.distortion import (
    BidirectionalPolynomial,
    BrownConrady,
    BrownConradyExt,
    BrownConradyExtPolynomial,
    BrownConradyPolynomial,
    EPolynomialOrder,
    PolynomialMap,
)

# Test probe points replicating Riley test_verif_distortion.zig (normalized)
# For 800x500 px, 5.3 um pitch, f=50 mm -> f_px = 5.0e-2 / 5.3e-6 = 9433.96 px
PROBE_POINTS: list[tuple[float, float]] = [
    (0.0, 0.0),  # Center: (400, 250) px
    (-360.0 / 9433.962264150943, -225.0 / 9433.962264150943),  # (40, 25)
    (360.0 / 9433.962264150943, -225.0 / 9433.962264150943),  # (760, 25)
    (360.0 / 9433.962264150943, 225.0 / 9433.962264150943),  # (760, 475)
    (-360.0 / 9433.962264150943, 225.0 / 9433.962264150943),  # (40, 475)
    (-216.0 / 9433.962264150943, 85.0 / 9433.962264150943),  # (184, 335)
    (224.0 / 9433.962264150943, -95.0 / 9433.962264150943),  # (624, 155)
]

# Preset test cases matching Riley's verifconstants.zig
RILEY_VERIF_CASES = [
    (
        "mild_barrel",
        BrownConrady(k1=-5.0e-2, k2=1.0e-2, k3=0.0, p1=0.0, p2=0.0),
    ),
    (
        "mild_pincushion",
        BrownConrady(k1=5.0e-2, k2=-1.0e-2, k3=0.0, p1=0.0, p2=0.0),
    ),
    (
        "strong_barrel",
        BrownConrady(k1=-1.5e-1, k2=3.0e-2, k3=0.0, p1=0.0, p2=0.0),
    ),
    (
        "mixed_asymmetric",
        BrownConrady(
            k1=-8.0e-2, k2=1.5e-2, k3=0.0, p1=1.0e-3, p2=-1.0e-3,
        ),
    ),
    (
        "brown_conrady_ext",
        BrownConradyExt(
            k1=-5.0e-2, k2=1.0e-2, k3=0.0,
            k4=2.0e-2, k5=-5.0e-3, k6=0.0,
            p1=1.0e-3, p2=-1.0e-3,
        ),
    ),
    (
        "polynomial_quadratic",
        PolynomialMap(
            order=EPolynomialOrder.QUADRATIC,
            coeffs_u=(0.0, 0.01, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0),
            coeffs_v=(0.0, 0.0, -0.01, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        ),
    ),
    (
        "polynomial_cubic",
        PolynomialMap(
            order=EPolynomialOrder.CUBIC,
            coeffs_u=(0.0, 0.01, 0.0, 0.0, 0.5, 0.0, 2.0, 0.0, 0.0, 0.0),
            coeffs_v=(0.0, 0.0, -0.01, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0),
        ),
    ),
    (
        "brown_conrady_polynomial",
        BrownConradyPolynomial(
            brown_conrady=BrownConrady(
                k1=-5.0e-2, k2=1.0e-2, k3=0.0, p1=1.0e-3, p2=-1.0e-3,
            ),
            polynomial=BidirectionalPolynomial(
                forward_map=PolynomialMap(
                    order=EPolynomialOrder.QUADRATIC,
                    coeffs_u=(
                        0.0, 0.005, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0,
                    ),
                    coeffs_v=(
                        0.0, 0.0, -0.005, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    ),
                ),
            ),
        ),
    ),
    (
        "brown_conrady_ext_polynomial",
        BrownConradyExtPolynomial(
            brown_conrady_ext=BrownConradyExt(
                k1=-5.0e-2, k2=1.0e-2, k3=0.0,
                k4=2.0e-2, k5=-5.0e-3, k6=0.0,
                p1=1.0e-3, p2=-1.0e-3,
            ),
            polynomial=BidirectionalPolynomial(
                forward_map=PolynomialMap(
                    order=EPolynomialOrder.QUADRATIC,
                    coeffs_u=(
                        0.0, 0.005, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0,
                    ),
                    coeffs_v=(
                        0.0, 0.0, -0.005, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    ),
                ),
            ),
        ),
    ),
]


@pytest.mark.parametrize("case_name, model", RILEY_VERIF_CASES)
def test_riley_verif_distortion_cases_round_trip(
    case_name: str,
    model,
) -> None:
    """Verify round trip accuracy across Riley test_verif presets."""
    pts = np.array(PROBE_POINTS, dtype=np.float64)
    x_in = pts[:, 0]
    y_in = pts[:, 1]

    # Forward mapping
    xd, yd = model.forward(x_in, y_in)

    # Inverse mapping
    x_rec, y_rec = model.inverse(xd, yd)

    # Re-projected forward mapping
    xd_rec, yd_rec = model.forward(x_rec, y_rec)

    # Maximum coordinate recovery error (normalized coordinates)
    err_coord = float(
        np.max(np.hypot(x_rec - x_in, y_rec - y_in))
    )
    # Maximum re-projection error (distorted coordinates)
    err_reproj = float(
        np.max(np.hypot(xd_rec - xd, yd_rec - yd))
    )

    # Target tolerance matching Riley's verif tolerances (< 1e-10 normalized)
    assert err_coord < 1.0e-10, (
        f"{case_name}: coord error {err_coord:.2e} exceeds tolerance"
    )
    assert err_reproj < 1.0e-10, (
        f"{case_name}: reproj error {err_reproj:.2e} exceeds tolerance"
    )
