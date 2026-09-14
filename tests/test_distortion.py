# ==============================================================================
# rasterpy: legacy DIC scalar-field rasterisers
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ==============================================================================
"""Tests for camera distortion models, OpenCV/NumPy oracles, and parity."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.polynomial.polynomial import polyder, polyval2d
import pytest

import rasterpy as rp
from rasterpy.pxint2d.distortion import (
    POLY_POWERS_U,
    POLY_POWERS_V,
    BidirectionalPolynomial,
    BrownConrady,
    BrownConradyExt,
    BrownConradyExtPolynomial,
    BrownConradyPolynomial,
    EPolynomialOrder,
    PolynomialMap,
)


def _eval_cv2_project_points(
    points_2d: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray:
    """Project normalized 2D points using OpenCV cv2.projectPoints as oracle."""
    num_pts = len(points_2d)
    pts_3d = np.zeros((num_pts, 3), dtype=np.float64)
    pts_3d[:, :2] = points_2d
    pts_3d[:, 2] = 1.0

    camera_matrix = np.eye(3, dtype=np.float64)
    rvec = np.zeros(3, dtype=np.float64)
    tvec = np.zeros(3, dtype=np.float64)

    projected, _ = cv2.projectPoints(
        pts_3d, rvec, tvec, camera_matrix, dist_coeffs,
    )
    return projected.reshape(num_pts, 2)


def _eval_numpy_poly2d_oracle(
    x_coord: np.ndarray,
    y_coord: np.ndarray,
    order: EPolynomialOrder,
    coeffs_u: tuple[float, ...],
    coeffs_v: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate 2D polynomial displacement and derivatives using NumPy."""
    mat_u = np.zeros((4, 4), dtype=np.float64)
    mat_v = np.zeros((4, 4), dtype=np.float64)

    term_count = order.term_count()
    for idx in range(term_count):
        cu = coeffs_u[idx] if idx < len(coeffs_u) else 0.0
        cv = coeffs_v[idx] if idx < len(coeffs_v) else 0.0
        pu = POLY_POWERS_U[idx]
        pv = POLY_POWERS_V[idx]
        mat_u[pu, pv] = cu
        mat_v[pu, pv] = cv

    du = polyval2d(x_coord, y_coord, mat_u)
    dv = polyval2d(x_coord, y_coord, mat_v)

    mat_u_dx = polyder(mat_u, m=1, axis=0)
    mat_u_dy = polyder(mat_u, m=1, axis=1)
    mat_v_dx = polyder(mat_v, m=1, axis=0)
    mat_v_dy = polyder(mat_v, m=1, axis=1)

    ddu_dx = polyval2d(x_coord, y_coord, mat_u_dx)
    ddu_dy = polyval2d(x_coord, y_coord, mat_u_dy)
    ddv_dx = polyval2d(x_coord, y_coord, mat_v_dx)
    ddv_dy = polyval2d(x_coord, y_coord, mat_v_dy)

    return (
        x_coord + du,
        y_coord + dv,
        np.column_stack((1.0 + ddu_dx, ddu_dy)),
        np.column_stack((ddv_dx, 1.0 + ddv_dy)),
    )


# ------------------------------------------------------------------------------
# Test Cases for Brown-Conrady vs OpenCV Oracle
# ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "k1, k2, k3, p1, p2",
    [
        (0.0, 0.0, 0.0, 0.0, 0.0),  # Zero distortion
        (0.12, 0.0, 0.0, 0.0, 0.0),  # Pure radial k1
        (0.08, -0.02, 0.005, 0.0, 0.0),  # Radial k1, k2, k3
        (0.0, 0.0, 0.0, 0.003, -0.002),  # Pure tangential p1, p2
        (-0.05, 0.01, -0.002, 0.001, 0.002),  # Combined radial + tangential
        (0.25, -0.08, 0.015, -0.004, 0.003),  # High barrel distortion
    ],
)
def test_brown_conrady_matches_opencv_oracle(
    k1: float,
    k2: float,
    k3: float,
    p1: float,
    p2: float,
) -> None:
    """Verify Brown-Conrady forward evaluation against OpenCV projectPoints."""
    model = BrownConrady(k1=k1, k2=k2, k3=k3, p1=p1, p2=p2)

    # Grid of test points in normalized image plane [-0.5, 0.5]
    grid_x, grid_y = np.meshgrid(
        np.linspace(-0.5, 0.5, 9),
        np.linspace(-0.4, 0.4, 7),
    )
    x_flat = grid_x.ravel()
    y_flat = grid_y.ravel()

    actual_xd, actual_yd = model.forward(x_flat, y_flat)

    # OpenCV distCoeffs: (k1, k2, p1, p2, k3)
    dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float64)
    pts_2d = np.column_stack((x_flat, y_flat))
    expected = _eval_cv2_project_points(pts_2d, dist_coeffs)

    np.testing.assert_allclose(
        actual_xd, expected[:, 0], atol=1.0e-15, rtol=1.0e-14,
    )
    np.testing.assert_allclose(
        actual_yd, expected[:, 1], atol=1.0e-15, rtol=1.0e-14,
    )


@pytest.mark.parametrize(
    "k1, k2, k3, k4, k5, k6, p1, p2",
    [
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.1, -0.03, 0.005, 0.02, 0.01, -0.002, 0.0, 0.0),
        (0.08, -0.02, 0.004, 0.01, -0.005, 0.001, 0.002, -0.003),
    ],
)
def test_brown_conrady_ext_matches_opencv_oracle(
    k1: float,
    k2: float,
    k3: float,
    k4: float,
    k5: float,
    k6: float,
    p1: float,
    p2: float,
) -> None:
    """Verify Brown-Conrady Extended against OpenCV rational model."""
    model = BrownConradyExt(
        k1=k1, k2=k2, k3=k3, k4=k4, k5=k5, k6=k6, p1=p1, p2=p2,
    )

    grid_x, grid_y = np.meshgrid(
        np.linspace(-0.45, 0.45, 7),
        np.linspace(-0.35, 0.35, 7),
    )
    x_flat = grid_x.ravel()
    y_flat = grid_y.ravel()

    actual_xd, actual_yd = model.forward(x_flat, y_flat)

    # OpenCV rational model: (k1, k2, p1, p2, k3, k4, k5, k6)
    dist_coeffs = np.array(
        [k1, k2, p1, p2, k3, k4, k5, k6], dtype=np.float64,
    )
    pts_2d = np.column_stack((x_flat, y_flat))
    expected = _eval_cv2_project_points(pts_2d, dist_coeffs)

    np.testing.assert_allclose(
        actual_xd, expected[:, 0], atol=1.0e-15, rtol=1.0e-14,
    )
    np.testing.assert_allclose(
        actual_yd, expected[:, 1], atol=1.0e-15, rtol=1.0e-14,
    )


@pytest.mark.parametrize(
    "s1, s2, s3, s4, tau_x, tau_y",
    [
        (0.001, -0.0005, 0.0008, -0.0002, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.02, -0.015),
        (0.0015, -0.0008, 0.001, -0.0004, 0.01, -0.01),
    ],
)
def test_brown_conrady_ext_prism_and_tilt_opencv(
    s1: float,
    s2: float,
    s3: float,
    s4: float,
    tau_x: float,
    tau_y: float,
) -> None:
    """Verify thin prism and sensor tilt against OpenCV 14-parameter oracle."""
    k1, k2, p1, p2 = -0.05, 0.01, 0.001, -0.001
    k4, k5 = 0.02, -0.005
    model = BrownConradyExt(
        k1=k1, k2=k2, p1=p1, p2=p2, k4=k4, k5=k5,
        s1=s1, s2=s2, s3=s3, s4=s4,
        tau_x=tau_x, tau_y=tau_y,
    )

    grid_x, grid_y = np.meshgrid(
        np.linspace(-0.25, 0.25, 7),
        np.linspace(-0.2, 0.2, 7),
    )
    x_flat = grid_x.ravel()
    y_flat = grid_y.ravel()

    actual_xd, actual_yd = model.forward(x_flat, y_flat)

    # OpenCV 14-parameter vector
    dist_coeffs = np.array(
        [
            k1, k2, p1, p2, 0.0, k4, k5, 0.0,
            s1, s2, s3, s4, tau_x, tau_y,
        ],
        dtype=np.float64,
    )
    pts_2d = np.column_stack((x_flat, y_flat))
    expected = _eval_cv2_project_points(pts_2d, dist_coeffs)

    np.testing.assert_allclose(
        actual_xd, expected[:, 0], atol=1.0e-15, rtol=1.0e-14,
    )
    np.testing.assert_allclose(
        actual_yd, expected[:, 1], atol=1.0e-15, rtol=1.0e-14,
    )


# ------------------------------------------------------------------------------
# Test Cases for Polynomial Models vs NumPy Oracle
# ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "order, coeffs_u, coeffs_v",
    [
        (
            EPolynomialOrder.LINEAR,
            (0.01, 0.02, -0.015),
            (-0.005, 0.012, 0.03),
        ),
        (
            EPolynomialOrder.QUADRATIC,
            (0.005, 0.01, -0.02, 0.015, -0.01, 0.008),
            (-0.008, 0.015, 0.005, -0.02, 0.012, -0.015),
        ),
        (
            EPolynomialOrder.CUBIC,
            (
                0.002, 0.01, -0.015, 0.008, -0.005,
                0.012, 0.004, -0.006, 0.003, -0.002,
            ),
            (
                -0.003, 0.008, 0.012, -0.01, 0.015,
                -0.008, -0.005, 0.004, -0.003, 0.006,
            ),
        ),
    ],
)
def test_polynomial_forward_and_jacobian_against_numpy_oracle(
    order: EPolynomialOrder,
    coeffs_u: tuple[float, ...],
    coeffs_v: tuple[float, ...],
) -> None:
    """Verify polynomial displacement and Jacobian against NumPy polyder."""
    model = PolynomialMap(
        order=order, coeffs_u=coeffs_u, coeffs_v=coeffs_v,
    )

    x_test = np.linspace(-0.4, 0.4, 11)
    y_test = np.linspace(-0.3, 0.3, 11)
    grid_x, grid_y = np.meshgrid(x_test, y_test)
    x_flat = grid_x.ravel()
    y_flat = grid_y.ravel()

    act_xd, act_yd, j00, j01, j10, j11 = model.forward_with_jac(
        x_flat, y_flat,
    )

    exp_xd, exp_yd, exp_j0, exp_j1 = _eval_numpy_poly2d_oracle(
        x_flat, y_flat, order, coeffs_u, coeffs_v,
    )

    np.testing.assert_allclose(act_xd, exp_xd, atol=1.0e-15, rtol=1.0e-14)
    np.testing.assert_allclose(act_yd, exp_yd, atol=1.0e-15, rtol=1.0e-14)

    np.testing.assert_allclose(j00, exp_j0[:, 0], atol=1.0e-15, rtol=1.0e-14)
    np.testing.assert_allclose(j01, exp_j0[:, 1], atol=1.0e-15, rtol=1.0e-14)
    np.testing.assert_allclose(j10, exp_j1[:, 0], atol=1.0e-15, rtol=1.0e-14)
    np.testing.assert_allclose(j11, exp_j1[:, 1], atol=1.0e-15, rtol=1.0e-14)


# ------------------------------------------------------------------------------
# Round-Trip Invertibility Verification (Forward -> Inverse -> Forward)
# ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model",
    [
        BrownConrady(
            k1=0.1, k2=-0.03, k3=0.005, p1=0.002, p2=-0.001,
        ),
        BrownConradyExt(
            k1=0.08, k2=-0.02, k3=0.004,
            k4=0.01, k5=-0.005, k6=0.001,
            p1=0.001, p2=-0.002,
        ),
        PolynomialMap(
            order=EPolynomialOrder.QUADRATIC,
            coeffs_u=(0.002, 0.01, -0.005, 0.008, -0.004, 0.003),
            coeffs_v=(-0.003, 0.006, 0.008, -0.005, 0.007, -0.004),
        ),
        PolynomialMap(
            order=EPolynomialOrder.CUBIC,
            coeffs_u=(
                0.001, 0.008, -0.006, 0.004, -0.003,
                0.005, 0.002, -0.003, 0.002, -0.001,
            ),
            coeffs_v=(
                -0.002, 0.005, 0.007, -0.004, 0.006,
                -0.003, -0.002, 0.003, -0.001, 0.002,
            ),
        ),
        BrownConradyPolynomial(
            brown_conrady=BrownConrady(
                k1=0.05, k2=-0.01, p1=0.001, p2=-0.001,
            ),
            polynomial=BidirectionalPolynomial(
                forward_map=PolynomialMap(
                    order=EPolynomialOrder.QUADRATIC,
                    coeffs_u=(0.001, 0.005, -0.004, 0.003, -0.002, 0.002),
                    coeffs_v=(-0.002, 0.004, 0.005, -0.003, 0.004, -0.002),
                ),
            ),
        ),
        BrownConradyExtPolynomial(
            brown_conrady_ext=BrownConradyExt(
                k1=0.05, k2=-0.01, k3=0.002,
                k4=0.01, k5=0.002, k6=-0.001,
                p1=0.001, p2=-0.001,
            ),
            polynomial=BidirectionalPolynomial(
                forward_map=PolynomialMap(
                    order=EPolynomialOrder.QUADRATIC,
                    coeffs_u=(0.001, 0.005, -0.004, 0.003, -0.002, 0.002),
                    coeffs_v=(-0.002, 0.004, 0.005, -0.003, 0.004, -0.002),
                ),
            ),
        ),
    ],
)
def test_distortion_round_trip_invertibility(model) -> None:
    """Verify forward -> inverse round-trip consistency across the field."""
    # Test points covering center, edges, and corners
    points_x = np.array([0.0, 0.45, -0.45, 0.45, -0.45, 0.23, -0.31])
    points_y = np.array([0.0, 0.35, -0.35, -0.35, 0.35, -0.28, 0.19])

    xd, yd = model.forward(points_x, points_y)
    x_rec, y_rec = model.inverse(xd, yd)

    np.testing.assert_allclose(x_rec, points_x, atol=1.0e-11, rtol=1.0e-10)
    np.testing.assert_allclose(y_rec, points_y, atol=1.0e-11, rtol=1.0e-10)


# ------------------------------------------------------------------------------
# Integration Test with PixIntGrid2D
# ------------------------------------------------------------------------------


def test_pixintgrid2d_with_camera_distortion() -> None:
    """Render eggbox through PixIntGrid2D with camera distortion."""
    camera = rp.Camera2D(
        pixels_num=np.array([64, 64], dtype=np.int32),
        pixels_size=1.0,
        roi_cent_world=np.array([0.0, 0.0, 0.0]),
        distortion=BrownConrady(k1=0.08, k2=-0.02, p1=0.002, p2=-0.001),
        focal_length=64.0,
    )
    mesh = rp.Mesh2D(
        rp.EElementType.QUAD4,
        np.array([[-40, -40], [40, -40], [40, 40], [-40, 40]], dtype=float),
        np.array([[0, 1, 2, 3]], dtype=np.int64),
    )
    renderer = rp.PixIntGrid2D(
        texture=rp.Eggbox(period=(5.0, 5.0)),
        options=rp.PxInt2DOpts(
            mapping=rp.EPxIntMapping.AFFINE,
            integration=rp.GaussRule(2),
        ),
    )
    result = renderer.render(rp.Scene2D(mesh=mesh, camera=camera))
    image = result.images[0, 0, :, :, 0]

    assert image.shape == (64, 64)
    assert np.all(np.isfinite(image))
    assert 0.0 <= image.min() < image.max() <= 1.0
