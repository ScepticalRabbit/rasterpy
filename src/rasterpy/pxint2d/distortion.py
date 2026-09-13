# ==============================================================================
# rasterpy: legacy DIC scalar-field rasterisers
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ==============================================================================
"""Camera distortion models for planar and volumetric pixel integrators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

# Fixed polynomial exponent pairs matching Riley order
POLY_POWERS_U: tuple[int, ...] = (0, 1, 0, 2, 1, 0, 3, 2, 1, 0)
POLY_POWERS_V: tuple[int, ...] = (0, 0, 1, 0, 1, 2, 0, 1, 2, 3)

DEFAULT_NEWTON_MAX_ITER: int = 50
DEFAULT_TOL_RESID: float = 1.0e-12
DEFAULT_TOL_DELTA: float = 1.0e-12
DEFAULT_TOL_DET: float = 1.0e-15


class EPolynomialOrder(IntEnum):
    """Supported polynomial distortion orders."""

    LINEAR = 1
    QUADRATIC = 2
    CUBIC = 3

    def term_count(self) -> int:
        """Return number of terms for the polynomial order."""
        if self == EPolynomialOrder.LINEAR:
            return 3
        if self == EPolynomialOrder.QUADRATIC:
            return 6
        if self == EPolynomialOrder.CUBIC:
            return 10
        raise ValueError(f"Unknown polynomial order: {self}")


class IDistortionModel(ABC):
    """Abstract interface for camera lens distortion models."""

    @abstractmethod
    def forward(
        self,
        x_coord: np.ndarray,
        y_coord: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply forward distortion: undistorted -> distorted."""

    @abstractmethod
    def forward_with_jac(
        self,
        x_coord: np.ndarray,
        y_coord: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Apply forward distortion and return (xd, yd, j00, j01, j10, j11)."""

    @abstractmethod
    def inverse(
        self,
        x_dist: np.ndarray,
        y_dist: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply inverse distortion: distorted -> undistorted."""


def _solve_newton_inverse(
    model: IDistortionModel,
    x_dist: np.ndarray,
    y_dist: np.ndarray,
    *,
    max_iter: int = DEFAULT_NEWTON_MAX_ITER,
    tol_resid: float = DEFAULT_TOL_RESID,
    tol_delta: float = DEFAULT_TOL_DELTA,
    tol_det: float = DEFAULT_TOL_DET,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve inverse distortion via 2D Newton-Raphson iteration."""
    x_cur = np.array(x_dist, dtype=np.float64, copy=True)
    y_cur = np.array(y_dist, dtype=np.float64, copy=True)

    for _ in range(max_iter):
        fwd_x, fwd_y, j00, j01, j10, j11 = model.forward_with_jac(
            x_cur, y_cur,
        )
        f0 = fwd_x - x_dist
        f1 = fwd_y - y_dist

        max_resid = float(np.max(np.maximum(np.abs(f0), np.abs(f1))))
        if max_resid < tol_resid:
            return x_cur, y_cur

        det = j00 * j11 - j01 * j10
        if np.any(np.abs(det) < tol_det):
            raise ValueError("Singular Jacobian encountered in distortion inv.")

        delta_x = (-f0 * j11 + j01 * f1) / det
        delta_y = (j10 * f0 - j00 * f1) / det

        x_cur += delta_x
        y_cur += delta_y

        max_delta = float(
            np.max(np.maximum(np.abs(delta_x), np.abs(delta_y)))
        )
        if max_delta < tol_delta:
            return x_cur, y_cur

    raise RuntimeError(
        f"Distortion inverse solver failed to converge in {max_iter} iters."
    )


@dataclass(slots=True)
class BrownConrady(IDistortionModel):
    """Standard 5-parameter Brown-Conrady camera lens distortion.

    Parameters
    ----------
    k1 : float
        First radial distortion coefficient.
    k2 : float
        Second radial distortion coefficient.
    k3 : float
        Third radial distortion coefficient.
    p1 : float
        First tangential distortion coefficient.
    p2 : float
        Second tangential distortion coefficient.
    """

    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    p1: float = 0.0
    p2: float = 0.0

    def _calc_radial(
        self,
        rad_sq: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute radial scale factor and its derivative w.r.t r^2."""
        rad4 = rad_sq * rad_sq
        rad6 = rad4 * rad_sq
        scale = 1.0 + self.k1 * rad_sq + self.k2 * rad4 + self.k3 * rad6
        dscale = self.k1 + 2.0 * self.k2 * rad_sq + 3.0 * self.k3 * rad4
        return scale, dscale

    def forward(
        self,
        x_coord: np.ndarray,
        y_coord: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate forward Brown-Conrady distortion."""
        rad_sq = x_coord * x_coord + y_coord * y_coord
        scale, _ = self._calc_radial(rad_sq)

        dx_tan = 2.0 * self.p1 * x_coord * y_coord + self.p2 * (
            rad_sq + 2.0 * x_coord * x_coord
        )
        dy_tan = self.p1 * (
            rad_sq + 2.0 * y_coord * y_coord
        ) + 2.0 * self.p2 * x_coord * y_coord

        return x_coord * scale + dx_tan, y_coord * scale + dy_tan

    def forward_with_jac(
        self,
        x_coord: np.ndarray,
        y_coord: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Evaluate forward Brown-Conrady distortion and Jacobian."""
        rad_sq = x_coord * x_coord + y_coord * y_coord
        scale, dscale = self._calc_radial(rad_sq)

        dx_tan = 2.0 * self.p1 * x_coord * y_coord + self.p2 * (
            rad_sq + 2.0 * x_coord * x_coord
        )
        dy_tan = self.p1 * (
            rad_sq + 2.0 * y_coord * y_coord
        ) + 2.0 * self.p2 * x_coord * y_coord

        x_dist = x_coord * scale + dx_tan
        y_dist = y_coord * scale + dy_tan

        j00 = (
            scale
            + 2.0 * x_coord * x_coord * dscale
            + 2.0 * self.p1 * y_coord
            + 6.0 * self.p2 * x_coord
        )
        j01 = (
            2.0 * x_coord * y_coord * dscale
            + 2.0 * self.p1 * x_coord
            + 2.0 * self.p2 * y_coord
        )
        j10 = j01
        j11 = (
            scale
            + 2.0 * y_coord * y_coord * dscale
            + 6.0 * self.p1 * y_coord
            + 2.0 * self.p2 * x_coord
        )
        return x_dist, y_dist, j00, j01, j10, j11

    def inverse(
        self,
        x_dist: np.ndarray,
        y_dist: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve inverse Brown-Conrady distortion via Newton-Raphson."""
        return _solve_newton_inverse(self, x_dist, y_dist)


@dataclass(slots=True)
class BrownConradyExt(IDistortionModel):
    """Extended rational 8-parameter Brown-Conrady lens distortion model.

    Parameters
    ----------
    k1, k2, k3 : float
        Radial numerator coefficients.
    k4, k5, k6 : float
        Radial denominator coefficients.
    p1, p2 : float
        Tangential distortion coefficients.
    """

    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    k4: float = 0.0
    k5: float = 0.0
    k6: float = 0.0
    p1: float = 0.0
    p2: float = 0.0

    def _calc_radial(
        self,
        rad_sq: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute rational radial scale factor and derivative w.r.t r^2."""
        rad4 = rad_sq * rad_sq
        rad6 = rad4 * rad_sq
        num = 1.0 + self.k1 * rad_sq + self.k2 * rad4 + self.k3 * rad6
        den = 1.0 + self.k4 * rad_sq + self.k5 * rad4 + self.k6 * rad6

        dnum = self.k1 + 2.0 * self.k2 * rad_sq + 3.0 * self.k3 * rad4
        dden = self.k4 + 2.0 * self.k5 * rad_sq + 3.0 * self.k6 * rad4

        scale = num / den
        dscale = (dnum * den - num * dden) / (den * den)
        return scale, dscale

    def forward(
        self,
        x_coord: np.ndarray,
        y_coord: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate forward extended rational Brown-Conrady distortion."""
        rad_sq = x_coord * x_coord + y_coord * y_coord
        scale, _ = self._calc_radial(rad_sq)

        dx_tan = 2.0 * self.p1 * x_coord * y_coord + self.p2 * (
            rad_sq + 2.0 * x_coord * x_coord
        )
        dy_tan = self.p1 * (
            rad_sq + 2.0 * y_coord * y_coord
        ) + 2.0 * self.p2 * x_coord * y_coord

        return x_coord * scale + dx_tan, y_coord * scale + dy_tan

    def forward_with_jac(
        self,
        x_coord: np.ndarray,
        y_coord: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Evaluate forward extended distortion and analytical Jacobian."""
        rad_sq = x_coord * x_coord + y_coord * y_coord
        scale, dscale = self._calc_radial(rad_sq)

        dx_tan = 2.0 * self.p1 * x_coord * y_coord + self.p2 * (
            rad_sq + 2.0 * x_coord * x_coord
        )
        dy_tan = self.p1 * (
            rad_sq + 2.0 * y_coord * y_coord
        ) + 2.0 * self.p2 * x_coord * y_coord

        x_dist = x_coord * scale + dx_tan
        y_dist = y_coord * scale + dy_tan

        j00 = (
            scale
            + 2.0 * x_coord * x_coord * dscale
            + 2.0 * self.p1 * y_coord
            + 6.0 * self.p2 * x_coord
        )
        j01 = (
            2.0 * x_coord * y_coord * dscale
            + 2.0 * self.p1 * x_coord
            + 2.0 * self.p2 * y_coord
        )
        j10 = j01
        j11 = (
            scale
            + 2.0 * y_coord * y_coord * dscale
            + 6.0 * self.p1 * y_coord
            + 2.0 * self.p2 * x_coord
        )
        return x_dist, y_dist, j00, j01, j10, j11

    def inverse(
        self,
        x_dist: np.ndarray,
        y_dist: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve inverse extended Brown-Conrady distortion."""
        return _solve_newton_inverse(self, x_dist, y_dist)


@dataclass(slots=True)
class PolynomialMap(IDistortionModel):
    """Directional 2D polynomial displacement distortion map.

    Parameters
    ----------
    order : EPolynomialOrder
        Polynomial degree (Linear, Quadratic, or Cubic).
    coeffs_u : tuple[float, ...]
        Coefficients for horizontal displacement (up to 10 terms).
    coeffs_v : tuple[float, ...]
        Coefficients for vertical displacement (up to 10 terms).
    """

    order: EPolynomialOrder = EPolynomialOrder.QUADRATIC
    coeffs_u: tuple[float, ...] = (0.0,) * 10
    coeffs_v: tuple[float, ...] = (0.0,) * 10

    def _eval_displacement(
        self,
        x_coord: np.ndarray,
        y_coord: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute displacement polynomial values."""
        num_terms = self.order.term_count()
        du = np.zeros_like(x_coord, dtype=np.float64)
        dv = np.zeros_like(y_coord, dtype=np.float64)

        for idx in range(num_terms):
            cu = self.coeffs_u[idx] if idx < len(self.coeffs_u) else 0.0
            cv = self.coeffs_v[idx] if idx < len(self.coeffs_v) else 0.0
            if cu == 0.0 and cv == 0.0:
                continue
            pu = POLY_POWERS_U[idx]
            pv = POLY_POWERS_V[idx]
            basis = (x_coord**pu) * (y_coord**pv)
            du += cu * basis
            dv += cv * basis

        return du, dv

    def forward(
        self,
        x_coord: np.ndarray,
        y_coord: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate forward polynomial map x + du, y + dv."""
        du, dv = self._eval_displacement(x_coord, y_coord)
        return x_coord + du, y_coord + dv

    def forward_with_jac(
        self,
        x_coord: np.ndarray,
        y_coord: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Evaluate forward polynomial map and analytical Jacobian."""
        num_terms = self.order.term_count()
        du = np.zeros_like(x_coord, dtype=np.float64)
        dv = np.zeros_like(y_coord, dtype=np.float64)
        ddu_dx = np.zeros_like(x_coord, dtype=np.float64)
        ddu_dy = np.zeros_like(y_coord, dtype=np.float64)
        ddv_dx = np.zeros_like(x_coord, dtype=np.float64)
        ddv_dy = np.zeros_like(y_coord, dtype=np.float64)

        for idx in range(num_terms):
            cu = self.coeffs_u[idx] if idx < len(self.coeffs_u) else 0.0
            cv = self.coeffs_v[idx] if idx < len(self.coeffs_v) else 0.0
            if cu == 0.0 and cv == 0.0:
                continue
            pu = POLY_POWERS_U[idx]
            pv = POLY_POWERS_V[idx]

            basis = (x_coord**pu) * (y_coord**pv)
            du += cu * basis
            dv += cv * basis

            if pu > 0:
                basis_dx = float(pu) * (x_coord ** (pu - 1)) * (y_coord**pv)
                ddu_dx += cu * basis_dx
                ddv_dx += cv * basis_dx
            if pv > 0:
                basis_dy = float(pv) * (x_coord**pu) * (y_coord ** (pv - 1))
                ddu_dy += cu * basis_dy
                ddv_dy += cv * basis_dy

        j00 = 1.0 + ddu_dx
        j01 = ddu_dy
        j10 = ddv_dx
        j11 = 1.0 + ddv_dy

        return x_coord + du, y_coord + dv, j00, j01, j10, j11

    def inverse(
        self,
        x_dist: np.ndarray,
        y_dist: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Invert polynomial displacement via Newton-Raphson iteration."""
        return _solve_newton_inverse(self, x_dist, y_dist)


@dataclass(slots=True)
class BidirectionalPolynomial(IDistortionModel):
    """Polynomial distortion with forward and/or inverse maps.

    Parameters
    ----------
    forward_map : PolynomialMap or None
        Forward polynomial map.
    inverse_map : PolynomialMap or None
        Inverse polynomial map.
    """

    forward_map: PolynomialMap | None = None
    inverse_map: PolynomialMap | None = None

    def forward(
        self,
        x_coord: np.ndarray,
        y_coord: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate forward map (or invert inverse map)."""
        if self.forward_map is not None:
            return self.forward_map.forward(x_coord, y_coord)
        if self.inverse_map is not None:
            return self.inverse_map.inverse(x_coord, y_coord)
        raise ValueError("BidirectionalPolynomial has no map defined.")

    def forward_with_jac(
        self,
        x_coord: np.ndarray,
        y_coord: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Evaluate forward map with Jacobian."""
        if self.forward_map is not None:
            return self.forward_map.forward_with_jac(x_coord, y_coord)
        raise NotImplementedError(
            "forward_with_jac requires an explicit forward_map."
        )

    def inverse(
        self,
        x_dist: np.ndarray,
        y_dist: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate inverse map (or invert forward map)."""
        if self.inverse_map is not None:
            return self.inverse_map.forward(x_dist, y_dist)
        if self.forward_map is not None:
            return self.forward_map.inverse(x_dist, y_dist)
        raise ValueError("BidirectionalPolynomial has no map defined.")


@dataclass(slots=True)
class BrownConradyPolynomial(IDistortionModel):
    """Chained Brown-Conrady + Polynomial distortion model.

    Parameters
    ----------
    brown_conrady : BrownConrady
        Brown-Conrady distortion stage.
    polynomial : BidirectionalPolynomial
        Polynomial distortion stage.
    """

    brown_conrady: BrownConrady
    polynomial: BidirectionalPolynomial

    def forward(
        self,
        x_coord: np.ndarray,
        y_coord: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate chained forward map: x -> BC -> Poly -> xd."""
        x_bc, y_bc = self.brown_conrady.forward(x_coord, y_coord)
        return self.polynomial.forward(x_bc, y_bc)

    def forward_with_jac(
        self,
        x_coord: np.ndarray,
        y_coord: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Evaluate chained forward map with composite Jacobian."""
        x_bc, y_bc, j00_a, j01_a, j10_a, j11_a = (
            self.brown_conrady.forward_with_jac(x_coord, y_coord)
        )
        x_d, y_d, j00_b, j01_b, j10_b, j11_b = (
            self.polynomial.forward_with_jac(x_bc, y_bc)
        )
        j00 = j00_b * j00_a + j01_b * j10_a
        j01 = j00_b * j01_a + j01_b * j11_a
        j10 = j10_b * j00_a + j11_b * j10_a
        j11 = j10_b * j01_a + j11_b * j11_a
        return x_d, y_d, j00, j01, j10, j11

    def inverse(
        self,
        x_dist: np.ndarray,
        y_dist: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate chained inverse map: xd -> Poly^-1 -> BC^-1 -> x."""
        x_poly_inv, y_poly_inv = self.polynomial.inverse(x_dist, y_dist)
        return self.brown_conrady.inverse(x_poly_inv, y_poly_inv)


@dataclass(slots=True)
class BrownConradyExtPolynomial(IDistortionModel):
    """Chained Extended Brown-Conrady + Polynomial distortion model.

    Parameters
    ----------
    brown_conrady_ext : BrownConradyExt
        Extended Brown-Conrady distortion stage.
    polynomial : BidirectionalPolynomial
        Polynomial distortion stage.
    """

    brown_conrady_ext: BrownConradyExt
    polynomial: BidirectionalPolynomial

    def forward(
        self,
        x_coord: np.ndarray,
        y_coord: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate chained forward map: x -> BC-Ext -> Poly -> xd."""
        x_bc, y_bc = self.brown_conrady_ext.forward(x_coord, y_coord)
        return self.polynomial.forward(x_bc, y_bc)

    def forward_with_jac(
        self,
        x_coord: np.ndarray,
        y_coord: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Evaluate chained forward map with composite Jacobian."""
        x_bc, y_bc, j00_a, j01_a, j10_a, j11_a = (
            self.brown_conrady_ext.forward_with_jac(x_coord, y_coord)
        )
        x_d, y_d, j00_b, j01_b, j10_b, j11_b = (
            self.polynomial.forward_with_jac(x_bc, y_bc)
        )
        j00 = j00_b * j00_a + j01_b * j10_a
        j01 = j00_b * j01_a + j01_b * j11_a
        j10 = j10_b * j00_a + j11_b * j10_a
        j11 = j10_b * j01_a + j11_b * j11_a
        return x_d, y_d, j00, j01, j10, j11

    def inverse(
        self,
        x_dist: np.ndarray,
        y_dist: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate chained inverse map: xd -> Poly^-1 -> BC-Ext^-1 -> x."""
        x_poly_inv, y_poly_inv = self.polynomial.inverse(x_dist, y_dist)
        return self.brown_conrady_ext.inverse(x_poly_inv, y_poly_inv)


__all__ = [
    "BidirectionalPolynomial",
    "BrownConrady",
    "BrownConradyExt",
    "BrownConradyExtPolynomial",
    "BrownConradyPolynomial",
    "EPolynomialOrder",
    "IDistortionModel",
    "POLY_POWERS_U",
    "POLY_POWERS_V",
    "PolynomialMap",
]
