# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Analytic two-dimensional finite-element renderers."""

from .distortion import (
    POLY_POWERS_U,
    POLY_POWERS_V,
    BidirectionalPolynomial,
    BrownConrady,
    BrownConradyExt,
    BrownConradyExtPolynomial,
    BrownConradyPolynomial,
    EPolynomialOrder,
    IDistortionModel,
    PolynomialMap,
)
from .grid import PixIntGrid2D
from .model import (
    AnalyticRule,
    Eggbox,
    EPxIntMapping,
    GaussianPSF,
    GaussRule,
    PxInt2DOpts,
    RectRule,
    quantise_image,
)
from .speck import AdditiveSpeckles, PixIntSpeck2D

__all__ = [
    "AdditiveSpeckles",
    "AnalyticRule",
    "BidirectionalPolynomial",
    "BrownConrady",
    "BrownConradyExt",
    "BrownConradyExtPolynomial",
    "BrownConradyPolynomial",
    "EPolynomialOrder",
    "EPxIntMapping",
    "Eggbox",
    "GaussRule",
    "GaussianPSF",
    "IDistortionModel",
    "POLY_POWERS_U",
    "POLY_POWERS_V",
    "PixIntGrid2D",
    "PixIntSpeck2D",
    "PolynomialMap",
    "PxInt2DOpts",
    "RectRule",
    "quantise_image",
]
