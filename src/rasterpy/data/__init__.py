# ============================================================================== 
# rasterpy: legacy DIC scalar-field rasterisers
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================== 
"""Packaged demonstration data."""

from importlib.resources import files
from pathlib import Path


def render_mechanical_3d_path() -> Path:
    """Return the bundled Exodus case used by the optional example."""
    return Path(files("rasterpy.data").joinpath("case26_out.e"))


__all__ = ["render_mechanical_3d_path"]
