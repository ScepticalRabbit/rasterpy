# ============================================================================
# rasterpy: legacy DIC scalar-field rasterisers
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
from rasterpy import data


def test_bundled_exodus_example_data_exists() -> None:
    """The optional historical example can locate its packaged Exodus file."""
    data_path = data.render_mechanical_3d_path()

    assert data_path.is_file()
    assert data_path.stat().st_size > 0
