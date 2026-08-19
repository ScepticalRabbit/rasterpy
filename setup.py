# ============================================================================== 
# rasterpy: legacy DIC scalar-field rasterisers
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================== 
from Cython.Build import cythonize
import numpy as np
from setuptools import Extension, setup


extensions = [
    Extension(
        "rasterpy.cython.rastercyth",
        ["src/rasterpy/cython/rastercyth.py"],
        include_dirs=[np.get_include()],
    ),
]


setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
    ),
)
