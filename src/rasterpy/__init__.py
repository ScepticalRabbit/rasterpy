# ============================================================================== 
# rasterpy: legacy DIC scalar-field rasterisers
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================== 
"""Legacy NumPy and Cython scalar-field rasterisers for DIC studies."""

from .camera import Camera, Camera2D, CameraData
from .camera_tools import CameraTools
from .capabilities import RenderCapabilities
from .cython_renderer import RasterCY
from .image_tools import EImageType, ImageTools, image_save
from .imagedef2d import ImageDef2D, ImageDefOpts
from .imagewarp2d import IImageWarp2D
from .mesh import EElementType, Mesh, Mesh2D, RenderMesh
from .numpy_renderer import RasterNP, RasterNumpy
from .options import RasterOpts, RenderOptions
from .pxint2d import (
    AdditiveSpeckles,
    AnalyticRule,
    Eggbox,
    EPxIntMapping,
    GaussianPSF,
    GaussRule,
    PixIntGrid2D,
    PixIntSpeck2D,
    PxInt2DOpts,
    RectRule,
)
from .result import ImageWarpResult
from .scene import RenderScene, Scene, Scene2D

__all__ = [
    "AdditiveSpeckles",
    "AnalyticRule",
    "Camera",
    "Camera2D",
    "CameraData",
    "CameraTools",
    "EElementType",
    "EImageType",
    "EPxIntMapping",
    "Eggbox",
    "GaussianPSF",
    "GaussRule",
    "IImageWarp2D",
    "ImageDef2D",
    "ImageDefOpts",
    "ImageTools",
    "ImageWarpResult",
    "Mesh",
    "Mesh2D",
    "PixIntGrid2D",
    "PixIntSpeck2D",
    "PxInt2DOpts",
    "RasterCY",
    "RasterNP",
    "RasterNumpy",
    "RasterOpts",
    "RectRule",
    "RenderCapabilities",
    "RenderMesh",
    "RenderOptions",
    "RenderScene",
    "Scene",
    "Scene2D",
    "image_save",
]
