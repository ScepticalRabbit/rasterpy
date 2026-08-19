# rasterpy

`rasterpy` preserves pyvale's original NumPy and Cython scalar-field 3D
rasterisers as a small standalone, archival Python package. It is suitable for
reproducing historic studies and inspecting the earlier renderer designs. For
new DIC uncertainty-quantification workflows, use Riley through `pyvale`.

## Install

```shell
uv sync --group dev
```

This builds the Cython implementation in the active environment. The runtime
package requires NumPy, SciPy, and Pillow.

## Tests

```shell
uv run pytest
```

The regression suite renders a tiny in-memory triangle scene through both
engines. It does not need MOOSE, Exodus, or any external data.

## Examples

```shell
uv run python examples/static_triangle_numpy.py
uv run python examples/static_triangle_cython.py
```

An optional historical example loads the bundled `case26_out.e` MOOSE Exodus
result without running MOOSE:

```shell
uv sync --extra examples
uv run python examples/exodus_case26_numpy.py
```

The Exodus example uses pyvale only as an optional reader and surface-mesh
converter. The rasteriser package itself has no pyvale runtime dependency.

## API

```python
import rasterpy

renderer = rasterpy.RasterNumpy(rasterpy.RasterOpts(parallel=None))
image = renderer.render(scene)
```

The historic names `CameraData`, `RenderMesh`, `RenderScene`, `RasterNumpy`,
`RasterNP`, `RasterCY`, and `RasterOpts` remain available.

## Licence

MIT. Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher).
