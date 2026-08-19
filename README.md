# rasterpy

`rasterpy` preserves pyvale's original NumPy and Cython scalar-field 3D
rasterisers as a small standalone, archival Python package. It is suitable for
reproducing historic studies and inspecting the earlier renderer designs. For
new DIC uncertainty-quantification workflows, use Riley through `pyvale`.

## Install

```shell
uv pip install -e ".[dev]"
```

This builds the Cython implementation in the active environment. The runtime
package requires NumPy, SciPy, and Pillow.

## Tests

```shell
pytest
```

The regression suite renders a tiny in-memory triangle scene through both
engines. It does not need MOOSE, Exodus, or any external data.

## Examples

```shell
python examples/static_triangle_numpy.py
python examples/static_triangle_cython.py
```

The bundled historical example loads `case26_out.e` without running MOOSE:

```shell
python examples/ex_render_exodus.py
```

Pyvale is a runtime dependency and provides the Exodus reader and surface-mesh
converter used by the example.

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
