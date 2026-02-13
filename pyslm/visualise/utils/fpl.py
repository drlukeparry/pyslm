from typing import *

import fastplotlib as fpl
import numpy as np

import pygfx

from fastplotlib.graphics._positions_base import PositionsGraphic

from fastplotlib.graphics.selectors import (
    LinearRegionSelector,
    LinearSelector,
    RectangleSelector,
    PolygonSelector
)
from fastplotlib.graphics.features import (
    Thickness,
    VertexPositions,
    VertexColors,
    UniformColor,
    VertexCmap,
    SizeSpace
)

class LineSegmentGraphic(PositionsGraphic):
    """
    Custom Line Segment Graphic for displaying pairs of vertices or discrete line segments
    that are not currently provided within the FastPlotLib distribution.
    """
    _features = {
        "data": VertexPositions,
        "colors": (VertexColors, UniformColor),
        "cmap": (VertexCmap, None),  # none if UniformColor
        "thickness": Thickness,
        "size_space": SizeSpace,
    }

    def __init__(self, data: Any,
                 thickness: float = 2.0,
                 dash_pattern: object = None,
                 colors: str | np.ndarray | Sequence = "w",
                 uniform_color: bool = False,
                 cmap: str = None,
                 cmap_transform: np.ndarray | Sequence = None,
                 isolated_buffer: bool = True,
                 size_space: str = "screen",
                 **kwargs) -> None:
        """
        Create a Line Segment Graphics in 2d or 3d

        :param data: Line data to plot. Can provide 1D, 2D, or a 3D data.
        :param thickness: Thickness of the line (default = 2.0)
        :param dash_pattern: Dash pattern used
        :param colors: Specify colors as a single human-readable string, a single RGBA array,
        :param uniform_color: if `True`, uses a uniform buffer for the line color,  basically saves GPU VRAM when the entire line has a single color
        :param cmap: Apply a colormap to the line instead of assigning colors manually, this overrides any argument passed to "colors". For supported colormaps see the ``cmap`` library catalogue: https://cmap-docs.readthedocs.io/en/stable/catalog
        :param cmap_transform:
        :param isolated_buffer:
        :param size_space: default "screen"  coordinate space in which the thickness is expressed ("screen", "world", "model")

        """

        super().__init__(data=data,
                         colors=colors,
                         uniform_color=uniform_color,
                         cmap=cmap,
                         cmap_transform=cmap_transform,
                         isolated_buffer=isolated_buffer,
                         size_space=size_space,
                         **kwargs)

        self._thickness = Thickness(thickness)

        if thickness < 1.1:
            MaterialCls = pygfx.LineSegmentMaterial
            aa = True
        else:
            MaterialCls = pygfx.LineMaterial

        aa = kwargs.get("alpha_mode", "auto") in ("blend", "weighted_blend")

        if uniform_color:
            geometry = pygfx.Geometry(positions=self._data.buffer)
            material = MaterialCls(aa=aa,
                                   thickness=self.thickness,
                                   color_mode="uniform",
                                   color=self.colors,
                                   dash_pattern=dash_pattern,
                                   pick_write=False,
                                   thickness_space=self.size_space,
                                   depth_compare="<=")
        else:
            material = MaterialCls(aa=aa,
                                   thickness=self.thickness,
                                   color_mode="vertex",
                                   pick_write=False,
                                   dash_pattern=dash_pattern,
                                   thickness_space=self.size_space,
                                   depth_compare="<=")

            geometry = pygfx.Geometry(positions=self._data.buffer, colors=self._colors.buffer)

        world_object: pygfx.Line = pygfx.Line(geometry=geometry, material=material)

        self._set_world_object(world_object)

    @property
    def thickness(self) -> float:
        """ Line thickness """
        return self._thickness.value

    @thickness.setter
    def thickness(self, value: float):
        self._thickness.set_value(self, value)

    def add_linear_selector(self, selection: float = None, axis: str = "x", **kwargs)  -> LinearSelector:
        return None

    def add_linear_region_selector(self, selection: tuple[float, float] = None,  padding: float = 0.0,
                                   axis: str = "x", **kwargs) -> LinearRegionSelector:
        return None

    def add_rectangle_selector(self, selection: tuple[float, float, float, float] = None,  **kwargs) -> None:
        return None

    def add_polygon_selector( self, selection: List[tuple[float, float]] = None,  **kwargs ) -> PolygonSelector:
        return None