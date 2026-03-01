import logging

from typing import Any, Callable, List, Optional, Union
from collections.abc import Iterable

import fastplotlib as fpl
from rendercanvas.auto import RenderCanvas, loop

import numpy as np
import pygfx as gfx

from shapely.geometry import Polygon, MultiPolygon

from .utils import LineSegmentGraphic
from ..geometry import Layer, HatchGeometry, ContourGeometry, PointsGeometry
from ..hatching.utils import getContoursFromShapelyPolygon


def setColour(obj, color):
    if hasattr(obj, 'color'):
        obj.color = color
    elif hasattr(obj, 'material'):
        obj.material.color = color
    else:
        pass

def _initialisePlot(plot3D: bool = False,
                    plotGrid: bool = True) -> Any:

    if plot3D:

        canvas = RenderCanvas(vsync=False, max_fps=60, update_mode='ondemand')
        renderer = gfx.renderers.WgpuRenderer(canvas, show_fps=True)

        camera = gfx.PerspectiveCamera(50)
        camera.up = (0, 0, 1)

        plotObj = fpl.Figure(cameras=camera, controller_types='orbit', renderer=renderer, canvas=canvas)
    else:

        canvas = RenderCanvas(vsync=False,  max_fps=60, update_mode='ondemand')
        renderer = gfx.renderers.WgpuRenderer(canvas, ppaa='none', sort_objects=False)

        plotObj = fpl.Figure(renderer=renderer, canvas=canvas)

    plotObj.canvas.set_title('PySLM - Visualisation')

    fig = plotObj[0, 0]
    # Remove the axis title because this does not add value
    fig._frame._world_object.remove(fig._frame._title_graphic.world_object)
    fig._frame._title_graphic.font_size = 0
    fig._frame._reset()

    # Set a white background
    fig.background_color = [1, 1, 1, 1]

    # Set sensible defaults and colours for the grid and axes
    fig.axes.grids.xy.visible = plotGrid
    fig.axes.grids.xy.major_color = [0.8, 0.8, 0.8, 1]
    fig.axes.grids.xy.major_thickness = 0.6
    fig.axes.grids.xy.minor_thickness = 0.3
    fig.axes.grids.xy.minor_color = [0.9, 0.9, 0.9, 1]

    objs = fig.axes.x.children + fig.axes.y.children
    for obj in objs:
        obj.visible = plotGrid and not plot3D
        setColour(obj, [0.3, 0.3, 0.3, 1])

    return plotObj


def plot(layer: Layer,
         zPos: Optional[float] = 0,
         plotContours: Optional[bool] = True,
         plotHatches: Optional[bool] = True,
         plotPoints: Optional[bool] = True,
         plotSquential: Optional[bool] = False,
         plot3D: Optional[bool] = False,
         plotGrid: Optional[bool] = True,
         plotJumps: Optional[bool] = False,
         index: Optional[Union[str, Callable]] = '',
         handle: Optional[Any] = None,
         cmap: Optional[str] = None,
         show: Optional[bool] = True) -> Any:
    """
    Plots all the scan vectors (contours and hatches) and point exposures for each Layer Geometry in a Layer
    using `fastplotlib`. The :class:`Layer` may be plotted in 3D by setting the plot3D parameter.

    :param layer: A single :class:`Layer` containing a set of various  :class:`LayerGeometry` objects
    :param zPos: The position of the layer when using the 3D plot (optional)
    :param plotContours: Plots the inner hatch scan vectors. Defaults to `True`
    :param plotHatches: Plots the hatch scan vectors
    :param plotPoints: Plots point exposures
    :param plot3D: Plots the layer in 3D
    :param plotArrows: Plot the direction of each scan vector. This reduces the plotting performance due to use of
                       matplotlib annotations, should be disabled for large datasets
    :param index: A string defining the property to plot the scan vector geometry colours against
    :param handle: fastplotlib plot handle to re-use
    """


    cmap = 'viridis' if cmap is None else cmap

    if handle is None:
        plot_obj = _initialisePlot(plot3D, plotGrid=plotGrid)
    else:
        plot_obj = handle

    if len(layer.geometry) == 0:
        # return the plot object if an empty layer
        return plot_obj


    scanVectors = []

    # Collect all scan vector geometries
    for geom in layer.geometry:
        coords = None
        if isinstance(geom, HatchGeometry):
            coords = geom.coords.reshape(-1, 2, 2)
        elif isinstance(geom, ContourGeometry):
            coords = np.hstack([geom.coords, np.roll(geom.coords, -1, axis=0)])[:-1, :].reshape(-1, 2, 2)
        elif isinstance(geom, PointsGeometry):
            # Note that we duplicate the coordinates to represent emulate hatch vectors
            coords = np.tile(geom.coords.reshape(-1, 2), (1, 2)).reshape(-1, 2, 2)

        if coords is not None:
            scanVectors.append(coords)

    if len(scanVectors) == 0:
        logging.warning('pyslm.visualise.plotSequential: Empty layer')
        return plot_obj

    scanVectors = np.vstack(scanVectors)

    if plotJumps:
        # Plot the jumping vectors by rolling the entire stack of scan vectors
        svTmp = scanVectors.copy().reshape(-1, 2)
        svTmp = np.roll(svTmp, -1, axis=0)[0:-2]
        svTmp = svTmp.reshape(-1, 2, 2)

        if plot3D:
            jumps_3d = np.column_stack([svTmp.reshape(-1, 2), np.full(svTmp.shape[0] * 2, zPos)])
            lc2 = LineSegmentGraphic(data=jumps_3d.reshape(-1,3), thickness=1.0, dash_pattern=(3,3), colors=(0.7,0.7,0.7,1))
        else:
            lc2 = LineSegmentGraphic(data=svTmp.reshape(-1,2), thickness=1.0, dash_pattern=(3,3), colors=(0.7,0.7,0.7,1))
        plot_obj[0,0].add_graphic(lc2)

    if plotSquential:

        # Create colormap based on cumulative distance
        delta = scanVectors[:, 1, :] - scanVectors[:, 0, :]
        dist = np.sqrt(delta[:, 0] * delta[:, 0] + delta[:, 1] * delta[:, 1])
        cumDist = np.cumsum(dist)
        normalized_dist = cumDist / cumDist.max() if len(cumDist) > 0 else np.array([])

        if plot3D:
            hatches_3d = np.column_stack([scanVectors.reshape(-1, 2), np.full(scanVectors.shape[0] * 2, zPos)])


            lc = LineSegmentGraphic(data=hatches_3d.reshape(-1, 3), cmap_transform=np.repeat(normalized_dist, 2), cmap=cmap,
                                    thickness=0.5)
        else:


            lc = LineSegmentGraphic(data=scanVectors.reshape(-1, 2), cmap_transform=np.repeat(normalized_dist, 2), cmap=cmap,
                                    thickness=0.5)

        plot_obj[0, 0].add_graphic(lc)
        # Plot points geometry
        pointsGeom = layer.getPointsGeometry()
        if len(pointsGeom) > 0:
            coords = np.vstack([geom.coords for geom in pointsGeom])
            plot_obj[0, 0].add_scatter(coords, size=3, color='black')
    else:

        if plotHatches:
            hatchGeoms = layer.getHatchGeometry()

            if len(hatchGeoms) > 0:

                hatches = np.vstack([hatchGeom.coords.reshape(-1, 2, 2) for hatchGeom in hatchGeoms])

                # Compute colors based on index parameter
                if type(index) is str and hasattr(hatchGeoms[0], index):
                    values = [np.tile(getattr(hGeom, index), [int(len(hGeom.coords)/2), 1]) for hGeom in hatchGeoms]
                    values = np.vstack(values).ravel()
                    # Normalize values to [0, 1] for colormap
                    values = (values - values.min()) / (values.max() - values.min() + 1e-10)

                elif type(index) is str and index == 'length':
                    delta = hatches[:, 1, :] - hatches[:, 0, :]
                    dist = np.sqrt(delta[:, 0]*delta[:, 0] + delta[:, 1]*delta[:, 1])
                    values = dist / (dist.max() + 1e-10)

                elif callable(index):
                    values = np.vstack([index(hatchGeom) for hatchGeom in hatchGeoms])
                    values = values.ravel()
                    values = (values - values.min()) / (values.max() - values.min() + 1e-10)

                else:
                    # Plot the sequential index of the hatch vector
                    values = np.arange(len(hatches)) / len(hatches)

                if plot3D:
                    hatches_3d = np.column_stack([hatches.reshape(-1, 2), np.full(hatches.shape[0]*2, zPos)])

                    lc = LineSegmentGraphic(data=hatches_3d.reshape(-1, 3), cmap_transform=np.repeat(values, 2),
                                            cmap=cmap, thickness=0.5)
                else:

                    lc = LineSegmentGraphic(data=hatches.reshape(-1,2), cmap_transform = np.repeat(values,2), cmap=cmap, thickness=0.5)

                lc = plot_obj[0,0].add_graphic(lc)

        if plotContours:

            segs = []
            for contourGeom in layer.getContourGeometry():
                coords = np.asarray(contourGeom.coords)
                if coords.shape[0] < 2:
                    continue
                # build consecutive pairs (p0,p1), closing the loop by rolling
                pairs = np.hstack([coords, np.roll(coords, -1, axis=0)])[:-1, :].reshape(-1, 2, 2)
                segs.append(pairs)

            if len(segs) > 0:

                segs = np.vstack(segs)

                if plot3D:
                    segs = np.column_stack([segs.reshape(-1, 2), np.full(segs.shape[0] * 2, zPos)])
                else:
                    segs = np.column_stack([segs.reshape(-1, 2), np.full(segs.shape[0] * 2, 0.0)])


                lineColor = '#204a87'
                lineGraphic = LineSegmentGraphic(data=segs, colors=lineColor, uniform_color=True,  thickness=0.5)
                plot_obj[0,0].add_graphic(lineGraphic) #

            if False:

                for contourGeom in layer.getContourGeometry():

                    if hasattr(contourGeom, 'subType'):
                        if contourGeom.subType == 'inner':
                            lineColor = '#f57900'
                            lineWidth = 1.0
                        elif contourGeom.subType == 'outer':
                            lineColor = '#204a87'
                            lineWidth = 1.4
                    else:
                        lineColor = (0.0,0.0,0.0,1.0)
                        lineWidth = 0.7

                    if plot3D:
                        contour_3d = np.column_stack([contourGeom.coords[:, 0], contourGeom.coords[:, 1],
                                                      np.full(len(contourGeom.coords), zPos)])

                        #lineGraphic = LineSegmentGraphic(data=contour_3d, colors=lineColor, uniform_color=True,  thickness=0.5)
                        #plot_obj[0,0].add_graphic(lineGraphic) #
                        plot_obj[0,0].add_line(data=contour_3d, colors=lineColor, thickness=lineWidth)
                    else:

                        plot_obj[0,0].add_line(data=contourGeom.coords, colors=lineColor, thickness=lineWidth)
        if plotPoints:

            pointGeoms = layer.getPointsGeometry()

            if len(pointGeoms) > 0:

                scatterPoints = np.vstack([pointsGeom.coords for pointsGeom in pointGeoms])

                if callable(index):
                    values = np.vstack([index(pointGeom) for pointGeom in pointGeoms])
                    pntColors = values.ravel()
                else:
                    # Plot the sequential index of the hatch vector
                    pntColors = np.arange(len(scatterPoints)) / len(scatterPoints)


                if plot3D:
                    scatterPoints3D = np.column_stack([scatterPoints[:, 0], scatterPoints[:, 1],
                                                         np.full(len(scatterPoints), zPos)])
                    plot_obj[0,0].add_scatter(scatterPoints3D, sizes=5, colors=(0.0,0.0,0.0,1.0))

                else:
                    plot_obj[0,0].add_scatter(scatterPoints, sizes=5, colors=(0.0,0.0,0.0,1.0))


    if(show):
        plot_obj.show();  fpl.loop.run()

    return plot_obj


def plotPolygon(polygons: List[Any], zPos: Optional[float] = 0.0,
                lineColor: Optional[Any] = 'k', lineWidth: Optional[float] = 0.7,
                fillColor: Optional[Any] = 'r',
                plot3D: Optional[bool] = False, plotFilled: Optional[bool] = False,
                handle: Optional[Any] = None, show: Optional[bool] = True) -> Any:
    """
    A helper method for visualising polygons using fastplotlib.

    .. note::
        This function uses fastplotlib for GPU-accelerated visualization.
        Method cannot deal with complex polygons (those with interiors).

    This function can interpret a variety of different polygon formats used throughout the library,
    including those from Shapely, and those from coordinates generated by ClipperLib.

    :param polygons: A list of polygons
    :param zPos: The z position of the polygons if plot3D is enabled
    :param lineColor: Line color (matplotlib format: string like 'k', 'r' or hex '#f57900')
    :param lineWidth: Line width used for rendering
    :param fillColor: Fill color for the polygon if plotFilled is enabled
    :param plot3D: Plot the polygons in 3D
    :param plotFilled: Plot filled polygons (uses filled line segments)
    :param handle: A previous fastplotlib Figure object to reuse
    :param show: If True, display the plot interactively

    :return: fastplotlib Figure object
    """

    # Initialize or reuse plot
    if handle is None:
        plot_obj = _initialisePlot(plot3D, plotGrid=True)
    else:
        plot_obj = handle

    # Extract contour coordinates from various polygon formats
    contourCoords = []

    if isinstance(polygons, MultiPolygon):
        polygons = list(polygons.geoms)

    if not isinstance(polygons, Iterable):
        polygons = [polygons]

    for poly in polygons:
        if isinstance(poly, Polygon):
            contourCoords += getContoursFromShapelyPolygon(poly)
        elif isinstance(poly, MultiPolygon):
            contourCoords += [getContoursFromShapelyPolygon(p) for p in list(poly)]
        else:
            # Assume it's already coordinate array
            contourCoords.append(poly)

    # Plot each contour
    fig = plot_obj[0, 0]

    for contour in contourCoords:
        if len(contour) < 2:
            continue

        # Close the polygon by adding first point at the end
        closed_contour = np.vstack([contour, contour[0:1]])


        if plotFilled:
            # Use filled representation with line segments
            fig.add_polygon(closed_contour, colors=fillColor, offset=(0.0,0.0,zPos+1e-6))

        lc = LineSegmentGraphic(data=closed_contour, thickness=lineWidth,
                                colors=lineColor, uniform_color=True,  offset=(0.0,0.0,zPos))
        fig.add_graphic(lc)


    if show:
        plot_obj.show()
        fpl.loop.run()

    return plot_obj


def plotLayers(layers: List[Layer],
               plotContours: Optional[bool] = True,
               plotHatches: Optional[bool] = True,
               plotPoints: Optional[bool] = True,
               plot3D : Optional[bool] = True,
               handle: Optional[Any] = None) -> Any:
    """
    Plots a list of :class:`Layer`, specifically the scan vectors (contours and hatches) and point exposures for each
    :class:`LayerGeometry` using `fastplotlib`. The Layer may be plotted in 3D.

    :param layers: A list of :class:`Layer`
    :param plotContours: Plots the inner hatch scan vectors. Defaults to `True`
    :param plotHatches: Plots the hatch scan vectors
    :param plotPoints: Plots point exposures
    :param handle: fastplotlib plot handle to re-use
    """

    plot_obj = handle

    for i, layer in enumerate(layers):
        print(i, layer)
        plot_obj = plot(layer, layer.z,
                        plot3D=plot3D, plotContours=plotContours, plotHatches=plotHatches, plotPoints=plotPoints,
                        handle=plot_obj, show=False)

    plot_obj.show(); fpl.loop.run()
    return plot_obj