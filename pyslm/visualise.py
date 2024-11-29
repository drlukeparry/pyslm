import logging

from typing import Any, List, Tuple, Optional
from collections.abc import Iterable

import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.collections as mc

import numpy as np

from shapely.geometry import Polygon, MultiPolygon

from .core import Part
from .geometry import Layer, HatchGeometry, ContourGeometry, PointsGeometry


def getContoursFromShapelyPolygon(poly: Polygon, mergeRings: Optional[bool] = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the  contours from boundaries extracted from exterior and interior paths of shapely geometry.

    :param poly: Shapely polygons
    :param mergeRings: If ``True`` combines the interior and exteriors within each path group
    :return: The paths extracted from the polygon paths
    """
    outerRings = []
    innerRings = []

    outerRings += [np.array(tuple(poly.exterior.coords))]

    for ring in poly.interiors:
        innerRings += [np.array(tuple(ring.coords))]

    if mergeRings:
        return outerRings + innerRings
    else:
        return outerRings, innerRings


def plotPolygon(polygons: List[Any], zPos = 0.0,
                lineColor: Optional[Any] = 'k', lineWidth: Optional[float] = 0.7, fillColor: Optional[Any] = 'r',
                plot3D: Optional[bool] = False, plotFilled: Optional[bool] = False,
                handle: Tuple[plt.Figure, plt.Axes] = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Helper method for plotting polygons (numpy coordinates) and those composed of Python lists.

    :note:
        Method cannot deal with complex polygon i.e. those with interiors due to limitation with Matplotlib

    :param polygons: A list of polygons
    :param zPos: The z position of the polygons if plot3D is enabled
    :param lineColor: Line color used for matplotlib (optional)
    :param lineWidth: Line width used for matplotlib (optional)
    :param fillColor:  Fill color for the polygon if plotFilled is enabled (optional)
    :param plot3D: Plot the polygons in 3D
    :param plotFilled: Plot filled
    :param handle: A previous matplotlib (Figure, Axis) object

    :return: A tuple with the matplotlib (Figure, Axis)
    """

    if handle:
        fig = handle[0]
        ax = handle[1]

    else:
        if plot3D:
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            ax = plt.axes(projection='3d')
        else:
            fig, ax = plt.subplots()

    #ax.axis('equal')
    plotNormalize = matplotlib.colors.Normalize()

    patchList = []

    contourCoords = []

    if  isinstance(polygons, MultiPolygon):
        polygons = list(polygons.geoms)

    if not isinstance(polygons, Iterable):
        polygons = [polygons]

    for poly in polygons:
        if isinstance(poly, Polygon):
            contourCoords += getContoursFromShapelyPolygon(poly)
        elif isinstance(poly, MultiPolygon):
            contourCoords += [getContoursFromShapelyPolygon(p) for p in list(poly)]
        else:
            contourCoords.append(poly)

    for contour in contourCoords:

        if plot3D:
            ax.plot(contour[:, 0], contour[:, 1], zs=zPos, color=lineColor, linewidth=lineWidth)
        else:
            if plotFilled:
                polygon = matplotlib.patches.Polygon(contour, fill=True, linewidth=lineWidth, edgecolor=lineColor, color=fillColor, facecolor=fillColor)
                #ax.add_patch(polygon)
                patchList.append(polygon)

            else:
                ax.plot(contour[:, 0], contour[:, 1], color=lineColor, linewidth=lineWidth)

    if plotFilled:
        p = mc.PatchCollection(patchList, alpha=1)
        ax.add_collection(p)

    return fig, ax


def plotLayers(layers: List[Layer],
               plotContours: Optional[bool] = True, plotHatches: Optional[bool] = True,
               plotPoints: Optional[bool] = True,
               handle: Optional[Tuple[plt.Figure, plt.Axes]] = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots a list of :class:`Layer`, specifically the scan vectors (contours and hatches) and point exposures for each
    :class:`LayerGeometry` using `Matplotlib`. The Layer may be plotted in 3D by setting the plot3D parameter.

    :param layers: A list of :class:`Layer`
    :param plotContours: Plots the inner hatch scan vectors. Defaults to `True`
    :param plotHatches: Plots the hatch scan vectors
    :param plotPoints: Plots point exposures
    :param handle: Matplotlib handle to re-use
    """
    if handle:
        fig = handle[0]
        ax = handle[1]
    else:
        fig = plt.figure()
        ax = plt.axes(projection='3d')

    for layer in layers:
        fig, ax = plot(layer, float(layer.z)/1000,
                       plot3D=True, plotContours=plotContours, plotHatches=plotHatches, plotPoints=plotPoints,
                       handle=(fig, ax))

    return fig, ax


def plotSequential(layer: Layer,
                   plotArrows: Optional[bool] = False, plotOrderLine: Optional[bool] = False,
                   plotJumps: Optional[bool] = False,
                   handle: Optional[Tuple[plt.Figure, plt.Axes]]  = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots sequentially the all the scan vectors (contours and hatches) for all Layer Geometry in a Layer
    using `Matplotlib`. The :class:`Layer` may be only plotted across a single 2D layer.

    :param layer: A single :class:`Layer` containing a set of various  :class:`LayerGeometry` objects
    :param plotArrows: Plot the direction of each scan vector. This reduces the plotting performance due to use of
                       matplotlib annotations, should be disabled for large datasets
    :param plotOrderLine: Plots an additional line showing the order of vector scanning
    :param plotJumps:  Plots the jumps (in dashed lines) between vectors
    :param handle: Matplotlib handle to re-use
    """

    if handle:
        fig = handle[0]
        ax = handle[1]

    else:
        fig, ax = plt.subplots()
        ax.axis('equal')

    plotNormalize = matplotlib.colors.Normalize()

    scanVectors = []

    for geom in layer.geometry:

        if isinstance(geom, HatchGeometry):
            coords = geom.coords.reshape(-1, 2, 2)
        elif isinstance(geom, ContourGeometry):
            coords = np.hstack([geom.coords, np.roll(geom.coords, -1, axis=0)])[:-1,:].reshape(-1,2,2)
        elif isinstance(geom, PointsGeometry):
            # Note that we duplicate the coordinates to represent emulate hatch vectors
            coords = np.tile(geom.coords.reshape(-1, 2), (1,2)).reshape(-1,2,2)

        scanVectors.append(coords)

    if len(scanVectors) == 0:
        logging.warning('pyslm.visualise.plotSequential: Empty layer')
        return

    scanVectors = np.vstack(scanVectors)

    lc = mc.LineCollection(scanVectors, cmap=plt.cm.rainbow, linewidths=1.0)

    pointsGeom = layer.getPointsGeometry()

    if len(pointsGeom) > 0:
        coords = np.vstack([geom.coords for geom in pointsGeom])
        print('coords shape', coords.shape)
        ax.scatter(coords[:,0], coords[:,1], color='#000', s=1.0)

    if plotOrderLine:
        midPoints = np.mean(scanVectors, axis=1)
        idx6 = np.arange(len(scanVectors))
        ax.plot(midPoints[idx6][:, 0], midPoints[idx6][:, 1])

    """
    Plot the sequential index of the hatch vector and generating the colourmap by using the cumulative distance
    across all the scan vectors in order to normalise the length based effectively on the distance
    """
    delta = scanVectors[:, 1, :] - scanVectors[:, 0, :]
    dist = np.sqrt(delta[:, 0] * delta[:, 0] + delta[:, 1] * delta[:, 1])
    cumDist = np.cumsum(dist)
    #lc.set_array(np.arange(len(scanVectors)))
    lc.set_array(cumDist)

    # Add all the line collections to the figure
    ax.add_collection(lc)

    if plotJumps:
        # Plot the jumping vectors by rolling the entire stack of scan vectors
        svTmp = scanVectors.copy().reshape(-1, 2)
        svTmp = np.roll(svTmp, -1, axis=0)[0:-2]
        svTmp = svTmp.reshape(-1, 2, 2)

        # scanVectors = np.vstack([scanVectors, svTmp])

        jumpLC = mc.LineCollection(svTmp, cmap=plt.cm.get_cmap('Greys'), linewidths=0.3, linestyles="--", lw=0.7)

        ax.add_collection(jumpLC)

    ax.plot()

    if plotArrows:
        for hatch in scanVectors:
            midPoint = np.mean(hatch, axis=0)
            delta = hatch[1, :] - hatch[0, :]

            ax.annotate('', xytext=midPoint - delta * 1e-4,
                         xy=midPoint,
                         arrowprops={'arrowstyle': "->", 'facecolor': 'black'})

    return fig, ax

def plot(layer: Layer, zPos:Optional[float] = 0,
         plotContours: Optional[bool] = True, plotHatches: Optional[bool] = True, plotPoints: Optional[bool] = True,
         plot3D: Optional[bool] = False, plotArrows: Optional[bool] = False, plotOrderLine: Optional[bool] = False,
         plotColorbar: Optional[bool] = False,
         index: Optional[str] = '',
         handle: Optional[Tuple[plt.Figure, plt.Axes]] = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots the all the scan vectors (contours and hatches) and point exposures for each Layer Geometry in a Layer
    using `Matplotlib`. The :class:`Layer` may be plotted in 3D by setting the plot3D parameter.

    :param layer: A single :class:`Layer` containing a set of various  :class:`LayerGeometry` objects
    :param zPos: The position of the layer when using the 3D plot (optional)
    :param plotContours: Plots the inner hatch scan vectors. Defaults to `True`
    :param plotHatches: Plots the hatch scan vectors
    :param plotPoints: Plots point exposures
    :param plot3D: Plots the layer in 3D
    :param plotArrows: Plot the direction of each scan vector. This reduces the plotting performance due to use of
                       matplotlib annotations, should be disabled for large datasets
    :param plotOrderLine: Plots an additional line showing the order of vector scanning
    :param plotColorbar: Plots a colorbar for the hatch section
    :param index: A string defining the property to plot the scan vector geometry colours against
    :param handle: Matplotlib handle to re-use
    """

    if handle:
        fig = handle[0]
        ax = handle[1]

    else:
        if plot3D:
            fig = plt.figure()
            ax = plt.axes(projection='3d')

        else:
            fig, ax = plt.subplots()
            ax.axis('equal')

    plotNormalize = matplotlib.colors.Normalize()

    if plotHatches:
        hatchGeoms = layer.getHatchGeometry()

        if len(hatchGeoms) > 0:

            hatches = np.vstack([hatchGeom.coords.reshape(-1, 2, 2) for hatchGeom in hatchGeoms])

            lc = mc.LineCollection(hatches, cmap=plt.cm.rainbow, linewidths=0.5)

            """ Plot """
            if type(index) is str and str and hasattr(hatchGeoms[0], index):

                values = np.vstack([np.tile(getattr(hatchGeom, index), [int(len(hatchGeom.coords)/2),1]) for hatchGeom in hatchGeoms])
                lc.set_array(values.ravel())

            elif type(index) is str and index == 'length':

                delta = hatches[:,1,:] - hatches[:,0,:]
                dist = np.sqrt(delta[:,0]*delta[:,0] + delta[:,1]*delta[:,1])
                lc.set_array(dist.ravel())

            elif callable(index):

                values = np.vstack([index(hatchGeom) for hatchGeom in hatchGeoms])
                lc.set_array(values.ravel())

            else:
                # Plot the sequential index of the hatch vector
                lc.set_array(np.arange(len(hatches)))

            if plotArrows and not plot3D:
                for hatch in hatches:
                    midPoint = np.mean(hatch, axis=0)
                    delta = hatch[1, :] - hatch[0, :]

                    ax.annotate('', xytext = midPoint - delta * 1e-4,
                                     xy = midPoint,
                                     arrowprops={'arrowstyle': "->", 'facecolor': 'black'})

            if plot3D:
                ax.add_collection3d(lc, zs=zPos)
            else:
                ax.add_collection(lc)
                ax.plot()

            if not plot3D and plotOrderLine:

                midPoints = np.mean(hatches, axis=1)
                idx6 = np.arange(len(hatches))
                ax.plot(midPoints[idx6][:, 0], midPoints[idx6][:, 1])

            if plotColorbar:
                axcb = fig.colorbar(lc)

    if plotContours:

        for contourGeom in layer.getContourGeometry():

            if hasattr(contourGeom, 'subType'):
                if contourGeom.subType == 'inner':
                    lineColor = '#f57900'
                    lineWidth = 1
                elif contourGeom.subType == 'outer':
                    lineColor = '#204a87'
                    lineWidth = 1.4
            else:
                lineColor = 'k'
                lineWidth = 0.7

            if plotArrows and not plot3D:
                for i in range(contourGeom.coords.shape[0] - 1):
                    midPoint = np.mean(contourGeom.coords[i:i + 2], axis=0)
                    delta = contourGeom.coords[i + 1, :] - contourGeom.coords[i, :]

                    ax.annotate('',
                                 xytext=midPoint - delta * 1e-4,
                                 xy=midPoint,
                                 arrowprops={'arrowstyle': "->", 'facecolor': 'black'})

            if plot3D:
                ax.plot(contourGeom.coords[:, 0], contourGeom.coords[:, 1], zs=zPos, color=lineColor,
                        linewidth=lineWidth)
            else:
                ax.plot(contourGeom.coords[:, 0], contourGeom.coords[:, 1], color=lineColor,
                        linewidth=lineWidth)

    if plotPoints:

        pointGeoms = layer.getPointsGeometry()

        if len(pointGeoms) > 0:

            scatterPoints = np.vstack([pointsGeom.coords for pointsGeom in layer.getPointsGeometry()])

            pointrGeoms =  layer.getPointsGeometry()

            pntColors = None
            if callable(index):
                values = np.vstack([index(pointGeom) for pointGeom in pointrGeoms])
                pntColors = values.ravel()

            else:
                # Plot the sequential index of the hatch vector
                pntColors =  np.arange(len(scatterPoints))

            if plot3D:
                scatterObj = ax.scatter3D(scatterPoints[:, 0], scatterPoints[:, 1], zPos, c=pntColors)
            else:

                scatterObj = ax.scatter(scatterPoints[:, 0], scatterPoints[:, 1], c=pntColors)

            #axcb = fig.colorbar(scaterObj)

            #for pointsGeom in layer.getPointsGeometry():
            #   ax.scatter(pointsGeom.coords[:, 0], pointsGeom.coords[:, 1], 'x')

    if False:
        world_limits = ax.get_w_lims()
        ax.set_box_aspect((world_limits[1] - world_limits[0], world_limits[3] - world_limits[2],
                           world_limits[5] - world_limits[4]))
    return fig, ax


def plotHeatMap(part: Part, z: float, exposurePoints: np.ndarray, resolution: float = 0.25) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots an effective heat map given the exposure points and at a given z position. The heatmap is discretised by
    summing the energy input of all exposure points onto an image and then capturing the aerial heat input by dividing
    by the pixel area.

    :param part: The part that as been sliced.
    :param z: The layer z-position to slice
    :param exposurePoints: (nx3) array of exposure points
    :param resolution: resolution to generate the heatmap to process on
    """

    fig, ax = plt.subplots()
    ax.axis('equal')

    bitmap = part.getBitmapSlice(z, 0.1)
    offset = part.boundingBox[:2]

    if exposurePoints.shape[1] != 3:
        raise ValueError('Exposure points must include energy deposited i.e. 3rd column')

    # Offset the coordinates based on the resolution and the bounding box of the part
    exposurePoints[:, :2] -= part.boundingBox[:2] + resolution / 2
    expPointTrans = np.floor(exposurePoints[:, :2] / resolution).astype(int)

    # Get a bitmap object to work on
    bitmapSlice  = part.getBitmapSlice(z, resolution).astype(int)
    slice = np.zeros(bitmapSlice.shape)

    for i in range(len(expPointTrans)):
        slice[expPointTrans[i,1], expPointTrans[i,0]] += exposurePoints[i,2]

    slice /= resolution*resolution

    ax.imshow(slice, origin='lower', cmap='hot', interpolation='nearest')

    return fig, ax
