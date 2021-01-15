from typing import List
import numpy as np

from ..geometry import Layer, LayerGeometry, HatchGeometry, ContourGeometry, PointsGeometry, BuildStyle, Model


def getLayerGeometryJumpDistance(layerGeom: LayerGeometry) -> float:
    """
    Calculates the jump distance of the laser between adjacent exposure points and hatches, principally used for
    :class:`HatchGeometry` and :class:`PointsGeometry`.

    :param layerGeom: Layer Geometry to find
    :return: The total path length of the layer geometry
    """
    totalJumpDist = 0.0

    if isinstance(layerGeom, HatchGeometry):
        # jump calculation estimation
        coords = layerGeom.coords[1:-1].reshape(-1, 2, 2)
        delta = np.diff(coords, axis=1).reshape(-1, 2)
        lineDist = np.hypot(delta[:, 0], delta[:, 1])
        totalJumpDist = np.sum(lineDist)

    if isinstance(layerGeom, PointsGeometry):
        # jump calculation estimation
        delta = np.diff(layerGeom.coords, axis=0)
        lineDist = np.hypot(delta[:, 0], delta[:, 1])
        totalJumpDist = np.sum(lineDist)

    return totalJumpDist


def getLayerGeometryPathLength(layerGeom: LayerGeometry) -> float:
    """
    Calculates the total path length scanned by the laser across a single :class:`LayerGeometry` and is used for 1D
    geometry types i.e. :class:`HatchGeometry` and :class:`ContourGeometry`.

    :param layerGeom: The Layer Geometry to be measured

    :return: The total path length of the line
    """
    # distance calculation
    coords = layerGeom.coords

    if isinstance(layerGeom, ContourGeometry):
        delta = np.diff(coords, axis=0)
        lineDist = np.hypot(delta[:, 0], delta[:, 1])
        totalPathDist = np.sum(lineDist)

    if isinstance(layerGeom, HatchGeometry):
        coords = coords.reshape(-1, 2, 2)
        delta = np.diff(coords, axis=1).reshape(-1, 2)
        lineDist = np.hypot(delta[:, 0], delta[:, 1])
        totalPathDist = np.sum(lineDist)

    return totalPathDist


def getLayerJumpLength(layer: Layer) -> float:
    """
    Returns the total jump length across the :class:`Layer`

    :param layer: The Layer to measure
    :return: Returns the path length for the layer
    """
    totalJumpDistance = 0.0
    intraJumpDistance = 0.0

    lastCoord = None

    for layerGeom in layer.geometry:
        totalJumpDistance += getLayerGeometryJumpDistance(layerGeom)

        if lastCoord is not None:
            delta = lastCoord - layerGeom.coords[0, :]
            intraJumpDistance += np.hypot(delta[0], delta[1])
            lastCoord = None

        lastCoord = (layerGeom.coords[0,:])

    totalJumpDistance += intraJumpDistance

    return totalJumpDistance


def getLayerPathLength(layer: Layer) -> float:
    """
    Returns the total path length across the :class:`Layer`

    :param layer: The Layer to measure
    :return: Returns the path length for the layer
    """
    totalPathDistance = 0.0

    for layerGeom in layer.geometry:
        totalPathDistance += getLayerGeometryPathLength(layerGeom)

    return totalPathDistance


def getLayerGeometryTime(layerGeometry: LayerGeometry, model: Model) -> float:
    """
    Returns the total time taken to scan across a :class:`LayerGeometry`.

    :param layerGeometry:  The layer geometry to process
    :param model: The model containing the buildstyles which is used by the layer geometry

    :return: The time taken to scan across the layer geometry
    """
    # Find the build style
    # [TODO] use getLayerGeometrybyId in libSLM
    bstyle = next(x for x in model.buildStyles if x.bid == layerGeometry.bid)

    return getLayerGeometryPathLength(layerGeometry) / bstyle.laserSpeed

def getLayerTime(layer : Layer, model: Model) -> float:
    """
    Returns the total time taken to scan a :class:`Layer`.

    :param layer: The layer to process
    :param model: The model containing the buildstyle used

    :return The time taken to scan across the layer
    """
    layerTime = 0.0

    for layerGeom in layer.geometry:
        layerTime += getLayerGeometryTime(layerGeom, model)

    return layerTime
