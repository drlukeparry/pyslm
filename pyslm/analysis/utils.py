from typing import List, Union
import numpy as np

from ..geometry import Layer, LayerGeometry, HatchGeometry, ContourGeometry, PointsGeometry, BuildStyle, Model, utils

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
    Calculates the total path length scanned by the laser across a single :class:`~pyslm.geometry.LayerGeometry` and is
    used for 1D geometry types i.e. :class:`HatchGeometry` and :class:`ContourGeometry`.

    :param layerGeom: The  :class:`LayerGeometry` to be measured
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

    return float(totalPathDist)


def getLayerJumpLength(layer: Layer) -> float:
    """
    Returns the total jump length across the :class:`Layer`

    :param layer: The :class:`Layer` to measure
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
    Returns the total path length across the :class:`~pyslm.geometry.Layer`

    :param layer: The :class:`~pyslm.geometry.Layer` to measure
    :return: Returns the path length for the layer
    """
    totalPathDistance = 0.0

    for layerGeom in layer.geometry:
        totalPathDistance += getLayerGeometryPathLength(layerGeom)

    return totalPathDistance


def getEffectiveLaserSpeed(bstyle: BuildStyle) -> float:
    """
    Returns the effective laser speed given a BuildStyle using the point distance and point exposure time
    :param bstyle: The :class:`~pyslm.geometry.BuildStyle` containing a valid point exposure

    :return: The laser speed [mm/s]

    """
    if bstyle.laserSpeed < 1e-7:
        if bstyle.pointExposureTime < 1e-7:
            raise ValueError('Build Style ({:s}) should have a valid point exposure time and point distance set'.format(bstyle.name))

        """
        Note a multiplier is currently needed as it is assumed point distance is in microns 
        and point exposure time in micro-seconds
        """
        return float(bstyle.pointDistance) / float(bstyle.pointExposureTime) * 1e3

    else:
        # Return the laser speed
        return bstyle.laserSpeed


def getLayerGeometryTime(layerGeometry: LayerGeometry, models: List[Model]) -> float:
    """
    Returns the total time taken to scan across a :class:`~pyslm.geometry.LayerGeometry`.

    :param layerGeometry: The :class:`~pyslm.geometry.LayerGeometry` to process
    :param models: A list of :class:`~pyslm.geometry.Model` which is used by the :class:`geometry.LayerGeometry`
    :return: The time taken to scan across the :class:`~pyslm.geometry.LayerGeometry`
    """
    # Find the build style

    bstyle = utils.getBuildStyleById(models, layerGeometry.mid, layerGeometry.bid)
    return getLayerGeometryPathLength(layerGeometry) / getEffectiveLaserSpeed(bstyle)


def getLayerTime(layer: Layer, models: List[Model]) -> float:
    """
    Returns the total time taken to scan a :class:`Layer`.

    :param layer: The layer to process
    :param models: The list of :class:`Model` containing the :class:`BuildStyle` used
    :return: The time taken to scan across the layer
    """
    layerTime = 0.0

    for layerGeom in layer.geometry:
        layerTime += getLayerGeometryTime(layerGeom, models)

    return layerTime
