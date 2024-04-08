from typing import List, Optional, Union
import numpy as np

from ..geometry import Layer, LayerGeometry, HatchGeometry, ContourGeometry, PointsGeometry, BuildStyle, Model, utils

def getLayerGeometryJumpDistance(layerGeom: LayerGeometry) -> float:
    """
    Calculates the jump distance of the laser between adjacent exposure points and hatches, principally used for
    :class:`~pyslm.geometry.HatchGeometry` and :class:`~pyslm.geometry.PointsGeometry`.

    :param layerGeom: The `LayerGeometry` to calculate the jump distance
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
    used for 1D geometry types i.e. :class:`~pyslm.geometry.HatchGeometry` and :class:`~pyslm.geometry.ContourGeometry`.

    :param layerGeom: The `LayerGeometry` path length to be measured
    :return: The total path length of the target geometry
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

    if isinstance(layerGeom, PointsGeometry):
        raise Exception('Cannot pass a PointsGeometry to calculate the total path length')

    return float(totalPathDist)


def getLayerGeomTotalJumps(layerGeom: LayerGeometry) -> int:
    """
    Returns the total number of scan jumps across a :class:`~pyslm.geometry.LayerGeometry`.

    .. note::
        It is assumed that there is a single jump for each ContourGeometry and PointsGeometry

    :param layerGeom: The :class:`~pyslm.geometry.LayerGeometry` to measure
    :return: Returns the total number of jumps
    """
    coords = layerGeom.coords

    numJumps = 0

    if isinstance(layerGeom, ContourGeometry):
        numJumps = 1

    if isinstance(layerGeom, HatchGeometry):
        numJumps = coords.shape[0] / 2

    if isinstance(layerGeom, PointsGeometry):
        numJumps = 1

    return numJumps


def getIntraLayerGeometryJumpLength(layer: Layer) -> float:
    """
    Returns the intra-layer geometry jump length across the entire :class:`~pyslm.geometry.Layer`

    :param layer: The :class:`~pyslm.geometry.Layer` to measure
    :return: Returns the jump path length for the layer
    """
    intraJumpDistance = 0.0

    lastCoord = None

    for layerGeom in layer.geometry:
        if lastCoord is not None:
            delta = lastCoord - layerGeom.coords[0, :]
            intraJumpDistance += np.hypot(delta[0], delta[1])
            lastCoord = None

        lastCoord = (layerGeom.coords[0, :])

    return intraJumpDistance


def getLayerJumpLength(layer: Layer) -> float:
    """
    Returns the total jump length across the :class:`~pyslm.geometry.Layer`

    :param layer: The `Layer` to measure the jump length
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

        lastCoord = (layerGeom.coords[0, :])

    totalJumpDistance += intraJumpDistance

    return totalJumpDistance


def getLayerPathLength(layer: Layer) -> float:
    """
    Returns the total path length across the :class:`~pyslm.geometry.Layer`

    :param layer: The `Layer` to measure
    :return: Returns the path length for the layer
    """
    totalPathDistance = 0.0

    for layerGeom in layer.geometry:
        totalPathDistance += getLayerGeometryPathLength(layerGeom)

    return totalPathDistance


def getEffectiveLaserSpeed(bstyle: BuildStyle) -> float:
    """
    Returns the effective laser speed given a :class:`~pyslm.geometry.BuildStyle` using the point distance and point
    exposure time. This includes the dwell time between pulse :attr:`~pyslm.geometry.BuildStyle.jumpDelay` and the
    :attr:`BuildStyle.jumpSpeed` assigned to each individual :class:`~pyslm.geometry.BuildStyle`.

    :param bstyle: The `BuildStyle` containing a valid point exposure time and point distances
    :return: The laser scan speed [`mm/s`]
    """

    if bstyle.laserSpeed < 1e-7:

        if bstyle.pointExposureTime < 1e-7:
            raise ValueError('Build Style ({:s}) should have a valid point exposure time and point distance set'.format(bstyle.name))

        """
        Note a multiplier is currently needed as it is assumed point distance is in microns 
        and point exposure time in micro-seconds
        """

        # Calculate the point jump time based on the jump speed [mm/s] - typically around 5000 mm/s
        pointJumpTime = 0.0

        if bstyle.jumpSpeed > 1e-7:
            pointJumpTime = float(bstyle.pointDistance) * 1e-3 / bstyle.jumpSpeed

        # Point distance [microns), point exposure time (mu s)
        return float(bstyle.pointDistance) * 1e-3 / (float(bstyle.pointExposureTime)*1e-6 + pointJumpTime + float(bstyle.pointDelay) * 1e-6)

    else:
        # Return the laser speed
        return bstyle.laserSpeed


def getLayerGeometryTime(layerGeom: LayerGeometry, models: List[Model],
                         includeJumpTime: Optional[bool] = False,
                         jumpSpeed: Optional[float] = 5000.0,
                         jumpDelay: Optional[float] = 0.0) -> float:
    """
    Returns the total time taken to scan across a :class:`~pyslm.geometry.LayerGeometry`.

    :param layerGeom: The `LayerGeometry` to process
    :param models: A list of `Model` which is used by the `LayerGeometry`
    :param includeJumpTime: Include the jump time between scan vectors in the calculation (default = `False`)
    :param jumpSpeed: The default jump speed used to scan between scan vectors [`mm/s`]
    :param jumpDelay: The default jump delay used to scan between scan vectors [`mu s`]

    :return: The time taken to scan across the `LayerGeometry`
    """

    # Find the build style
    scanTime = 0.0
    totalJumpTime = 0.0

    bstyle = utils.getBuildStyleById(models, layerGeom.mid, layerGeom.bid)

    if isinstance(layerGeom, HatchGeometry) or isinstance(layerGeom, ContourGeometry):
        scanTime = getLayerGeometryPathLength(layerGeom) / getEffectiveLaserSpeed(bstyle)
    elif isinstance(layerGeom, PointsGeometry):
        scanTime = layerGeom.coords.shape[0] * float(bstyle.pointExposureTime) * 1e-6
    else:
        raise Exception('Invalid LayerGeometry object passed as an argument')

    if includeJumpTime:

        if float(bstyle.jumpSpeed) > 1e-6:

            # Add distance to transverse across each scan vector (if applicable)
            totalJumpTime = getLayerGeometryJumpDistance(layerGeom) / float(bstyle.jumpSpeed)
        else:
            # Add distance to transverse across each scan vector and assume a constant jump speed
            totalJumpTime = getLayerGeometryJumpDistance(layerGeom) / float(jumpSpeed)

        # Add a jump delay (optional) between scan vectors (if applicable)

        if bstyle.jumpDelay > 1e-6:
            totalJumpTime += getLayerGeomTotalJumps(layerGeom) * float(bstyle.jumpDelay) * 1e-6
        else:
            totalJumpTime += getLayerGeomTotalJumps(layerGeom) * float(jumpDelay) * 1e-6

    return scanTime + totalJumpTime


def getLayerTime(layer: Layer, models: List[Model],
                 includeJumpTime: Optional[bool] = True,
                 laserJumpSpeed: Optional[float] = 5000) -> float:
    """
    Returns the total time taken to scan across a :class:`~pyslm.geometry.Layer`. This includes the additional dwell time
    laser pulses :attr:`BuildStyle.jumpDelay` and the jump time between both scan vectors and consecutive
    :class:`~pyslm.geometry.LayerGeometry` groups. The time taken between adjacent scan vectors and layer geometries is assumed to have
    an instantaneous acceleration at constant velocity.

    :param layer: The `Layer` to process
    :param models: The `Model` list containing the :class:`~pyslm.geometry.BuildStyle` set used
    :param includeJumpTime: Include the jump time between and within each `LayerGeometry`
    :param laserJumpSpeed: The default laser jump speed used whilst scanning between layer geometry [`mm/s`]

    :return: The time taken to scan across the layer
    """
    layerTime = 0.0

    for layerGeom in layer.geometry:
        layerTime += getLayerGeometryTime(layerGeom, models, includeJumpTime)

    if includeJumpTime:
        layerTime += getIntraLayerGeometryJumpLength(layer) / laserJumpSpeed

        # Include jump times between layer geometry groups
        intraGeomJumpDelayTime = 0.0

        # Include the jump time between layer geometry groups
        for layerGeom in layer.geometry:
            bstyle = utils.getBuildStyleById(models, layerGeom.mid, layerGeom.bid)
            intraGeomJumpDelayTime += bstyle.jumpDelay

        layerTime += intraGeomJumpDelayTime

    return layerTime
