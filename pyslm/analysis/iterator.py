import logging
from typing import Any, List, Optional, Tuple, Union
from abc import ABC

import numpy as np

from ..geometry import Layer, LayerGeometry, HatchGeometry, ContourGeometry, PointsGeometry, BuildStyle, Model
from ..geometry import utils as geomUtils
from .utils import *


class LaserState:
    """
    The LaserState Class is a simple structure used for storing the state of the current exposure point at
    time :math:`t` including the position and the active :class:`BuildStyle` and whether it is active/firing which
    may happen during the layer dwell time.
    """
    def __init__(self):
        self.position = (0, 0)
        self.buildStyle = None
        self.isActive = False


class TimeNode:
    """
    The TimeNode class provides a structure for storing the pre-calculated time of a Layer or LayerGeometry structure,
    which is stored in :attr:`TimeNode.time`. The TimeNode is constructed with  both references to children and parent
    nodes within the functional scope, which build a parsable tree structure. The references are to other TimeNodes
    and static and changes are not dynamically propagated, therefore caution is advised that the entire Cache tree
    is updated after a known change.
    """
    def __init__(self, parent=None, id: Optional[int]=0, value: Optional[Any] = None):

        self.parent = parent
        self.id = id
        self.value = value
        self.time = 0.0
        self.children: List[TimeNode] = []

    def getChildrenTime(self) -> float:
        """
        Get the total time taken by the children

        :return: The total time taken

        """
        time = 0.0
        for child in self.children:
            time += child.time
            time += child.getChildrenTime()
        return time


class ScanVectorIterator:
    """
    ScanVectorIterator provides an iterator that will traverse across every scan vector (linear) across both hatch
    and contour scan vectors for all layers passed into the constructor.
    """
    def  __init__(self,  layers: List[Layer]):

        self._vectors = []
        self._layers = layers

        self._layerIt = 0
        self._layerScanVecIt = 0
        self._layerScanVectors = []

    @property
    def vectors(self):
        return self._vectors

    def initialise(self):
        """
        Initialises the iterator based on the input of collection of layers.

        :return: The list of reshaped vectors
        """
        layerVecs = []
        for layer in self._layers:
            layerVecs += self.getLayerVectors(layer)

        return self.reshapeVectors(layerVecs)

    @staticmethod
    def reshapeVectors(vectorList: np.ndarray):
        return np.vstack(vectorList).reshape([-1, 2, 2])

    @staticmethod
    def getLayerVectors(layer: Layer) -> List[np.ndarray]:
        """
        Returns a list of scan vectors groups from a :class:`Layer`.

        :param layer: The Layer to obtain the scan vectors from
        :return:  The scan vector list
        """

        layerVecs = []

        for geom in layer.geometry:
            if isinstance(geom, ContourGeometry):
                """ Trick to change contour vectors to normal lines """
                layerVecs.append(np.tile(geom.coords, (1, 2)).reshape(-1, 2)[1:-1])
            elif isinstance(geom, HatchGeometry):
                layerVecs.append(geom.coords)

        return layerVecs

    def __iter__(self):

        self._layerIt = 0
        self._layerScanVecIt = 0

        return self

    def __next__(self):

        if self._layerScanVecIt < len( self._layerScanVectors):
            scanVector = self._layerScanVectors[self._layerScanVecIt]
            self._layerScanVecIt += 1
            return scanVector
        else:
            # New layer
            if self._layerIt < len(self._layers):
                layerVectors = self.getLayerVectors(self._layers[self._layerIt])
                self._layerScanVectors = self.reshapeVectors(layerVectors)
                self._layerScanVecIt = 0
                self._layerIt += 1
                return self.__next__()
            else:
                raise StopIteration


class Iterator(ABC):
    """
    Basic Iterator which parses through both :class:`Layer` and :class:`LayerGeometry` groups and incrementally goes
    through the geometry based on time values generated in conjunction with the associated :class:`Model`.
    """
    def  __init__(self, models: List[Model], layers: List[Layer]):

        self._time = 0.0
        self._layerGeomTime = 0.0
        self._layerInc = 0
        self._layerGeomInc = 0

        self._layers = layers
        self._models = models
        self._recoaterTime = 0.0

        self._pointer = None

        # Variables for cache
        self._cacheValid = False
        self._cache = []
        self._tree = None

    @property
    def time(self) -> float:
        """ The current time [s] of the iterator (read only) """
        return self._time

    @property
    def currentLayerGeometryTime(self) -> float:
        """ The current start time of the active :class:`LayerGeometry` """
        return self._layerGeomTime

    @property
    def tree(self) -> TimeNode:
        """
        A tree of :class:`TimeNode` for the entire build
        """
        if not self._cacheValid:

            self._generateCache()

        return self._tree

    def _generateCache(self) -> None:

        logging.info('Generating  Iterator TimeNode Cache Tree')
        self._tree = TimeNode()

        for layerId, layer in enumerate(self.layers):

            # Create the layer
            layerNode = TimeNode(self._tree, id=layerId, value=layer)
            self._tree.children.append(layerNode)

            for layerGeomId, layerGeom in enumerate(layer.geometry):

                geomNode = TimeNode(layerNode, id=layerGeomId, value=layerGeom)
                geomNode.time = getLayerGeometryTime(layerGeom, self._models)

                layerNode.children.append(geomNode)

        self._cacheValid = True

    @property
    def dwellTime(self) -> float:
        """
        The total layer dwell time [s]. This can be re-implemented in a derived class to be a more complex function or
        provide additional variables which control the overall dwell time per layers such as the number of re-coats
        and the re-coating speed, often specified by the machine.
        """
        return self.recoaterTime

    @property
    def recoaterTime(self) -> float:
        """ The re-coater time [s] added after the layer """
        return self._recoaterTime

    @recoaterTime.setter
    def recoaterTime(self, time: float):
        self._recoaterTime = time

    @property
    def layers(self) -> List[Layer]:
        """ A :class:`Layer` list to be processed by the iterator """
        return self._layers

    @layers.setter
    def layers(self, layers: List[Layer]):
        self._layers = layers
        self._cacheValid = False

    @property
    def models(self) -> List[Model]:
        """ A :class:`Model` list to be processed by the iterator """
        """ The models of the iterator"""
        return self._models

    @models.setter
    def models(self, models: List[Model]):
        self._models = models
        self._cacheValid = False

    def getBuildTime(self) -> float:
        """
        Gets the total build-time of the entire list of layers including additional dwell time between layers.
        This function simply parses through the entire :class:`TimeNode` tree and adds on the dwell time per layer.
        """

        time = 0.0

        for lNode in self.tree.children:
            time += lNode.getChildrenTime()
            time += self.dwellTime

        return time

    def getLayerGeomTime(self, layerId: int, layerGeomId: int ) -> float:
        """
        Gets the total time for each :class:`LayerGeometry` given a unique a :class:`Layer` index and a
        :class:`LayerGeometry` index.

        :param layerId: The layer index in the list
        :param layerGeomId: The layer geometry index within the :class:`Layer`
        :return: The time for the LayerGeometry
        """
        return self.tree.children[layerId].children[layerGeomId].time

    def getLayerTime(self, layerId: int) -> float:
        """
        Gets the total time for a :class:`Layer` given a unique a :class:`Layer` index

        :param layerId: The layer index in the list
        :return: The time fo the layer
        """
        return self.tree.children[layerId].getChildrenTime()

    def getTimeByLayerGeometryId(self, layerId: int, layerGeomId: int) -> float:
        """
        Gets the current time for a :class:`LayerGeometry` given a unique a :class:`Layer` index and a
        :class:`LayerGeometry` index.

        :param layerId: The layer index in the list
        :param layerGeomId: The layer geometry index within the :class:`Layer`
        :return: The time for the LayerGeometry
        """
        time = 0.0

        if layerId >= len(self.layers):
            raise ValueError('Layer Id ({:d}) is not in the correct range'.format(layerId))

        layerNode = self.tree.children[layerId]

        if layerGeomId >= len(layerNode.children):
            raise ValueError('Layer Geom Id ({:d}) is not in the correct range'.format(layerGeomId))

        for lNode in self.tree.children[:layerId]:
            time += lNode.getChildrenTime()
            time += self.dwellTime

        for layerGeomNode in layerNode.children[:layerGeomId]:
            time += layerGeomNode.time

        return time

    def getLayerGeometryNodeByTime(self, time: float) -> TimeNode:
        """
        Gets the :class:`TimeNode` for a :class:`LayerGeometry` given a time

        :param time: The time
        :return: The LayerGeometry TimeNode
        """
        buildTime = layerEndTime = 0.0

        for layerId, layerNode in enumerate(self.tree.children):

            layerEndTime += layerNode.getChildrenTime() + self.dwellTime

            if buildTime < time < layerEndTime:
                # The time of the layer geometry is in the layer

                layerGeomEndTime = buildTime

                for layerGeomId, layerGeomNode in enumerate(layerNode.children):
                    layerGeomEndTime += layerGeomNode.time

                    if buildTime < time < layerGeomEndTime:
                        return layerGeomNode

                    buildTime = layerGeomEndTime

                return layerNode

            buildTime = layerEndTime

        return None

    def getLayerGeometryByTime(self, time: float) -> LayerGeometry:

        node = self.getLayerGeometryNodeByTime(time)
        return node.value if node else None

    def getLayerGeometryIdByTime(self, time: float) -> int:

        node = self.getLayerGeometryNodeByTime(time)
        return node.id if node else None

    def getLayerNodeByTime(self, time: float) -> Union[TimeNode, None]:

        layerTime = 0.0

        for layerId, layerNode in enumerate(self.tree.children):

            layerTimeStart = layerTime
            layerTime += layerNode.getChildrenTime() + self.dwellTime

            if layerTimeStart < time < layerTime:
                return layerNode

        return None

    def getLayerByTime(self, time: float) -> Layer:
        """
        Gets the current :class:`Layer` based on the search time

        :param time: The time for locating the :class:`Layer`
        :return: The Layer at time t
        """
        node = self.getLayerNodeByTime(time)
        return node.value if node else None

    def getLayerIdByTime(self, time: float) -> int:
        """
        Gets the current :class:`Layer` index based on the search time

        :param time: The time for finding the layer
        :return: The Layer Index in the list if found
        """
        node = self.getLayerNodeByTime(time)
        return node.id if node else None

    def getTimeByLayerId(self, layerIdx: int) -> float:
        """
        Gets the current time in the Build based on the layer index.

        :param layerIdx: The layer index
        :return: The time at the start of the Layer
        """
        layerTime = 0.0

        if layerIdx >= len(self.layers):
            raise ValueError('Layer Id ({:d}) is not in the correct range'.format(layerIdx))

        for lNode in self.tree.children[:layerIdx]:
            layerTime += lNode.getChildrenTime()
            layerTime += self.dwellTime

        return layerTime

    def getCurrentLayerGeometry(self) -> LayerGeometry:
        """
        Gets the current :class:`LayerGeometry` for the current iteration

        :return: The active LayerGeometry
        """
        return self.getCurrentLayer().geometry[self._layerGeomInc]

    def getCurrentLayer(self) -> Layer:
        """
        Gets the current layer of the iterator

        :return: The current layer
        """
        return self.layers[self._layerInc]

    def seekByLayerGeometry(self, layerId: int, layerGeomId: int) -> None:
        """
        Instructs the iterator to seek ahead in time to the specific :class:`LayerGeometry`

        :param layerId:  The index of the layer within the list of layers provided to the iterator
        :param layerGeomId:  The index of the layer geometry within the specified layer
        """
        self._time = self._layerGeomTime = self.getTimeByLayerGeometryId(layerId, layerGeomId)
        self._layerInc = layerId
        self._layerGeomInc = layerGeomId

    def seekByLayer(self, layerId: int) -> None:
        """
        Instructs the iterator to seek ahead in time to the specific :class:`Layer`

        :param layerId: The index of the layer within the list of layers provided to the iterator
        """
        self._time = self._layerGeomTime = self.getTimeByLayerId(layerId)
        self._layerInc = layerId
        self._layerGeomInc = 0

    def seek(self, time: float) -> None:
        """
        Instructs the iterator to seek ahead to a specific time.

        :param time: The time to seek to
        """
        layerGeomNode = self.getLayerGeometryNodeByTime(time)

        if not layerGeomNode:
            raise Exception('Seek time {.3f} is not within the build time'.format(time))

        self._time = time
        self._layerInc = layerGeomNode.parent.id
        self._layerGeomInc = layerGeomNode.id

    def __iter__(self):
        self._time = 0.0
        self._layerGeomTime = 0.0
        self._layerInc = 0
        self._layerGeomInc = 0

        return self

    def __next__(self):

        layer = self.layers[self._layerInc]

        if self._layerGeomInc < len(layer.geometry) - 1:
            layerGeom = layer.geometry[self._layerGeomInc]

            # Update the variables
            self._layerGeomInc += 1
            self._layerGeomTime = self.getTimeByLayerGeometryId(self._layerInc, self._layerGeomInc)
            return layerGeom
        else:
            # new layer
            if self._layerInc < len(self.layers) -1:

                self._layerInc += 1
                self._layerGeomInc = 0
                layer = self.layers[self._layerInc]
                layerGeom = layer.geometry[self._layerGeomInc]
                self._layerGeomTime = self.getTimeByLayerGeometryId(self._layerInc, self._layerGeomInc)

                return layerGeom
            else:
                raise StopIteration


class LayerGeometryIterator(Iterator):

    def __init__(self, models: List[Model], layers: List[Layer]):
        super().__init__(models, layers)

    def __iter__(self):
        super().__iter__()
        return self


class ScanIterator(Iterator):
    """
    The Scan Iterator class provides a  method to iterate at a variable :attr:`timestep` across a BuildFile
    consisting of list of :class:`Layer` and :class:`Model` provided as the input. Typically this is used in
    numerical simulation of powder-bed fusion processes and also its temporal visualisation. Properties include the
    current position are available via :meth:`getCurrentLaserPosition` and the current laser parameters in
    :meth:`getCurrentBuildStyle` and if the laser is currently active :meth:`isLaserOn`.

    .. note::
        The Iterator classes *assumes* that the laser position during rastering is linearly interpolated across each scan
        vector, based on the :attr:`timestep`, which can be modulated during the iterator.

    ScanIterator builds upon  :class:`Iterator`  and utilises the TimeTree cache generated for each :class:`Layer`
    and its set of :class:`LayerGeometry` objects respectively. If the current :attr:`time` is within the current
    :class:`LayerGeometry` the current point is interpolated across the individual scan vectors depending on its type in
    :meth:`getPointInLayerGeometry` using the current BuildStyle associated with the LayerGeometry.
    """
    def __init__(self, models: List[Model], layers: List[Layer]):
        super().__init__(models, layers)

        self._timestep: float = 1e-3
        self._scanId = 0

        self._scanCache = 0

    @property
    def timestep(self) -> float:
        """ The current timestep [s] for the Scan Iterator """
        return self._timestep

    @timestep.setter
    def timestep(self, value: float):
        self._timestep = value

    def isLaserOn(self) -> bool:
        """
        Determines if the laser is currently on.

        :return: Status of the laser
        """
        layerStartTime = self._layerGeomTime
        layerEndTime = layerStartTime + self.getLayerGeomTime(self._layerInc, self._layerGeomInc)
        if layerStartTime < self._time < layerEndTime:
            return True
        else:
            return False

    def getPointInLayerGeometry(self, timeOffset: object, layerGeom: object) -> Tuple[float, float]:
        """
        Interpolates the current laser point based on a `timeoffset` from the start of the selected
        :class:`LayerGeometry`. It iterates across each scan vector based on a total distance accumulated and locates
        the scan vector to interpolate the position.

        :param timeOffset: Time offset within the LayerGeometry
        :param layerGeom: The LayerGeometry to interpolate the laser point across

        :return:  The current position of the laser of the time
        """
        buildStyle = geomUtils.getBuildStyleById(self.models, layerGeom.mid, layerGeom.bid)

        laserVelocity = getEffectiveLaserSpeed(buildStyle)

        if isinstance(layerGeom, ContourGeometry):
            offsetDist = timeOffset  * laserVelocity

            # Find the distances for all the scan vectors
            delta = np.diff(layerGeom.coords, axis=0)
            dist = np.hypot(delta[:,0], delta[:,1])
            cumDist = np.cumsum(dist)
            cumDist2 = np.insert(cumDist, 0,0)

            if offsetDist > cumDist2[-1]:
                raise Exception('Error offset distance > cumDist {:.3f}, {:.3f}'.format(offsetDist, cumDist2[-1]))

            id = 0
            for i, vec in enumerate(cumDist2):
                if offsetDist < vec:
                    id = i
                    break

            linearOffset = (offsetDist - cumDist2[id-1]) / dist[id-1]
            point = layerGeom.coords[id-1] + delta[id-1] * linearOffset

            # note scipy interpolate works
            """
            from scipy import interpolate
            x = cumDist2
            y = layerGeom.coords
            f = interpolate.interp1d(x,y, axis=0)
            """
        elif isinstance(layerGeom, HatchGeometry):

            offsetDist = timeOffset * laserVelocity

            coords = layerGeom.coords.reshape(-1,1).reshape(-1,2,2)
            delta = np.diff(coords, axis=1).reshape(-1,2)
            dist = np.hypot(delta[:, 0], delta[:, 1])
            cumDist = np.cumsum(dist)
            cumDist2 = np.insert(cumDist, 0, 0)


            id = 0
            for i, vec in enumerate(cumDist2):
                if offsetDist < vec:
                    id = i
                    break

            linearOffset = (offsetDist - cumDist2[id-1]) / dist[id-1]
            point = coords[id-1][0] + delta[id-1] * linearOffset

        return point

    def getCurrentLaserPosition(self) -> Tuple[float, float]:
        """
        Returns the current position of the point exposure at the current :attr:`time`.

        :return: A tuple representing the exposure point :math:`(x,y)`
        """
        layerGeom = self.getCurrentLayerGeometry()
        offsetTime = self.time - self.currentLayerGeometryTime

        point = self.getPointInLayerGeometry(offsetTime, layerGeom)

        return point[0], point[1]

    def getCurrentBuildStyle(self) -> BuildStyle:
        """
        Gets the current :class:`BuildStyle`

        :return: The current BuildStyle
        """
        layerGeom = self.getCurrentLayerGeometry()

        return geomUtils.getBuildStyleById(self.models, layerGeom.mid, layerGeom.bid)

    def __iter__(self):
        super().__iter__()
        return self

    def __next__(self):

        if self._time < self._layerGeomTime:
            position = np.array((0,0))
        else:
            position = self.getCurrentLaserPosition()

        # Increment the timestep
        layerGeomTime = self.getLayerGeomTime(self._layerInc, self._layerGeomInc)

        self._time += self._timestep

        endTime = self._layerGeomTime + layerGeomTime

        if self._time > endTime:
            # Iterate to the next Layer Geometry (and) Layer
            super().__next__()

        return position[0], position[1], self._time - self._timestep
