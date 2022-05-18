import numpy as np
import numpy as np

from enum import Enum
import abc

from typing import Any, List, Optional, Tuple


class LaserMode:
    CW = 0
    """ Continuous Wave """

    Pulse = 1
    """ Pulsed mode (Default option) """

class Header:
    """
    The Header provides basic information about the machine build file, such as the name of the file
    (:attr:`filename`), version and the :attr:`zUnit` used for calculating the actual Layer z position in the machine.

    Typically the :attr:`zUnit` is set to 1000 :math:`\mu m` corresponding to a conversion factor from mm to microns.
    The :attr:`version` tuple is set corresponding to the chosen machine build format specification available in libSLM
    and what is compatible with the firmware of the SLM system.
    """
    def __init__(self):
        self.filename = ""
        self.version = (0,0)
        self.zUnit = 1000


class BuildStyle:
    """
    A :class:`BuildStyle` represents a collection of laser parameters used for scanning across a single
    :class:`LayerGeometry`. This consists of essential laser parameters including

    * :attr:`laserPower`,
    * :attr:`laserSpeed`,
    * :attr:`.pointDistance` and :attr:`pointExposureTime` - (required for pulsed mode lasers).

    A unique buildstyle id (:attr:`bid`) must be set within each
    :class:`Model` group that it is stored in and later assigned for each :class:`LayerGeometry` group. Additional,
    metadata can be set that for some Machine Build File formats are used such as :attr:`name` and :attr:`description`.
    For single and multi-laser systems, :attr:` laserId` corresponds with the parameter set associated with the laser.
    This offers opportunity to tailor the behavior of multiple beams applied to the same area such as providing
    a pre-heating or annealing exposure pass.

    .. note::
        For single laser systems the :attr:`laserId` is set to `1`
    """

    def __init__(self):
        self._name = ""
        self._description = ""
        self._bid = 0
        self._laserPower = 0.0
        self._laserSpeed = 0.0
        self._laserFocus = 0.0
        self._laserId = 1
        self._laserMode = 1
        self._pointDistance = 0
        self._pointExposureTime = 0
        self._pointDelay = 0
        self._jumpDelay = 0
        self._jumpSpeed = 0

    def __str__(self):
        str = "Build Style: (name: {:s}, id: {:d})\n".format(self._name, self._bid)
        str += "  laser power: {:.1f} W, laser speed: {:.1f}, laser id: {:d}\n".format(self._laserPower,
                                                                                       self._laserSpeed,
                                                                                       self._laserId)
        return str

    @property
    def bid(self) -> int:
        """
        A unique id used for each :class:`BuildStyle` object within each :class:`Model` that can be referenced by
        a :class:`LayerGeometry`
        """
        return self._bid

    @bid.setter
    def bid(self, bid):
        self._bid = bid

    @property
    def name(self) -> str:
        """ The name of the :class:`BuildStyle`"""
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = name

    @property
    def description(self) -> str:
        """
        The description of the :class:`BuildStyle`. This is usually not export by most machine build file formats
        but is useful to assign to help differentiate each build-style."""
        return self._description

    @description.setter
    def description(self, desc: str):
        self._description = desc

    @property
    def laserId(self) -> int:
        """ The ID of the laser beam used for the exposure. Typically set to `1` for single laser systems """
        return self._laserId

    @laserId.setter
    def laserId(self, value: int):
        self._laserId = value

    @property
    def laserMode(self) -> int:
        """
        Determines the laser mode to use via :class:`LaserMode` which is either continuous wave (CW) or
        pulsed (Pulsed) laser operation
        """
        return self._laserMode

    @laserMode.setter
    def laserMode(self, value):
        self._laserMode = value

    @property
    def laserPower(self) -> float:
        """ The average laser power of the exposure point """
        return self._laserPower

    @laserPower.setter
    def laserPower(self, laserPower: float):
        self._laserPower = laserPower

    @property
    def laserFocus(self) -> float:
        """ The laser focus position used, typically given as increment position """
        return self._laserFocus

    @laserFocus.setter
    def laserFocus(self, focus: int):
        self._laserFocus = focus

    @property
    def laserSpeed(self) -> float:
        """
        The laser speed typically expresses as :math:`mm/s.

        .. note::
            For pulsed laser mode systems this is typically ignored. """
        return self._laserSpeed

    @laserSpeed.setter
    def laserSpeed(self, laserSpeed: float):
        self._laserSpeed = laserSpeed

    @property
    def pointExposureTime(self) -> int:
        """
        The point exposure time (usually expressed as an integer :math:`\\mu s`).
        """
        return self._pointExposureTime

    @pointExposureTime.setter
    def pointExposureTime(self, pointExposureTime: int):
        self._pointExposureTime = pointExposureTime

    @property
    def pointDistance(self) -> int:
        """ The point exposure distance (usually expressed as an integer :math:`\\mu m`). """
        return self._pointDistance

    @pointDistance.setter
    def pointDistance(self, pointDistance: int):
        self._pointDistance = pointDistance

    @property
    def pointDelay(self) -> int:
        """
        The delay added between individual point exposure (usually expressed as an integer [:math:`\\mu s]`).
        This must be set to zero (default) if it is not explicitly used.
        """
        return self._pointDelay

    @pointDelay.setter
    def pointDelay(self, delay: int):
        self._pointDelay = delay

    @property
    def jumpDelay(self) -> int:
        """
        The jump delay between scan vectors (usually expressed as an integer [:math:`\mu s`]). This must be set to
        zero (default) if it is not explicitly used.
        """
        return self._jumpDelay

    @jumpDelay.setter
    def jumpDelay(self, delay: int):
        self._jumpDelay = delay

    @property
    def jumpSpeed(self) -> int:
        """
        The jump speed between scan vectors (usually expressed as an integer :math:`mm/s`). This must be set to
        zero (default) if it is not explicitly used.
        """
        return self._jumpSpeed

    @jumpSpeed.setter
    def jumpSpeed(self, speed: int):
        self._jumpSpeed = speed

    def setStyle(self, bid: int, focus: int, power: float,
                 pointExposureTime: int, pointExposureDistance: int, laserSpeed: Optional[float] = 0.0,
                 laserId: Optional[int] = 1, laserMode: Optional[LaserMode] = 1,
                 name: Optional[str] = "", description: Optional[str] = ""):

        self._bid = bid
        self._laserFocus = focus
        self._laserPower = power
        self._pointExposureTime = pointExposureTime
        self._pointDistance = pointExposureDistance
        self._laserSpeed = laserSpeed
        self._name = name
        self._description = description
        self._laserId = laserId
        self._laserMode = laserMode


class Model:
    """
    A Model represents a parametric group or in practice a part which contains a set unique
    and assignable :class:`BuildStyle` used a specific :class:`LayerGeometry`. The buildstyles are stored in
    :attr:`buildStyles`.

    Each Model must have a unique model-id (:attr:`mid`). Additionally, for some build formats, the top layer id
    (:attr:`topLayerId`) should correspond :attr:`Layer.id` value of the last layer's :class:`LayerGeometry` that
    uses this Model. It is recommended that :class:`ModelValidator` should
    be used to verify that all Models have a unique model-id and that the correct :attr:`topLayerId` is set, using the
    following methods

    .. code-block:: python

        pyslm.geometry.ModelValidator.validateModel(model)

    or

    .. code-block:: python

        models = [modelA, modelB]
        pyslm.geometry.ModelValidator.validateBuild(models, layer_list)

    Additional generic metadata can be stored for reference and describing the Model that are not necessarily
    used when exporting to build file including

    * :attr:`name` - Name of the Model
    * :attr:`buildStyleName` - Name of the Build Style Used for this Model (e.g. parameter set)
    * :attr:`buildStyleDescription`- Description of the Model Build Style set.
    """
    def __init__(self, mid: Optional[int] = 0):
        self._mid = mid
        self._topLayerId = 0
        self._name = ""
        self._buildStyleDescription = ""
        self._buildStyleName = ""
        self._buildStyles = []

    def __len__(self):
        return len(self.buildStyles)

    @property
    def buildStyles(self) -> List[BuildStyle]:
        """ The BuildStyles associated with this model """
        return self._buildStyles

    @buildStyles.setter
    def buildStyles(self, buildStyles: List[BuildStyle]):
        self._buildStyles = buildStyles

    @property
    def mid(self) -> int:
        """The unique id for this Model"""
        return self._mid

    @mid.setter
    def mid(self, mid: int):
        self._mid = mid

    @property
    def name(self) -> str:
        """ The name described by the model"""
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = name

    @property
    def topLayerId(self) -> int:
        """ The Top Layer of all Layer Geometries using this model"""
        return self._topLayerId

    @topLayerId.setter
    def topLayerId(self, topLayer: int):
        self._topLayerId = topLayer

    @property
    def buildStyleDescription(self):
        """ The description of the BuildStyles applied to the Model """
        return self._buildStyleDescription

    @buildStyleDescription.setter
    def buildStyleDescription(self, description: str):
        self._buildStyleDescription = description

    @property
    def buildStyleName(self) -> str:
        """ The BuildStyle name applied to the Model"""
        return self._buildStyleName

    @buildStyleName.setter
    def buildStyleName(self, name):
        self._buildStyleName = name


class LayerGeometryType(Enum):
    Invalid = 0
    Polygon = 1
    Hatch = 2
    Pnts = 3


class LayerGeometry(abc.ABC):
    """
    A Layer Geometry is the base class type used for storing a group of scan vectors or exposures. This is assigned a
    model id (:attr:`mid`) and a build style (:attr:`bid`).

    A set of coordinates are always available via :attr:`coords`. The coordinates should always be a numpy array that
    with a shape of Nx2 corresponding to the LayerGeometry type.
    """

    def __init__(self, mid: Optional[int] = 0, bid: Optional[int] = 0, coords: Optional[np.ndarray] = None):
        self._bid = bid
        self._mid = mid

        self._coords = np.array([])

        if coords:
            self._coords = coords

    def boundingBox(self) -> np.ndarray:
        return np.hstack([np.min(self.coords, axis=0), np.max(self.coords, axis=0)])

    @property
    def coords(self) -> np.ndarray:
        """ Coordinate data stored by the LayerGeometry."""
        return self._coords

    @coords.setter
    def coords(self, coordValues: np.ndarray):
        if coordValues.shape[-1] != 2:
            raise ValueError('Coordinates provided to layer geometry must have (X,Y) values only')

        self._coords = coordValues

    @property
    def mid(self) -> int:
        """
        The Model Id used for the LayerGeometry The Model Id refers to the collection of unique build-styles
        assigned to a part within a build.
        """
        return self._mid

    @mid.setter
    def mid(self, modelId: int):
        self._mid = modelId

    @property
    def bid(self) -> int:
        """
        The Build Style Id for the LayerGeometry. The Build Style Id refers to the collection of laser parameters
        used during scanning of scan vector group and must be available within the :class:`Model` used.
        """
        return self._bid

    @bid.setter
    def bid(self, buildStyleId: int):
        self._bid = buildStyleId

    @abc.abstractmethod
    def type(self) -> LayerGeometryType:
        """
        Returns which type the :class:`layerGeometry` is in the derived class.

        """
        return LayerGeometryType.Invalid


class HatchGeometry(LayerGeometry):
    """
    HatchGeometry represents a :class:`LayerGeometry` consisting of a series coordinates pairs :math:`[(x_0,y_0), (x_1,x_2)]`
    representing the start and end points of a scan vectors. This allows the point source to jump between scan vectors,
    unlike :class:`ContourGeometry`. Typically, the scan vectors are used for infilling large internal regions and
    are arranged parallel at a set distance from each other.
    """
    def __init__(self, mid: Optional[int] = 0, bid: Optional[int] = 0,
                       coords: Optional[np.ndarray] = None):

        super().__init__(mid, bid, coords)

    def __str__(self):
        return 'Hatch Geometry <bid, {:d}, mid, {:d}>'.format(self._bid, self._mid)

    def __len__(self):
        return self.numHatches()

    def type(self):
        return LayerGeometryType.Hatch

    def numHatches(self) -> int:
        """
        Number of hatches within this LayerGeometry
        """
        return self.coords.shape[0] / 2


class ContourGeometry(LayerGeometry):
    """
     ContourGeometry represents a :class:`LayerGeometry` consisting of a series of connected coordinates
     :math:`[(x_0,y_0), ..., (x_{n-1},x_{n-1})]` representing a continuous line. This allows the exposure point to
     efficiently follow a path without jumping,  unlike :class:`HatchGeometry`. Typically, the scan vectors are used for
     generated the boundaries of a part across a layer.
     """
    def __init__(self, mid: Optional[int] = 0, bid: Optional[int] = 0, coords: Optional[np.ndarray] = None):

        super().__init__(mid, bid, coords)

    def numContours(self) -> int:
        """
        Number of contour vectors in the geometry group.
        """
        return self.coords.shape[0] - 1

    def __len__(self):
        return self.numContours()

    def __str__(self):
        return 'Contour Geometry'

    def type(self):
        return LayerGeometryType.Polygon


class PointsGeometry(LayerGeometry):
    """
     PointsGeometry represents a :class:`LayerGeometry` consisting of a series of discrete or disconnected exposure
     points :math:`[(x_0,y_0), ..., (x_{n-1},x_{n-1})]` . This allows the user to prescribe very specific exposures
     to the bed, for very controlled and articulated scan styles. Typically, the exposure points are used either for
     lattice structures or support structures.

     .. warning::
        It is impracticable and inefficient to use these for large aerial regions.

     """
    def __init__(self, mid: Optional[int] = 0, bid: Optional[int] = 0, coords: Optional[np.ndarray] = None):

        super().__init__(mid, bid, coords)

    def numPoints(self) -> int:
        """ Number of individual point exposures within the geometry group"""
        return self.coords.shape[0]

    def __len__(self):
        return self.numPoints()

    def __str__(self):
        return 'Points Geometry'

    def type(self):
        return LayerGeometryType.Pnts


class ScanMode:
    """
    The scan mode is an enumeration class used to re-order all :class:`LayerGeometry` when accessing the entire collection
    from the :class:`Layer`.
    """
    Default = 0
    ContourFirst = 1
    HatchFirst = 2


class Layer:
    """
    Slice Layer is a simple class structure for containing a set of SLM :class:`LayerGeometry` including specific
    derivatives including: :class:`ContourGeometry`, :class:`HatchGeometry`, :class:`PointsGeometry` types stored in
    :attr:`geometry` and also the current slice or layer position in :attr:`z`.

    The layer z position is stored in an integer format to remove any specific rounding - typically this is specified
    as the number of microns.
    """

    def __init__(self, z: Optional[int] = 0, id: Optional[int] = 0):
        self._z = z
        self._id = id
        self._geometry = []
        self._name = ""
        self._layerFilePosition = 0

    @property
    def layerFilePosition(self):
        """ The position of the layer in the build file, when available. """
        return self._layerFilePosition

    def isLoaded(self) -> bool:
        return True

    @property
    def name(self) -> str:
        """ The name of the Layer"""
        return self._name

    @name.setter
    def name(self, name : str):
        self._name = name

    @property
    def layerId(self) -> int:
        """
        The layer id for the Layer. This corresponds to a position in z based on a uniform layer thickness defined
        in the header of the machine build file (:attr:`Header.zUnit`) """
        return self._id

    @layerId.setter
    def layerId(self, id: int):
        self._id = id

    @property
    def z(self) -> int:
        """
        The Z Position of the :class:`Layer` is given as an integer to ensure that no rounding errors are given to the
        slm systen. Under most situations this should correspond as the product of the layer id (:attr:`Layer.layerId`)
        and the zUnit - layer thickness (:attr:`Header.zUnit`).
        """
        return self._z

    @z.setter
    def z(self, z: int):
        self._z = z

    def __len__(self):
        return len(self._geometry)

    def __str__(self):
        return 'Layer <z = {:.3f}>'.format(self._z)

    def appendGeometry(self, geom: LayerGeometry):
        """
        Complimentary method to match libSLM API. This appends any :class:`LayerGeometry` and derived classes into the
        Layer in sequential order.

        :param geom: The LayerGeometry to add to the layer
        """

        self._geometry.append(geom)

    def getGeometry(self, scanMode: ScanMode = ScanMode.Default) -> List[Any]:
        """
        Contains all the layer geometry groups in the layer.
        """
        geoms = []

        if scanMode is ScanMode.ContourFirst:
            geoms += self.getContourGeometry()
            geoms += self.getHatchGeometry()
            geoms += self.getPointsGeometry()
        elif scanMode is ScanMode.HatchFirst:
            geoms += self.getHatchGeometry()
            geoms += self.getContourGeometry()
            geoms += self.getPointsGeometry()
        else:
            geoms = self._geometry

        return geoms

    @property
    def geometry(self) -> List[Any]:
        """
        :class:`LayerGeometry` sections that are stored in the layer.
        """

        return self._geometry

    @geometry.setter
    def geometry(self, geoms: List[LayerGeometry]):
        self._geometry = geoms

    def getContourGeometry(self) -> List[HatchGeometry]:
        """
        Returns a list of all :class:`ContourGeometry` stored in the layer.
        """

        geoms = []
        for geom in self._geometry:
            if isinstance(geom, ContourGeometry):
                geoms.append(geom)

        return geoms

    def getHatchGeometry(self) -> List[HatchGeometry]:
        """
        Returns a list of all :class:`HatchGeometry` stored in the layer.
        """

        geoms = []
        for geom in self._geometry:
            if isinstance(geom, HatchGeometry):
                geoms.append(geom)

        return geoms

    def getPointsGeometry(self) -> List[PointsGeometry]:
        """
        Returns a list of all :class:`PointsGeometry` stored in the layer.
        """
        geoms = []
        for geom in self._geometry:
            if isinstance(geom, PointsGeometry):
                geoms.append(geom)

        return geoms

