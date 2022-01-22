import abc
import time
from typing import Any, List, Optional, Tuple, Union
import logging

import numpy as np

from pyslm import pyclipper

from shapely.geometry import Polygon as ShapelyPolygon
from .sorting import AlternateSort, BaseSort, LinearSort
from ..geometry import Layer, Model, LayerGeometry, ContourGeometry, HatchGeometry, PointsGeometry


def getExposurePoints(layer: Layer, models: List[Model], includePowerDeposited: bool = True):
    """
    A utility method to return a list of exposure points given a :class:`Layer` with an associated
    :class:`Model` which contains the :class:`BuildStyle` that provides the point
    exposure distance or an effective laser speed to spatially discretise the scan vectors into a series of points.
    If the optional parameter `includePowerDeposited` is set to True, the laser power deposited in included.

    .. note::
        The :attr:`BuildStyle.pointDistance` parameter must be set or this method will fail.

    :param layer: The layer to process
    :param models: A list of models containing buildstyles which are referenced with the layer's :class:`LayerGeometry`
    :param includePowerDeposited: Set to `True`` to return the calculated power deposited.
    :return: Returns a list of coordinates (nx2) in the global domain with an optional power deposited.
    """

    if not isinstance(models, list):
        models = [models]

    exposurePoints = []


    for layerGeom in layer.geometry:

        # Get the model given the mid
        model = next(x for x in models if x.mid == layerGeom.mid)

        #Get the buildstyle from the model
        buildStyle = next(x for x in model.buildStyles if x.bid == layerGeom.bid)

        if buildStyle.pointDistance < 1:
            raise ValueError('The point distance parameter in the buildstyle (mid: {:d}, bid: {:d}) must be set'.format(model.mid, buildStyle.bid))

        pointDistance = buildStyle.pointDistance * 1e-3 # Convert to mm
        energyPerExposure = buildStyle.laserPower * (buildStyle.pointExposureTime * 1e-6) # convert to mu s

        if isinstance(layerGeom, HatchGeometry):

            # Calculate the length of the hatch vector and the direction
            coords = layerGeom.coords.reshape(-1, 2, 2)
            delta = np.diff(coords, axis=1).reshape(-1, 2)
            lineDist = np.hypot(delta[:, 0], delta[:, 1]).reshape(-1, 1)

            # Normalise each scan vector direction
            dir = -1.0 * delta / lineDist

            # Calculate the number of exposure points across the hatch vector based on its length
            numPoints = np.ceil(lineDist / pointDistance).astype(np.int)

            # Pre-populate some arrays to extrapolate the exposure points from
            totalPoints = int(np.sum(numPoints))
            idxArray = np.zeros([totalPoints, 1])
            pntsArray = np.zeros([totalPoints, 2])
            dirArray = np.zeros([totalPoints, 2])

            # Take the first coordinate
            p0 = coords[:, 1, :].reshape(-1, 2)

            idx = 0
            for i in range(len(numPoints)):
                j = int(numPoints[i])
                idxArray[idx:idx + j, 0] = np.arange(0, j)
                pntsArray[idx:idx + j] = p0[i]
                dirArray[idx:idx + j] = dir[i]
                idx += j

            # Calculate the hatch exposure points
            hatchExposurePoints = pntsArray + pointDistance * idxArray * dirArray

            # Add an extra column for the energy deposited per exposure
            if includePowerDeposited:
                col = np.ones([len(hatchExposurePoints),1])
                col[:] = energyPerExposure

                hatchExposurePoints = np.hstack([hatchExposurePoints, col])

            # append to the list
            exposurePoints.append(hatchExposurePoints)

        if isinstance(layerGeom, ContourGeometry):

            # Calculate the length of the hatch vector and the direction
            coords = layerGeom.coords

            delta = np.diff(coords, axis=0)
            lineDist = np.hypot(delta[:, 0], delta[:, 1]).reshape(-1, 1)

            # Normalise each scan vector direction
            dir = 1.0 * delta / lineDist

            # Calculate the number of exposure points across the hatch vector based on its length
            numPoints = np.ceil(lineDist / pointDistance).astype(np.int)

            # Pre-populate some arrays to extrapolate the exposure points from
            totalPoints = int(np.sum(numPoints))
            idxArray = np.zeros([totalPoints, 1])
            pntsArray = np.zeros([totalPoints, 2])
            dirArray = np.zeros([totalPoints, 2])

            # Take the first coordinate
            p0 = coords

            idx = 0
            for i in range(len(numPoints)):
                j = int(numPoints[i])
                idxArray[idx:idx + j, 0] = np.arange(0, j)
                pntsArray[idx:idx + j] = p0[i]
                dirArray[idx:idx + j] = dir[i]
                idx += j

            # Calculate the hatch exposure points
            hatchExposurePoints = pntsArray + pointDistance * idxArray * dirArray

            # Add an extra column for the energy deposited per exposure
            if includePowerDeposited:
                col = np.ones([len(hatchExposurePoints),1])
                col[:] = energyPerExposure

                hatchExposurePoints = np.hstack([hatchExposurePoints, col])

            # append to the list
            exposurePoints.append(hatchExposurePoints)

    exposurePoints = np.vstack(exposurePoints)

    return exposurePoints


class BaseHatcher(abc.ABC):
    """
    The BaseHatcher class provides common methods used for generating the 'contour' and infill 'hatch' scan vectors
    for a geometry slice typically a multi-polygon region.

    The class provides an interface tp generate a variety of hatching patterns used. The developer should re-implement a
    subclass and re-define the abstract method, :meth:`BaseHatcher.hatch`, which will be called.

    The user typically specifies a boundary, which may be offset the boundary of region using
    :meth:`offsetBoundary`. This is typically performed before generating the infill.
    Following offsetting, the a series of hatch lines are generated using :meth:`~BaseHatcher.generateHatching` to fill
    the entire boundary region using :meth:`polygonBoundingBox`. To obtain the final clipped infill, the
    hatches are clipped using :meth:`~BaseHatcher.clipLines` which are clipped in the same sequential order they are
    generated using a technique explained further in the class method. The generated scan paths should be stored into
    collections of :class:`~pyslm.geometry.LayerGeometry` accordingly.

    For all polygon manipulation operations used for offsetting and clipping, internally this calls provides automatic
    conversion to the integer coordinate system used by ClipperLib by internally calling
    :meth:`scaleToClipper` and :meth:`scaleFromClipper`.
    """


    PYCLIPPER_SCALEFACTOR = 1e5
    """ 
    The scaling factor used for polygon clipping and offsetting in `PyClipper <https://pypi.org/project/pyclipper/>`_ 
    for the decimal component of each polygon coordinate. This should be set to inverse of the required decimal 
    tolerance i.e. 0.01 requires a minimum scale factor of 100. This scaling factor is used 
    in :meth:`~BaseHatcher.scaleToClipper` and :meth:`~BaseHatcher.scaleFromClipper`. 
    
    :note:
        From experience, 1e4, mostly works, however, there are some artefacts generated during clipping hatch vectors.
        Therefore at a small peformance cost 1e5 is recommended.
    """

    def __init__(self):
        pass

    def __str__(self):
        return 'BaseHatcher <{:s}>'.format(self.name)

    @staticmethod
    def scaleToClipper(feature: Any):
        """
        Transforms geometry created **to pyclipper**  by upscaling into the integer coordinates  **from** the original
        floating point coordinate system.

        :param feature: The geometry to scale to pyclipper
        :return: The scaled geometry
        """
        return pyclipper.scale_to_clipper(feature, BaseHatcher.PYCLIPPER_SCALEFACTOR)

    @staticmethod
    def scaleFromClipper(feature: Any):
        """
        Transforms geometry created **from pyclipper** upscaled integer coordinates back **to** the original
        floating-point coordinate system.

        :param feature: The geometry to scale to pyclipper
        :return: The scaled geometry
        """
        return pyclipper.scale_from_clipper(feature, BaseHatcher.PYCLIPPER_SCALEFACTOR)

    @staticmethod
    def clipperToHatchArray(coords: np.ndarray) -> np.ndarray:
        """
        A helper method which converts the raw polygon edge lists returned by `PyClipper <https://pypi.org/project/pyclipper/>`_
        into a numpy array.

        :param coords: The list of hatches generated from pyclipper
        :return: The hatch coordinates transfromed into a (n x 2 x 3) numpy array.
        """
        return np.transpose(np.dstack(coords), axes=[2, 0, 1])

    @classmethod
    def error(cls) -> float:
        """
        Returns the accuracy of the polygon clipping depending on the chosen scale factor :attr:`.PYCLIPPER_SCALEFACTOR`.
        """
        return 1. / cls.PYCLIPPER_SCALEFACTOR

    @staticmethod
    def _getChildPaths(poly):

        offsetPolys = []

        # Create single closed polygons for each polygon
        paths = [path.Contour for path in poly.Childs]  # Polygon holes
        paths.append(poly.Contour)  # Path holes

        # Append the first point to the end of each path to close loop
        for path in paths:
            path.append(path[0])

        paths = BaseHatcher.scaleFromClipper(paths)

        offsetPolys.append(paths)

        for polyChild in poly.Childs:
            if len(polyChild.Childs) > 0:
                for polyChild2 in polyChild.Childs:
                    offsetPolys += BaseHatcher._getChildPaths(polyChild2)

        return offsetPolys

    @staticmethod
    def offsetPolygons(polygons, offset: float):
        """
        Offsets a set of boundaries across a collection of polygons by the offset parameter.

        .. note::
            Note that if any polygons are expanded overlap with adjacent polygons, the offsetting will **NOT** unify
            into a single shape.

        :param polygons: A list of closed polygons which are individually offset from each other.
        :param offset: The offset distance applied to the polygon
        :return: A list of boundaries offset from the subject
        """
        return [BaseHatcher.offsetBoundary(poly, offset) for poly in polygons]

    @staticmethod
    def offsetBoundary(paths, offset: float):
        """
        Offsets a single path for a single polygon.

        :param paths: Closed polygon path list for offsetting
        :param offset: The offset applied to the poylgon
        :return: A list of boundaries offset from the subject
        """
        pc = pyclipper.PyclipperOffset()

        clipperOffset = BaseHatcher.scaleToClipper(offset)

        # Append the paths to libClipper offsetting algorithm
        for path in paths:
            pc.AddPath(BaseHatcher.scaleToClipper(path),
                       pyclipper.JT_ROUND,
                       pyclipper.ET_CLOSEDPOLYGON)

        # Perform the offseting operation
        boundaryOffsetPolys = pc.Execute2(clipperOffset)

        offsetContours = []
        # Convert these nodes back to paths
        for polyChild in boundaryOffsetPolys.Childs:
            offsetContours += BaseHatcher._getChildPaths(polyChild)


        return offsetContours

    @staticmethod
    def polygonBoundingBox(obj: Any) -> np.ndarray:
        """
        Returns the bounding box of the polygon - typically this represents a single shape with an exterior and a list of
        boundaries within an array

        :param obj: Geometry object
        :return: A (1x6) numpy array representing the bounding box of a polygon
        """
        # Path (n,2) coords that

        if not isinstance(obj, list):
            obj = [obj]

        bboxList = []

        for subObj in obj:
            path = np.array(subObj)[:, :2]  # Use only coordinates in XY plane
            bboxList.append(np.hstack([np.min(path, axis=0), np.max(path, axis=0)]))

        bboxList = np.vstack(bboxList)
        bbox = np.hstack([np.min(bboxList[:, :2], axis=0), np.max(bboxList[:, -2:], axis=0)])

        return bbox

    @staticmethod
    def boundaryBoundingBox(boundaries):
        """
        Returns the bounding box of a list of boundaries, typically generated by the tree representation in PyClipper.

        :param boundaries: A list of polygon
        :return: A (1x6) numpy array
        """

        bboxList = [BaseHatcher.polygonBoundingBox(boundary) for boundary in boundaries]

        bboxList = np.vstack(bboxList)
        bbox = np.hstack([np.min(bboxList[:, :2], axis=0), np.max(bboxList[:, -2:], axis=0)])

        return bbox

    def clipLines2(paths, lines):
        #from  _martinez import Contour
        #from _martinez import Point
        #from _martinez import Polygon
        #from _martinez import OperationType, compute

        from martinez.contour import Contour
        from martinez.point import Point
        from martinez.polygon import Polygon
        from martinez.boolean import OperationType, compute

        import matplotlib.pyplot as plt

        left_line = Polygon([Contour([Point(1.0, -100.0), Point(1.0, 100.0)], [], True),
                             Contour([Point(1.1, -100.0), Point(1.1, 100.0)], [], True)])

        contours = []
        paths2 = [paths[0]]
        pyPoints = []
        for path in paths2:
            for boundary in path:
                points =  []

                for point in boundary:
                    points.append(Point(point[0], point[1]))
                    pyPoints.append(point[:2])

                #points.append(Point(boundary[0][0], boundary[0][1]))

                contours.append(Contour(points,[],True))
                #pc.AddPath(BaseHatcher.scaleToClipper(boundary), pyclipper.PT_CLIP, True)

        #plt.plot(lines[:,0], lines[:,1])
        pyPoints = np.vstack(pyPoints)
        plt.plot(pyPoints[:,0], pyPoints[:,1])
        polygon = Polygon(contours)

        # Reshape line list to create n lines with 2 coords(x,y,z)
        #lineList = lines.reshape(-1, 2, 3)
        #lineList = tuple(map(tuple, lineList))
        #lineList = BaseHatcher.scaleToClipper(lineList)


        edges = []

        lineList =  lines.reshape([-1, 2, 3])

        i = 0

        results = []


        for i in np.arange(0,lineList.shape[0]):
            #i += 1
            point = lineList[i]

            edge = Contour([Point(point[0,0], point[0,1]),
                            Point(point[1,0], point[1,1])], [], True)
            edges.append(edge)

            #edgePoly = Polygon([edge])
            #results.append(compute(polygon, edgePoly, OperationType.INTERSECTION))

        edgesPoly = Polygon(edges)
        result = compute(polygon, edgesPoly, OperationType.INTERSECTION)


        #for result in results:
        for r in result.contours:
            points = [(p.x,p.y) for p in r.points]
            points = np.vstack(points)

            plt.plot(points[:,0], points[:,1])

        return results


    @staticmethod
    def clipLines(paths, lines):
        """
        This function clips a series of lines (hatches) across a closed polygon using `Pyclipper <https://pypi.org/project/pyclipper/>`_.

        .. note ::
            The order is guaranteed from the list of lines used, so these do not require sorting usually. However,
            the position may require additional sorting to cater for the user's requirements.

        :param paths: The set of boundary paths for trimming the lines
        :param lines: The un-trimmed lines to clip from the boundary
        :return: A list of trimmed lines (open paths)
        """
        #clipLines = BaseHatcher.clipLines2(paths, lines)

        if len(lines) == 0:
            # Input from generateHatching is empty so return empty
            return None

        pc = pyclipper.Pyclipper()

        for path in paths:
            for boundary in path:
                pc.AddPath(BaseHatcher.scaleToClipper(boundary), pyclipper.PT_CLIP, True)

        # Reshape line list to create n lines with 2 coords(x,y,z)
        lineList = lines.reshape(-1, 2, 3)
        lineList = tuple(map(tuple, lineList))
        lineList = BaseHatcher.scaleToClipper(lineList)

        pc.AddPaths(lineList, pyclipper.PT_SUBJECT, False)

        # Note open paths (lines) have to used PyClipper::Execute2 in order to perform trimming
        result = pc.Execute2(pyclipper.CT_INTERSECTION, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)

        # Cast from PolyNode Struct from the result into line paths since this is not a list
        lineOutput = pyclipper.PolyTreeToPaths(result)

        return BaseHatcher.scaleFromClipper(lineOutput)

    @staticmethod
    def clipContourLines(paths, contourPaths: List[np.ndarray]):
        """
        This function clips a series of (contour paths) across a closed polygon using
        `Pyclipper <https://pypi.org/project/pyclipper/>`_.

        .. note ::
            The order is guaranteed from the list of lines used, so these do not require sorting. However,
            the position may require additional sorting to cater for the user's requirements.

        :param paths: The set of boundary paths for trimming the lines
        :param contourPaths: The un-trimmed complex **open** paths to be clipped
        :return: A list of trimmed lines (open paths)
        """

        pc = pyclipper.Pyclipper()

        for path in paths:
            for boundary in path:
                pc.AddPath(BaseHatcher.scaleToClipper(boundary), pyclipper.PT_CLIP, True)

        # Reshape line list to create n lines with 2 coords(x,y,z)
        #lineList = lines.reshape(-1, 2, 3)
        #lineList = tuple(map(tuple, lineList))
        #lineList = BaseHatcher.scaleToClipper(lineList)

        for contour in contourPaths:
            path = BaseHatcher.scaleToClipper(contour)
            pc.AddPath(path, pyclipper.PT_SUBJECT, False)

        # Note open paths (lines) have to used PyClipper::Execute2 in order to perform trimming
        result = pc.Execute2(pyclipper.CT_INTERSECTION, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)

        # Cast from PolyNode Struct from the result into line paths since this is not a list
        lineOutput = pyclipper.PolyTreeToPaths(result)

        return BaseHatcher.scaleFromClipper(lineOutput)

    def generateHatching(self, paths, hatchSpacing: float, hatchAngle: Optional[float] = 90.0) -> np.ndarray:
        """
        Generates un-clipped hatches which is guaranteed to cover the entire polygon region base on the maximum extent
        of the polygon bounding box

        :param paths: Boundary paths to generate hatches to cover
        :param hatchSpacing: Hatch Spacing to use
        :param hatchAngle: Hatch angle (degrees) to rotate the scan vectors

        :return: Returns the list of un-clipped scan vectors
        """

        """
        The hatch angle
        Note the angle is reversed here because the rotation matrix is counter-clockwise
        """
        theta_h = np.radians(hatchAngle)# * -1.0)  # 'rad'

        # Get the bounding box of the paths
        bbox = self.boundaryBoundingBox(paths)

        # Expand the bounding box
        bboxCentre = np.mean(bbox.reshape(2, 2), axis=0)

        # Calculates the diagonal length for which is the longest
        diagonal = bbox[2:] - bboxCentre
        bboxRadius = np.sqrt(diagonal.dot(diagonal))

        # Construct a square which wraps the radius
        x = np.tile(np.arange(-bboxRadius, bboxRadius, hatchSpacing, dtype=np.float32).reshape(-1, 1), (2)).flatten()
        y = np.array([-bboxRadius, bboxRadius])
        y = np.resize(y, x.shape)
        z = np.arange(0, x.shape[0] / 2, 0.5).astype(np.int64)

        coords = np.hstack([x.reshape(-1, 1),
                            y.reshape(-1, 1),
                            z.reshape(-1, 1)])

        # Create the 2D rotation matrix with an additional row, column to preserve the hatch order
        c, s = np.cos(theta_h), np.sin(theta_h)
        R = np.array([(c, -s, 0),
                      (s, c, 0),
                      (0, 0, 1.0)])

        # Apply the rotation matrix and translate to bounding box centre
        coords = np.matmul(R, coords.T)
        coords = coords.T + np.hstack([bboxCentre, 0.0])

        return coords

    @abc.abstractmethod
    def hatch(self, boundaryFeature) -> Union[Layer, None]:
        """
        The hatch method should be re-implemented by a child class to generate a :class:`Layer` containing the scan
        vectors used for manufacturing the layer.

        :param boundaryFeature: The collection of boundaries of closed polygons within a layer.
        :raises: :class:`NotImplementedError`
        """
        raise NotImplementedError()


class InnerHatchRegion(abc.ABC):
    """
    The InnerHatchRegion class provides a representation for a single sub-region used for efficiently generating
    various sub-scale hatch infills. This requires providing a boundary (:attr:`~InnerHatchRegion.boundary`) to represent
    the region used. The user typically in derived :class:`BaseHatcher` class should set via
    :meth:`~InnerHatchRegion.setRequiresClipping` if the region requires further clipping.

    Finally the derived class must generate a set of hatch vectors covering the boundary region, by re-implementing the
    abstract method :meth:`~InnerHatchRegion.hatch`. If the boundary requires clipping, the interior hatches are also
    clipped.
    """

    def __init__(self):

        self._origin =  np.array([[0,0]])
        self._orientation = 0.0

        self._region = []
        self._requiresClipping = False
        self._isIntersecting = False

    def transformCoordinates2D(self, coords: np.ndarray) -> np.ndarray:
        """
        Transforms a set of (n x 2) coordinates using the rotation angle
        :attr:`InnerHatchRegion.orientation` using the 2D rotation matrix in :meth:`InnerHatchRegion.rotationMatrix2D`.

        :param coords: (nx2) coordinates to be transformed
        :return: The transformed coordinates
        """
        R = self.rotationMatrix2D()

        # Apply the rotation matrix and translate to bounding box centre
        coords = np.matmul(R, coords.T)
        coords = coords.T + np.hstack([self._origin])

        return coords

    def transformCoordinates(self, coords: np.ndarray) -> np.ndarray:
        """
        Transforms a set of (n x 3) coordinates with a sort id using the rotation angle
        :attr:`InnerHatchRegion.orientation` using the 3D rotation matrix in :meth:`InnerHatchRegion.rotationMatrix3D`.

        :param coords: (nx3) coordinates to be transformed
        :return:  The transformed coordinates
        """

        R = self.rotationMatrix3D()

        # Apply the rotation matrix and translate to bounding box centre
        coords = np.matmul(R, coords.T).T
        coords[:,:2] += self._origin

        return coords

    def rotationMatrix2D(self) -> np.ndarray:
        """
        Generates an affine matrix covering the transformation based on the origin and orientation based on a rotation
        around the local coordinate system. This should be used when only a series of x,y coordinate required to be
        transformed.

        :return: Affine Transformation Matrix
        """
        # Create the rotation matrix
        c, s = np.cos(self._orientation), np.sin(self._orientation)
        R = np.array([(c, -s),
                      (s, c)])
        return R

    def rotationMatrix3D(self) -> np.ndarray:
        """
        Generates an affine matrix covering the transformation based on the origin and orientation based on a rotation
        around the local coordinate system. A pseudo third row and column is provided to retain the hatch sort id used.

        :return: Affine Transformation Matrix
        """
        # Create the rotation matrix
        c, s = np.cos(self._orientation), np.sin(self._orientation)
        R = np.array([(c, -s, 0),
                      (s, c, 0),
                      (0, 0, 1.0)])

        return R

    @property
    def orientation(self) -> float:
        """
        The orientation describes the rotation of the local coordinate system with respect to the global
        coordinate system :math:`(x,y)`. The angle of rotation is given in rads.
        """
        return self._orientation

    @orientation.setter
    def orientation(self, angle: float):
        self._orientation = angle

    @property
    def origin(self):
        """ The origin is the :math:`(x',y')` position of the local coordinate system. """
        return self._origin

    @origin.setter
    def origin(self, coord):
        self._origin = coord

    def setIntersecting(self, intersectingState: bool) -> None:
        """
        Setting `True` indicates the region has been intersected

        :param intersectingState: True if the region intersects
        """
        self._isIntersecting = intersectingState

    def setRequiresClipping(self, clippingState: bool) -> None:
        """
        Sets the internal region to require additional clipping following hatch generation.

        :param clippingState: True if the region requires additional clipping
        """
        self._requiresClipping = clippingState

    def __str__(self):
        return 'InnerHatchRegion <{:s}>'

    @abc.abstractmethod
    def boundary(self) -> ShapelyPolygon:
        """ The boundary of the internal region

        :raises: :class:`NotImplementedError`
        """
        raise NotImplementedError

    def isIntersecting(self) -> bool:
        """
        Returns `True` if the region requires additional clipping.
        """

        return self._isIntersecting

    def requiresClipping(self) -> bool:
        """
        Returns `True` if the region requires additional clipping.
        """
        return self._requiresClipping

    @abc.abstractmethod
    def hatch(self) -> np.ndarray:
        """
        The hatch method should provide a list of hatch vectors, within the boundary. This must be re-implemented in
        the derived class. The hatch vectors should be ordered.

        :raises: :class:`NotImplementedError`
        """
        raise NotImplementedError()


class Hatcher(BaseHatcher):
    """
    Provides a generic SLM Hatcher 'recipe' with standard parameters for defining the hatch across regions. This
    includes generating multiple contour offsets and then a generic hatch infill pattern by re-implementing the
    :meth:`BaseHatcher.hatch` method. This class may be derived from to provide additional or customised behavior.
    """

    def __init__(self):

        super().__init__()

        # Contour private attributes
        self._scanContourFirst = False
        self._numInnerContours = 1
        self._numOuterContours = 1
        self._spotCompensation = 0.08  # mm
        self._contourOffset = 1.0 * self._spotCompensation
        self._volOffsetHatch = self._spotCompensation


        # Hatch private attributes
        self._layerAngleIncrement = 0  # 66 + 2 / 3
        self._hatchDistance = 0.08  # mm
        self._hatchAngle = 45
        self._hatchSortMethod = None
        self._hatchingEnabled = True

    @property
    def hatchDistance(self) -> float:
        """ The distance between adjacent hatch scan vectors """
        return self._hatchDistance

    @hatchDistance.setter
    def hatchDistance(self, value: float):
        self._hatchDistance = value

    @property
    def hatchAngle(self) -> float:
        """
        The base hatch angle used for hatching the region expressed in degrees :math:`[-180,180]`
        """
        return self._hatchAngle

    @hatchAngle.setter
    def hatchAngle(self, value: float):
        self._hatchAngle = value

    @property
    def layerAngleIncrement(self) -> float:
        """
        An additional offset used to increment the hatch angle between layers in degrees. This is typically set to
        66.6 :math:`^\\circ` per layer to provide additional uniformity of the scan vectors across multiple layers.
        By default this is set to `0.0` """
        return self._layerAngleIncrement

    @layerAngleIncrement.setter
    def layerAngleIncrement(self, value):
        self._layerAngleIncrement = value

    @property
    def hatchSortMethod(self):
        """ The hatch sort method used once the hatch vectors have been generated """
        return self._hatchSortMethod

    @hatchSortMethod.setter
    def hatchSortMethod(self, sortObj: Any):

        if sortObj is None:
            pass
        elif not isinstance(sortObj, BaseSort):
            raise TypeError("The Hatch Sort Method should be derived from the BaseSort class")

        self._hatchSortMethod = sortObj

    @property
    def scanContourFirst(self) -> bool:
        """
        Determines if the contour/border vectors :class:`LayerGeometry` are scanned first before the hatch vectors. By
        default this is set to `False`.
        """
        return self._scanContourFirst

    @scanContourFirst.setter
    def scanContourFirst(self, value: bool):
        self._scanContourFirst = value

    @property
    def numInnerContours(self) -> int:
        """
        The total number of inner contours to generate by offsets from the boundary region.
        """
        return self._numInnerContours

    @numInnerContours.setter
    def numInnerContours(self, value: int):
        self._numInnerContours = value

    @property
    def numOuterContours(self) -> int:
        """
        The total number of outer contours to generate by offsets from the boundary region.
        """
        return self._numOuterContours

    @numOuterContours.setter
    def numOuterContours(self, value: int):
        self._numOuterContours = value

    @property
    def spotCompensation(self) -> float:
        """
        The spot (laser point) compensation factor is the distance to offset the outer-boundary and other internal hatch
        features in order to factor in the exposure radius of the laser.
        """
        return self._spotCompensation

    @spotCompensation.setter
    def spotCompensation(self, value: float):
        self._spotCompensation = value

    @property
    def contourOffset(self) -> float:
        """
        The contour offset is the distance between the contour or border scans
        """
        return self._contourOffset

    @contourOffset.setter
    def contourOffset(self, offset: float):
        self._contourOffset = offset

    @property
    def volumeOffsetHatch(self) -> float:
        """
        An additional offset may be added (positive or negative) between the contour/border scans and the
        internal hatching for the bulk volume.
        """
        return self._volOffsetHatch

    @volumeOffsetHatch.setter
    def volumeOffsetHatch(self, value: float):
        self._volOffsetHatch = value

    @property
    def hatchingEnabled(self) -> bool:
        """ If the internal hatch region should be processed (default: True)"""
        return self._hatchingEnabled

    @hatchingEnabled.setter
    def hatchingEnabled(self, value):
        self._hatchingEnabled = value

    def hatch(self, boundaryFeature) -> Union[Layer, None]:
        """
        Generates a series of contour or boundary offsets along with a basic full region internal hatch.

        :param boundaryFeature: The collection of boundaries of closed polygons within a layer.
        :return: A :class:`Layer` object containing a list of :class:`LayerGeometry` objects generated
        """
        if len(boundaryFeature) == 0:
            return None

        layer = Layer(0, 0)
        # First generate a boundary with the spot compensation applied

        offsetDelta = 1e-6
        offsetDelta -= self._spotCompensation

        # Store all contour layer geometries to before adding at the end of each layer
        contourLayerGeometries = []
        hatchLayerGeometries = []

        for i in range(self._numOuterContours):

            if i > 0:
                offsetDelta -= self._contourOffset

            offsetBoundary = self.offsetBoundary(boundaryFeature, offsetDelta)

            for poly in offsetBoundary:
                for path in poly:
                    contourGeometry = ContourGeometry()
                    contourGeometry.coords = np.array(path)[:, :2]
                    contourGeometry.subType = "outer"
                    contourLayerGeometries.append(contourGeometry)  # Append to the layer



        # Repeat for inner contours
        for i in range(self._numInnerContours):

            if i > 0:
                offsetDelta -= self._contourOffset

            offsetBoundary = self.offsetBoundary(boundaryFeature, offsetDelta)

            for poly in offsetBoundary:
                for path in poly:
                    contourGeometry = ContourGeometry()
                    contourGeometry.coords = np.array(path)[:, :2]
                    contourGeometry.subType = "inner"
                    contourLayerGeometries.append(contourGeometry)  # Append to the layer

        # The final offset is applied to the boundary if there has been existing contour offsets applied
        if self._numInnerContours + self._numOuterContours > 0:
            offsetDelta -= self._volOffsetHatch

        curBoundary = self.offsetBoundary(boundaryFeature, offsetDelta)

        scanVectors = []

        if self.hatchingEnabled and len(curBoundary) > 0:
            paths = curBoundary

            # Hatch angle will change per layer
            # TODO change the layer angle increment
            layerHatchAngle = np.mod(self._hatchAngle + self._layerAngleIncrement, 180)
            #layerHatchAngle = float(self._hatchAngle + self._layerAngleIncrement)
            #layerHatchAngle -= np.floor(layerHatchAngle / 360. + 0.5) * 360.

            # The layer hatch angle needs to be bound by +ve X vector (i.e. -90 < theta_h < 90 )
            if layerHatchAngle > 90:
                layerHatchAngle = layerHatchAngle - 180

            # Generate the un-clipped hatch regions based on the layer hatchAngle and hatch distance
            hatches = self.generateHatching(paths, self._hatchDistance, layerHatchAngle)

            # Clip the hatch fill to the boundary
            clippedPaths = self.clipLines(paths, hatches)
            clippedLines = []

            # Merge the lines together
            if len(clippedPaths) > 0:

                clippedLines = BaseHatcher.clipperToHatchArray(clippedPaths)

                # Extract only x-y coordinates and sort based on the pseudo-order stored in the z component.
                clippedLines = clippedLines[:, :, :3]
                id = np.argsort(clippedLines[:, 0, 2])
                clippedLines = clippedLines[id, :, :]

                scanVectors.append(clippedLines)

                # Scan vectors have been created for the hatched region

                # Construct a HatchGeometry containing the list of points
                hatchGeom = HatchGeometry()

                # Only copy the (x,y) points from the coordinate array.
                hatchVectors = np.vstack(scanVectors)
                hatchVectors = hatchVectors[:, :, :2].reshape(-1, 2)

                # Note the does not require positional sorting
                if self.hatchSortMethod:
                    hatchVectors = self.hatchSortMethod.sort(hatchVectors)

                hatchGeom.coords = hatchVectors
                hatchLayerGeometries.append(hatchGeom)
        if False:
            # Iterate through each closed polygon region in the slice. The currently individually sliced.
            for contour in curBoundary:
                # print('{:=^60} \n'.format(' Generating hatches '))

                paths = contour

                # Hatch angle will change per layer
                # TODO change the layer angle increment
                layerHatchAngle = np.mod(self._hatchAngle + self._layerAngleIncrement, 180)

                # The layer hatch angle needs to be bound by +ve X vector (i.e. -90 < theta_h < 90 )
                if layerHatchAngle > 90:
                    layerHatchAngle = layerHatchAngle - 180

                # Generate the un-clipped hatch regions based on the layer hatchAngle and hatch distance
                hatches = self.generateHatching(paths, self._hatchDistance, layerHatchAngle)

                # Clip the hatch fill to the boundary
                clippedPaths = self.clipLines(paths, hatches)

                # Merge the lines together
                if len(clippedPaths) == 0:
                    continue

                clippedLines = self.clipperToHatchArray(clippedPaths)

                # Extract only x-y coordinates and sort based on the pseudo-order stored in the z component.
                clippedLines = clippedLines[:, :, :3]
                id = np.argsort(clippedLines[:, 0, 2])
                clippedLines = clippedLines[id, :, :]

                scanVectors.append(clippedLines)



        if self._scanContourFirst:
            layer.geometry.extend(contourLayerGeometries + hatchLayerGeometries)
        else:
            layer.geometry.extend(hatchLayerGeometries + contourLayerGeometries)

        # Append the contours hatch vecotrs
        return layer


class StripeHatcher(Hatcher):
    """
    The Stripe Hatcher extends the standard :class:`Hatcher` but generates a set of stripe hatches of a fixed width
    (:attr:`~.stripeWidth`) to cover a region. This a common scan strategy adopted by users of EOS systems.
    This has the effect of limiting the max length of the scan vectors  across a region in order to mitigate the
    effects of residual stress.
    """

    def __init__(self):
        super().__init__()

        self._stripeWidth = 5.0
        self._stripeOverlap = 0.1
        self._stripeOffset = 0.5

    def __str__(self):
        return 'StripeHatcher'

    @property
    def stripeWidth(self) -> float:
        """ The stripe width """
        return self._stripeWidth

    @stripeWidth.setter
    def stripeWidth(self, width: float):
        self._stripeWidth = width

    @property
    def stripeOverlap(self) -> float:
        """ The length of overlap between adjacent stripes"""
        return self._stripeOverlap

    @stripeOverlap.setter
    def stripeOverlap(self, overlap: float):
        self._stripeOverlap = overlap

    @property
    def stripeOffset(self) -> float:
        """
        The stripe offset is the relative distance (hatch spacing) to move the scan vectors between adjacent stripes
        """
        return self._stripeOffset

    @stripeOffset.setter
    def stripeOffset(self, offset: float):
        self._stripeOffset = offset

    def generateHatching(self, paths, hatchSpacing: float, hatchAngle: float = 90.0) -> np.ndarray:
        """
        Generates un-clipped hatches which is guaranteed to cover the entire polygon region based on the maximum extent
        of the polygon bounding box

        :param paths:
        :param hatchSpacing: Hatch Spacing to use
        :param hatchAngle: Hatch angle (degrees) to rotate the scan vectors

        :return: Returns the list of unclipped scan vectors
        """

        """
        The hatch angle
        Note the angle is reversed here because the rotation matrix is counter-clockwise
        """
        theta_h = np.radians(hatchAngle * -1.0)  # 'rad'

        # Get the bounding box of the paths
        bbox = self.boundaryBoundingBox(paths)

        # Expand the bounding box
        bboxCentre = np.mean(bbox.reshape(2, 2), axis=0)

        # Calculates the diagonal length for which is the longest
        diagonal = bbox[2:] - bboxCentre
        bboxRadius = np.sqrt(diagonal.dot(diagonal))

        numStripes = int(2 * bboxRadius / self._stripeWidth) + 1

        # Construct a square which wraps the radius
        hatchOrder = 0
        coords = []

        for i in np.arange(0, numStripes):
            startX = -bboxRadius + i * self._stripeWidth - self._stripeOverlap
            endX = startX + self._stripeWidth + self._stripeOverlap

            y = np.tile(np.arange(-bboxRadius + np.mod(i, 2) * self._stripeOffset * hatchSpacing,
                                  bboxRadius + np.mod(i, 2) * self._stripeOffset * hatchSpacing, hatchSpacing,
                                  dtype=np.float32).reshape(-1, 1), (2)).flatten()
            # x = np.tile(np.arange(startX, endX, hatchSpacing, dtype=np.float32).reshape(-1, 1), (2)).flatten()
            x = np.array([startX, endX])
            x = np.resize(x, y.shape)
            z = np.arange(hatchOrder, hatchOrder + y.shape[0] / 2, 0.5).astype(np.int64)

            hatchOrder += x.shape[0] / 2

            coords += [np.hstack([x.reshape(-1, 1),
                                  y.reshape(-1, 1),
                                  z.reshape(-1, 1)])]

        coords = np.vstack(coords)

        # Create the rotation matrix
        c, s = np.cos(theta_h), np.sin(theta_h)
        R = np.array([(c, -s, 0),
                      (s, c, 0),
                      (0, 0, 1.0)])

        # Apply the rotation matrix and translate to bounding box centre
        coords = np.matmul(R, coords.T)
        coords = coords.T + np.hstack([bboxCentre, 0.0])

        return coords


class BasicIslandHatcher(Hatcher):
    """
    BasicIslandHatcher extends the standard :class:`Hatcher` but generates a set of islands of fixed size
    (:attr:`.islandWidth`)  which covers a region.  This a common scan strategy adopted across SLM systems.
    This has the effect of limiting the max length of the scan whilst by orientating the scan vectors orthogonal
    to each other mitigating any preferential distortion or curling  in a single direction and any
    effects to micro-structure.

    :note:

        This method is not optimal and is provided as a reference for the user to improve their own understand and
        develop their own island scan strategies. For optimal performance, the user should refer instead to
        :class:`IslandHatcher`

    """

    def __init__(self):

        super().__init__()

        self._islandWidth = 5.0
        self._islandOverlap = 0.1
        self._islandOffset = 0.5

    def __str__(self):
        return 'IslandHatcher'

    @property
    def islandWidth(self) -> float:
        """ The island width """
        return self._islandWidth

    @islandWidth.setter
    def islandWidth(self, width: float):
        self._islandWidth = width

    @property
    def islandOverlap(self) -> float:
        """ The length of overlap between adjacent islands"""
        return self._islandOverlap

    @islandOverlap.setter
    def islandOverlap(self, overlap: float):
        self._islandOverlap = overlap

    @property
    def islandOffset(self) -> float:
        """
        The island offset is the relative distance (hatch spacing) to move the scan vectors between adjacent
        checkers.
        """
        return self._islandOffset

    @islandOffset.setter
    def islandOffset(self, offset: float):
        self._islandOffset = offset

    def generateHatching(self, paths, hatchSpacing: float, hatchAngle: float = 90.0) -> np.ndarray:
        """
        Generates un-clipped hatches which is guaranteed to cover the entire polygon region base on the maximum extent
        of the polygon bounding box.

        :param paths: The boundaries that the hatches should fill entirely
        :param hatchSpacing: The hatch spacing
        :param hatchAngle: The hatch angle (degrees) to rotate the scan vectors
        :return: Returns the list of unclipped scan vectors covering the region
        """
        # Hatch angle
        theta_h = np.radians(hatchAngle)  # 'rad'

        # Get the bounding box of the paths
        bbox = self.boundaryBoundingBox(paths)

        # Expand the bounding box
        bboxCentre = np.mean(bbox.reshape(2, 2), axis=0)

        # Calculates the diagonal length for which is the longest
        diagonal = bbox[2:] - bboxCentre
        bboxRadius = np.sqrt(diagonal.dot(diagonal))

        numIslands = int(2 * bboxRadius / self._islandWidth) + 1

        # Construct a square which wraps the radius
        hatchOrder = 0
        coords = []

        for i in np.arange(0, numIslands):
            for j in np.arange(0, numIslands):

                startX = -bboxRadius + i * (self._islandWidth) - self._islandOverlap
                endX = startX + (self._islandWidth) + self._islandOverlap

                startY = -bboxRadius + j * (self._islandWidth) - self._islandOverlap
                endY = startY + (self._islandWidth) + self._islandOverlap

                if np.mod(i + j, 2):
                    y = np.tile(np.arange(startY + np.mod(i + j, 2) * self._islandOffset * hatchSpacing,
                                          endY + np.mod(i + j, 2) * self._islandOffset * hatchSpacing, hatchSpacing,
                                          dtype=np.float32).reshape(-1, 1), (2)).flatten()

                    x = np.array([startX, endX])
                    x = np.resize(x, y.shape)
                    z = np.arange(hatchOrder, hatchOrder + y.shape[0] / 2, 0.5).astype(np.int64)

                else:
                    x = np.tile(np.arange(startX + np.mod(i + j, 2) * self._islandOffset * hatchSpacing,
                                          endX + np.mod(i + j, 2) * self._islandOffset * hatchSpacing, hatchSpacing,
                                          dtype=np.float32).reshape(-1, 1), (2)).flatten()

                    y = np.array([startY, endY])
                    y = np.resize(y, x.shape)
                    z = np.arange(hatchOrder, hatchOrder + y.shape[0] / 2, 0.5).astype(np.int64)

                hatchOrder += x.shape[0] / 2

                coords += [np.hstack([x.reshape(-1, 1),
                                      y.reshape(-1, 1),
                                      z.reshape(-1, 1)])]

        coords = np.vstack(coords)

        # Create the rotation matrix
        c, s = np.cos(theta_h), np.sin(theta_h)
        R = np.array([(c, -s, 0),
                      (s, c, 0),
                      (0, 0, 1.0)])

        # Apply the rotation matrix and translate to bounding box centre
        coords = np.matmul(R, coords.T)
        coords = coords.T + np.hstack([bboxCentre, 0.0])

        return coords
