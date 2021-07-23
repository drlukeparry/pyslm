from typing import Any, List, Optional, Tuple

import numpy as np

from pyslm import pyclipper
from shapely.geometry import LinearRing, MultiPolygon, Polygon

from ..geometry import Layer, LayerGeometry, ContourGeometry, HatchGeometry, PointsGeometry
from .hatching import Hatcher, InnerHatchRegion
from .utils import pathsToClosedPolygons


class Island(InnerHatchRegion):
    """
    Island represents a square sub-region containing a series of orthogonal hatches which represents a typically a
    checkerboard scan strategy.
    """

    _boundary = None
    """ Private class attribute which is used to cache the boundary generated"""

    def __init__(self, origin: np.ndarray = np.array([[0.0,0.0]]), orientation: Optional[float] = 0.0,
                       islandWidth: Optional[float] = 0.0, islandOverlap: Optional[float] = 0.0,
                       hatchDistance: Optional[float] = 0.1):

        super().__init__()

        self.posId = 0
        self.id = 0
        self.origin = origin
        self.orientation = orientation
        self._islandOverlap = islandOverlap
        self._islandWidth = islandWidth
        self._hatchDistance = hatchDistance

    def __str__(self):
        return 'IslandRegion'

    @property
    def hatchDistance(self) -> float:
        """ The distance between adjacent hatch vectors """
        return self._hatchDistance

    @hatchDistance.setter
    def hatchDistance(self, distance: float):
        self._hatchDistance = distance

    @property
    def islandWidth(self) -> float:
        """ The square island width """
        return self._islandWidth

    @islandWidth.setter
    def islandWidth(self, width: float):
        self._islandWidth = width

    @property
    def islandOverlap(self) -> float:
        """ The length of overlap between adjacent islands in both directions :math:`(x', y')`"""
        return self._islandOverlap

    @islandOverlap.setter
    def islandOverlap(self, overlap: float):
        self._islandOverlap = overlap

    def localBoundary(self) -> np.ndarray:
        """
        Returns the local square boundary based on the island width (:attr:`~Island.islandWidth`) and  the
        island overlap (:attr:`~Island.islandOverlap`). The island overlap provides an offset from the original boundary,
        so the user must compensate the actual overlap by a factor of a half. The boundary is cached into a static class
        attribute :attr:Island._boundary` since this remains constant typically across the entire hatching process.
        If the user desires to change this the user should re-implement the class and this method.

        :return: Coordinates representing the local boundary
        """

        if Island._boundary is None:
            sx = -self.islandOverlap
            sy = - self.islandOverlap

            ex = self._islandWidth + self.islandOverlap
            ey = self._islandWidth + self._islandOverlap

            # Generate a square island
            Island._boundary = np.array([(sx, sy),
                                         (sx, ey),
                                         (ex, ey),
                                         (ex, sy),
                                         (sx, sy)])

        return Island._boundary

    def boundary(self) -> Polygon:
        """
        Returns the transformed boundary obtained from :meth:`~Island.localBoundary` into
        the global coordinate system :math:`(x,y)`.

        :return: Boundary polygon
        """

        coords = self.localBoundary()
        return Polygon(self.transformCoordinates2D(coords))

    def generateInternalHatch(self, isOdd: bool = True) -> np.ndarray:
        """
        Generates a set of hatches orthogonal to the island's coordinate system :math:`(x', y')`.

        :param isOdd: The chosen orientation of the hatching
        :return: (nx3) Set of sorted hatch coordinates
        """

        startX = -self._islandOverlap
        startY = - self._islandOverlap

        endX = self._islandWidth + self._islandOverlap
        endY = self._islandWidth + self._islandOverlap

        if isOdd:
            y = np.tile(np.arange(startY, endY, self._hatchDistance).reshape(-1, 1), 2).flatten()
            x = np.array([startX, endX])
            x = np.resize(x, y.shape)
        else:
            x = np.tile(np.arange(startX, endX, self._hatchDistance).reshape(-1, 1), 2).flatten()
            y = np.array([startY, endY])
            y = np.resize(y, x.shape)

        z = np.arange(0, y.shape[0] / 2, 0.5).astype(np.int64)

        return np.hstack([x.reshape(-1, 1),
                          y.reshape(-1, 1),
                          z.reshape(-1,1)])

    def hatch(self) -> np.ndarray:
        """
        Generates a set of hatches orthogonal to the island's coordinate system depending on if the sum of
        :attr:`~Island.posId` is even or odd. The returned hatch vectors are transformed and sorted depending on the
        direction.

        :return: The transformed and ordered hatch vectors
        """
        isOdd = np.mod(sum(self.posId), 2)
        coords = self.generateInternalHatch(isOdd)
        return self.transformCoordinates(coords)


class IslandHatcher(Hatcher):
    """
    IslandHatcher extends the standard :class:`Hatcher` but generates a set of islands of fixed size (:attr:`~.islandWidth`)
    which covers a region.  This a common scan strategy adopted across SLM systems. This has the effect of limiting the
    maximum length of the scan whilst by orientating the scan vectors orthogonal to each other mitigating any
    preferential distortion or curling  in a single direction and any effects to micro-structure.
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
        """ The length of overlap between adjacent islands in both directions """
        return self._islandOverlap

    @islandOverlap.setter
    def islandOverlap(self, overlap: float):
        self._islandOverlap = overlap

    @property
    def islandOffset(self) -> float:
        """
        The island offset is the relative distance (hatch spacing) to move the scan vectors between
        adjacent checkers.
        """
        return self._islandOffset

    @islandOffset.setter
    def islandOffset(self, offset: float):
        self._islandOffset = offset

    def clipIslands(self, paths, pathSubjects):
        """
        Internal method which clips the boundaries of :class:`Island` obtained from :meth:`InnerHatchRegion.boundary`
        with a list of paths. It is not actually used but provided as a reference for users.
        """

        pc = pyclipper.Pyclipper()

        for path in paths:
            pc.AddPath(self.scaleToClipper(path), pyclipper.PT_CLIP, True)

        for subjPath in pathSubjects:
            pc.AddPath(self.scaleToClipper(subjPath), pyclipper.PT_SUBJECT, True)

        # Note open paths (lines) have to used PyClipper::Execute2 in order to perform trimming
        result = pc.Execute2(pyclipper.CT_INTERSECTION, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)

        # Cast from PolyNode Struct from the result into line paths since this is not a list
        output = pyclipper.PolyTreeToPaths(result)

        return self.scaleFromClipper(output)

    def generateIslands(self, paths, hatchAngle: Optional[float] = 90.0) -> List[Island]:
        """
        Generates un-clipped islands which is guaranteed to cover the entire polygon region base on the maximum extent
        of the polygon bounding box. This method can be re-implement in a derived class to specify a different Island
        type to be used and also its placement of the islands to fill the polygon region.

        :param paths: The boundaries that the hatches should fill entirely
        :param hatchAngle: The hatch angle (degrees) to rotate the scan vectors

        :return: Returns the list of unclipped scan vectors covering the region
        """

        # Hatch angle
        theta_h = np.radians(hatchAngle)  # 'rad'

        # Get the bounding box of the boundary
        bbox = self.boundaryBoundingBox(paths)

        # Expand the bounding box
        bboxCentre = np.mean(bbox.reshape(2, 2), axis=0)

        # Calculates the diagonal length for which is the longest
        diagonal = bbox[2:] - bboxCentre
        bboxRadius = np.sqrt(diagonal.dot(diagonal))

        # Construct a square which wraps the radius
        numIslands = int(2 * bboxRadius / self._islandWidth) + 1

        # Create the rotation matrix
        c, s = np.cos(theta_h), np.sin(theta_h)
        R = np.array([(c, -s),
                      (s, c)])

        islands = []
        id = 0

        for i in np.arange(0, numIslands):
            for j in np.arange(0, numIslands):

                # Apply the rotation matrix and translate to bounding box centre
                startX = -bboxRadius + i * self._islandWidth
                startY = -bboxRadius + j * self._islandWidth

                pos = np.array([(startX, startY)])

                # Apply the rotation matrix and translate
                pos = np.matmul(R, pos.T)
                pos = pos.T + bboxCentre

                island = Island(origin=pos, orientation=theta_h,
                                islandWidth=self._islandWidth, islandOverlap=self._islandOverlap,
                                hatchDistance=self._hatchDistance)

                island.posId = (i, j)
                island.id = id
                islands.append(island)

                id += 1

        return islands

    def hatch(self, boundaryFeature) -> Layer:
        """
        Generates the Island Scan Strategy for a layer given a list of boundary features

        :param boundaryFeature: A list of boundary features

        :return: A layer containing the layer geometry
        """

        if len(boundaryFeature) == 0:
            return

        layer = Layer(0, 0)

        """
        First generate the boundary with the spot compensation applied including outer and inner contours
        """

        offsetDelta = 0.0
        offsetDelta -= self._spotCompensation

        for i in range(self._numOuterContours):
            offsetDelta -= self._contourOffset
            offsetBoundary = self.offsetBoundary(boundaryFeature, offsetDelta)

            for poly in offsetBoundary:
                for path in poly:
                    contourGeometry = ContourGeometry()
                    contourGeometry.coords = np.array(path)[:, :2]
                    contourGeometry.subType = "outer"
                    layer.geometry.append(contourGeometry)  # Append to the layer

        # Repeat for inner contours
        for i in range(self._numInnerContours):

            offsetDelta -= self._contourOffset
            offsetBoundary = self.offsetBoundary(boundaryFeature, offsetDelta)

            for poly in offsetBoundary:
                for path in poly:
                    contourGeometry = ContourGeometry()
                    contourGeometry.coords = np.array(path)[:, :2]
                    contourGeometry.subType = "inner"
                    layer.geometry.append(contourGeometry)  # Append to the layer

        # The final offset is applied to the boundary

        offsetDelta -= self._volOffsetHatch

        curBoundary = self.offsetBoundary(boundaryFeature, offsetDelta)

        scanVectors = []

        # Iterate through each closed polygon region in the slice. The currently individually sliced.

        # Hatch angle will change per layer
        # TODO change the layer angle increment
        layerHatchAngle = np.mod(self._hatchAngle + self._layerAngleIncrement, 180)

        # The layer hatch angle needs to be bound by +ve X vector (i.e. -90 < theta_h < 90 )
        if layerHatchAngle > 90:
            layerHatchAngle = layerHatchAngle - 180

        # Generate the square island sub regions
        islands = self.generateIslands(curBoundary, self._hatchAngle)

        # All Island sub-regions need to have an intersection test
        self.intersectIslands(curBoundary, islands)

        # Sort the islands using a basic sort
        sortedIslands = sorted(islands, key=lambda island: (island.posId[0], island.posId[1]) )

        # Structure for storing the hatch scan vectors
        clippedCoords = []
        unclippedCoords = []

        # Generate the hatches for all the islands
        idx = 0
        for island in sortedIslands:

            # Generate the hatches for each island subregion
            coords = island.hatch()

            # Note for sorting later the order of the hatch vector is updated based on the sortedIsland
            coords[:, 2] += idx

            if island.isIntersecting():
                if island.requiresClipping():
                    clippedCoords.append(coords)
                else:
                    unclippedCoords.append(coords)

            # Update the index by incremented by the number of hatches
            # ISSUE - the max coordinate id should be used to update this but it adds additional computiatonal complexity
            idx += coords.shape[0] / 2

        clippedCoords = np.vstack(clippedCoords)
        unclippedCoords = np.vstack(unclippedCoords).reshape(-1,2,3)

        # Clip the hatches of the boundaries to fill to the boundary
        clippedPaths = self.clipLines(curBoundary, clippedCoords)
        clippedPaths = np.array(clippedPaths)

        # Merge hatches from both groups together
        hatches = np.vstack([clippedPaths, unclippedCoords])
        clippedLines = self.clipperToHatchArray(hatches)

        # Merge the lines together
        if len(clippedPaths) > 0:

            # Extract only x-y coordinates and sort based on the pseudo-order stored in the z component.
            clippedLines = clippedLines[:, :, :3]
            id = np.argsort(clippedLines[:, 0, 2])
            clippedLines = clippedLines[id, :, :]

            scanVectors.append(clippedLines)


        if len(clippedLines) > 0:
            # Scan vectors have been created for the hatched region

            # Construct a HatchGeometry containing the list of points
            hatchGeom = HatchGeometry()

            # Only copy the (x,y) points from the coordinate array.
            hatchVectors = np.vstack(scanVectors)
            hatchVectors  = hatchVectors[:, :, :2].reshape(-1, 2)

            # Note the does not require positional sorting
            if self.hatchSortMethod:
                hatchVectors = self.hatchSortMethod.sort(hatchVectors)

            hatchGeom.coords = hatchVectors

            layer.geometry.append(hatchGeom)

        return layer

    def intersectIslands(self, paths, islands: List[Island]) -> Tuple[Any, Any]:
        """
        Perform the intersection and overlap tests on the island sub regions. This should be performed before any
        clipping operations are performed.

        :param paths: List of coordinates describing the boundary
        :param islands: A list of Islands to have the intersection and overlap test

        :return: A tuple containing lists of clipped and unClipped islands
        """
        polys = []
        for path in paths:
            polys += pathsToClosedPolygons(path)

        poly = MultiPolygon(polys)

        intersectIslands = []
        overlapIslands = []

        intersectIslandsSet = set()
        overlapIslandsSet = set()

        for i in range(len(islands)):

            island = islands[i]
            s = island.boundary()

            if poly.overlaps(s):
                overlapIslandsSet.add(i)  # id
                overlapIslands.append(island.boundary)
                island.setRequiresClipping(True)

            if poly.intersects(s):
                intersectIslandsSet.add(i)  # id
                intersectIslands.append(island.boundary)
                island.setIntersecting(True)


        unTouchedIslandSet = intersectIslandsSet - overlapIslandsSet
        unTouchedIslands = [islands[i] for i in unTouchedIslandSet]

        return overlapIslandsSet, unTouchedIslandSet