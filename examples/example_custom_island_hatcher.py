"""
A simple example showing how to use PySLM with the IslandHatcher approach, which decomposes the layer into several
island regions, which are tested for intersection and then the hatches generated are more efficiently clipped.
"""

import numpy as np
import time

from shapely.geometry.polygon import LineString, LinearRing, Polygon
from shapely.geometry import LinearRing, MultiPolygon, Polygon
from shapely.geometry import MultiPolygon

import pyslm
import pyslm.visualise
from pyslm import hatching as hatching


class HexIsland(hatching.Island):
    """
    Derive the HexIsland subclass from the Island
    """
    def __init__(self, origin:np.ndarray = np.array([[0.0,0.0]]), orientation = 0.0,
                       islandWidth = 0.0, islandOverlap = 0.0,
                       hatchDistance = 0.1):

        super().__init__(origin = origin, orientation=orientation,
                         islandWidth=islandWidth, islandOverlap=islandOverlap,
                         hatchDistance=hatchDistance)

    def localBoundary(self) -> np.ndarray:
        # Redefine the local boundary to be the hexagon shape

        if HexIsland._boundary is None:
            # Simple approach is to use a radius to define the overall island size
            #radius = np.sqrt(2*(self._islandWidth*0.5 + self._islandOverlap)**2)

            numPoints = 6

            radius = self._islandWidth / np.cos(np.pi/numPoints)  / 2 + self._islandOverlap

            print('island', radius, self._islandWidth)


            # Generate polygon island
            coords = np.zeros((numPoints+1, 2))

            for i in np.arange(0,numPoints):
                # Subtracting -0.5 orientates the polygon along its face
                angle = (i-0.5)/numPoints*2*np.pi
                coords[i] = [np.cos(angle), np.sin(angle)]

            # Close the polygon
            coords[-1] = coords[0]

            # Scale the polygon
            coords *= radius

            # Assign to the static class attribute
            HexIsland._boundary = coords

        return HexIsland._boundary

    def generateInternalHatch(self, isOdd = True) -> np.ndarray:
        """
        Generates a set of hatches orthogonal to the island's coordinate system :math:`(x\\prime, y\\prime)`.

        :param isOdd: The chosen orientation of the hatching
        :return: (nx3) Set of sorted hatch coordinates
        """

        numPoints = 6

        radius = self._islandWidth / np.cos(np.pi / numPoints) / 2 + self._islandOverlap

        startX = -radius
        startY = -radius

        endX = radius
        endY = radius

        # Generate the basic hatch lines to fill the island region
        x = np.tile(np.arange(startX, endX, self._hatchDistance).reshape(-1, 1), 2).flatten()
        y = np.array([startY, endY])
        y = np.resize(y, x.shape)

        z = np.arange(0, y.shape[0] / 2, 0.5).astype(np.int64)

        coords =  np.hstack([x.reshape(-1, 1),
                             y.reshape(-1, 1),
                             z.reshape(-1,1)])

        # Toggle
        theta_h = np.deg2rad(90.0) if isOdd else np.deg2rad(0.0)

        # Create the 2D rotation matrix with an additional row, column to preserve the hatch order
        c, s = np.cos(theta_h), np.sin(theta_h)
        R = np.array([(c, -s, 0),
                      (s, c, 0),
                      (0, 0, 1.0)])

        # Apply the rotation matrix and translate to bounding box centre
        coords = np.matmul(R, coords.T).T

        # Clip the hatch fill to the boundary
        boundary = [self.localBoundary()]
        clippedLines = np.array(hatching.BaseHatcher.clipLines(boundary, coords))

        # Sort the hatches
        clippedLines = clippedLines[:, :, :3]
        id = np.argsort(clippedLines[:, 0, 2])
        clippedLines = clippedLines[id, :, :]

        # Convert to a flat 2D array of hatches and resort the indices
        coordsUp = clippedLines.reshape(-1,3)
        coordsUp[:,2] = np.arange(0, coordsUp.shape[0] / 2, 0.5).astype(np.int64)
        return coordsUp


class HexIslandHatcher(hatching.IslandHatcher):

    def __init__(self):
        super().__init__()

    def generateIslands(self, paths, hatchAngle: float = 90.0):
        """
        Generate a series of tessellating Hex Islands to fill the region. For now this requires re-implementing because
        the boundaries of the island may be different shapes and require a specific placement in order to correctly
        tessellate within a region.
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

        # Number of sides of the polygon island
        numPoints = 6

        # Construct a square which wraps the radius
        numIslandsX = int(2 * bboxRadius / self._islandWidth) + 1
        numIslandsY = int(2 * bboxRadius / ((self._islandWidth + self._islandOverlap) * np.sin(2*np.pi/numPoints)) )+ 1

        # Create the rotation matrix
        c, s = np.cos(theta_h), np.sin(theta_h)
        R = np.array([(c, -s),
                      (s, c)])

        islands = []
        id = 0

        print('Island width:', self._islandWidth)

        for i in np.arange(0, numIslandsX):
            for j in np.arange(0, numIslandsY):

                # gGenerate the island position
                startX = -bboxRadius + i * self._islandWidth + np.mod(j, 2) * self._islandWidth / 2
                startY = -bboxRadius + j * self._islandWidth * np.sin(2*np.pi/numPoints)

                pos = np.array([(startX, startY)])

                # Apply the rotation matrix and translate to bounding box centre
                pos = np.matmul(R, pos.T)
                pos = pos.T + bboxCentre

                # Generate a HexIsland and append to the island
                island = HexIsland(origin=pos, orientation=theta_h,
                                  islandWidth=self._islandWidth, islandOverlap=self._islandOverlap,
                                  hatchDistance=self._hatchDistance)

                island.posId = (i, j)
                island.id = id
                islands.append(island)

                id += 1

        return islands


# Imports the part and sets the geometry to  an STL file (frameGuide.stl)
solidPart = pyslm.Part('inversePyramid')
solidPart.setGeometry('../models/frameGuide.stl')
solidPart.dropToPlatform()

solidPart.origin[0] = 5.0
solidPart.origin[1] = 2.5
solidPart.scaleFactor = 2.0
solidPart.rotation = [0, 0.0, np.pi]

# Set te slice layer position
z = 14.99

# Create a StripeHatcher object for performing any hatching operations
myHatcher = HexIslandHatcher()
myHatcher.islandWidth = 5.0
myHatcher.islandOverlap = -0.15

# Set the base hatching parameters which are generated within Hatcher
myHatcher.hatchAngle = 00
myHatcher.volumeOffsetHatch = 0.08
myHatcher.spotCompensation = 0.06
myHatcher.numInnerContours = 2
myHatcher.numOuterContours = 1
myHatcher.hatchSortMethod = hatching.AlternateSort()

geomSlice = solidPart.getVectorSlice(z)


startTime = time.time()

#Perform the complete hatching operation


fig, ax = pyslm.visualise.plotPolygon(geomSlice)

# turn True to preview the hex island hatches

if True:
    islands = myHatcher.generateIslands(geomSlice, hatchAngle=0)

    # Plot using visualise.plotPolygon the original islands generated before intersection
    for island in islands:
        x, y = island.boundary().exterior.xy
        pyslm.visualise.plotPolygon([np.vstack([x, y]).T], handle=(fig, ax))

print('Hatching Started')
layer = myHatcher.hatch(geomSlice)
print('Completed Hatching')


"""
Plot the layer geometries using matplotlib
The order of scanning for the hatch region can be displayed by setting the parameter (plotOrderLine=True)
Arrows can be enables by setting the parameter plotArrows to True
"""
pyslm.visualise.plot(layer, plot3D=False, plotOrderLine=True, plotArrows=False)
