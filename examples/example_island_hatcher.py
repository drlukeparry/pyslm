"""
A simple example showing how to use PySLM  with the IslandHatcher approach, which decomposes the layer into several
island regions, which are tested for intersection and then the hatches generated are more efficiently clipped.
"""

import numpy as np
import time

from shapely.geometry import MultiPolygon

import pyslm
import pyslm.visualise
from pyslm import hatching as hatching

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
myHatcher = hatching.IslandHatcher()
myHatcher.islandWidth = 5.0
myHatcher.islandOverlap = -0.1

# Set the base hatching parameters which are generated within Hatcher
myHatcher.hatchAngle = 10
myHatcher.volumeOffsetHatch = 0.08
myHatcher.spotCompensation = 0.06
myHatcher.numInnerContours = 2
myHatcher.numOuterContours = 1
myHatcher.hatchSortMethod = hatching.AlternateSort()

# The traditional approach is to get the path ring coordinates and pass this to Island Hatcher
geomSlice = solidPart.getVectorSlice(z)

"""
Set to True to show the general process for how IslandHatcher works. 
Note: the slowest part is plotting all the polygon islands following the testing
"""
if True:
    # Generates a set of square islands which is guaranteed to cover the entire area of the boundaries.
    # The global orientation of the angle is provided as the second argument
    islands = myHatcher.generateIslands(geomSlice, 30)

    # The user can extract the ids of all the ids that are clipped or not clipped of the islands
    # The boundary should be provided to be clipped against.
    a, b = myHatcher.intersectIslands(geomSlice, islands)

    overlapIslands = [islands[i] for i in a]
    intersectIslands = [islands[i] for i in b]

    # The above intersectIsland internal method can also be achieved using the following approach below.

    # Get the Shapely Polygons from slicing the part
    poly = solidPart.getVectorSlice(z, False)

    # Use shapely MultiPolygon collection to allow full testing and clipping across all boundary regions
    poly = MultiPolygon(poly)

    intersectIslands = []
    overlapIslands = []

    # Python sets are used to perform boolean operations on a set to identify unclipped islands
    intersectIslandsSet = set()
    overlapIslandsSet= set()

    # Iterate across all the islands
    for i in range(len(islands)):

        island = islands[i]
        s = island.boundary()

        if poly.overlaps(s):
            overlapIslandsSet.add(i) # id
            overlapIslands.append(island)

        if poly.intersects(s):
            intersectIslandsSet.add(i)  # id
            intersectIslands.append(island)

    unTouchedIslandSet = intersectIslandsSet-overlapIslandsSet
    unTouchedIslands = [islands[i] for i in unTouchedIslandSet]

    print('Finished Island Clipping')

    fig, ax = pyslm.visualise.plotPolygon(geomSlice)

    # Plot using visualise.plotPolygon the original islands generated before intersection
    for island in islands:
        x, y = island.boundary().exterior.xy
        pyslm.visualise.plotPolygon([np.vstack([x,y]).T], handle=(fig, ax))

    for island in intersectIslands:
        x, y = island.boundary().exterior.xy
        pyslm.visualise.plotPolygon([np.vstack([x,y]).T], handle=(fig, ax),  plotFilled=True, lineColor='g', fillColor = '#19aeffff')

    for island in overlapIslands:
        x, y = island.boundary().exterior.xy
        pyslm.visualise.plotPolygon([np.vstack([x, y]).T], handle=(fig, ax), plotFilled=True, lineColor='b', fillColor = '#ff4141ff')


startTime = time.time()

# Perform the complete hatching operation
print('Hatching Started')

layer = myHatcher.hatch(geomSlice)
print('Completed Hatching')


"""
Plot the layer geometries using matplotlib
The order of scanning for the hatch region can be displayed by setting the parameter (plotOrderLine=True)
Arrows can be enables by setting the parameter plotArrows to True
"""
pyslm.visualise.plot(layer, plot3D=False, plotOrderLine=True, plotArrows=False)
