"""
A simple example showing how to use PySLM for generating slices across a 3D model
"""
import numpy as np
import pyslm
import pyslm.visualise
import pyslm.geometry
import pyslm.analysis
from pyslm import hatching as hatching#


# Imports the part and sets the geometry to  an STL file (frameGuide.stl)
solidPart = pyslm.Part('inversePyramid')
solidPart.setGeometry('../models/inversePyramid.stl')

solidPart.origin[0] = 5.0
solidPart.origin[1] = 2.5
solidPart.scaleFactor = 1.0
solidPart.rotation = [0, 0.0, 45]
solidPart.dropToPlatform()

# Create a StripeHatcher object for performing any hatching operations
myHatcher = hatching.BasicIslandHatcher()

# Set the base hatching parameters which are generated within Hatcher
myHatcher.hatchAngle = 10
myHatcher.volumeOffsetHatch = 0.08
myHatcher.spotCompensation = 0.06
myHatcher.numInnerContours = 2
myHatcher.numOuterContours = 1
myHatcher.hatchSortMethod = hatching.AlternateSort()

# Set the layer thickness
layerThickness = 0.04 # [mm]

# Perform the slicing. Return coords paths should be set so they are formatted internally.
#myHatcher.layerAngleIncrement = 66.7

#Perform the hatching operations
print('Hatching Started')

layers = []

# Create an individual part for each sample
model = pyslm.geometry.Model()
model.mid = 1
model.name = "Sample {:d}".format(1)

bstyle = pyslm.geometry.BuildStyle()
bstyle.setStyle(bid=1,
                focus=0, power=200.0,
                pointExposureTime=80, pointExposureDistance=50,
                laserMode=pyslm.geometry.LaserMode.Pulse)

model.buildStyles.append(bstyle)

layerId = 1
for z in np.arange(0, solidPart.boundingBox[5], layerThickness):

    # Typically the hatch angle is globally rotated per layer by usually 66.7 degrees per layer
    myHatcher.hatchAngle += 66.7
    # Slice the boundary
    geomSlice = solidPart.getVectorSlice(z)

    # Hatch the boundary using myHatcher
    layer = myHatcher.hatch(geomSlice)

    for geom in layer.geometry:
        geom.mid = 1
        geom.bid = 1

    # The layer height is set in integer increment of microns to ensure no rounding error during manufacturing
    layer.z = int(z*1000)
    layer.layerId = layerId
    model.topLayerId = layerId
    layers.append(layer)
    layerId += 1

print('Completed Hatching')

# Plot the layer geometries using matplotlib
# Note: the use of python slices to get the arrays

(fig, ax) = pyslm.visualise.plot(layers[-1], plot3D=False, plotOrderLine=True, plotArrows=True)

"""
Create the Scan Iterator based on the laser parameters within the list of models and the geometry stored within the layer.
Note, the laser point exposures generated across the contour and hatch are linearly interpolated across each individual
scan vector based on the timestep.
"""
scanIter = pyslm.analysis.ScanIterator([model], layers)

# Set the parameters for the scan iterator across the layer and also the timestep used.
scanIter.recoaterTime = 10 # s
scanIter.timestep = 5e-3

"""
An iterator function is generated, so that incrementally exposure points may be collected incrementally with
pythonic notation
"""

# Generate a list of point exposures - note the 3rd column is the current time
ab = np.array([point for point in scanIter])

# reset to layer one
scanIter.seekByLayer(1)
print("Current time at layer (1): {:.3f})".format(scanIter.time))

# Seek based on the time
scanIter.seek(time=0.4)

print("Current layer is {:d} @ time = 0.4s".format(scanIter.getCurrentLayer().layerId))

# Get the current laser state (position, laser parameters, firing)
laserX, laserY = scanIter.getCurrentLaserPosition()
laserOn = scanIter.isLaserOn()
bstyle = scanIter.getCurrentBuildStyle()

"""
Other useful metrics are cached such as the total build time
"""
totalBuildTime = scanIter.getBuildTime()

print('Total number of layers: {:d}'.format(len(layers)))

print('Total Build Time: {:.1f}s ({:.1f}hr)'.format(totalBuildTime, totalBuildTime/3600))

