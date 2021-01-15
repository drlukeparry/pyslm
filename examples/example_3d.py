"""
A simple example showing how to use PySLM for generating 3D vmodel
"""
import pyslm
from pyslm import hatching as hatching
import numpy as np

# Imports the part and sets the geometry to  an STL file (frameGuide.stl)
solidPart = pyslm.Part('inversePyramid')
solidPart.setGeometry('../models/inversePyramid.stl')

solidPart.origin[0] = 5.0
solidPart.origin[1] = 2.5
solidPart.scaleFactor = 1.0
solidPart.rotation = [0, 0.0, 45]
solidPart.dropToPlatform()
print(solidPart.boundingBox)

# Create a StripeHatcher object for performing any hatching operations
myHatcher = hatching.Hatcher()

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

for z in np.arange(0, solidPart.boundingBox[5], layerThickness):

    # Typically the hatch angle is globally rotated per layer by usually 66.7 degrees per layer
    myHatcher.hatchAngle += 66.7

    # Slice the boundary
    geomSlice = solidPart.getVectorSlice(z)

    # Hatch the boundary using myHatcher
    layer = myHatcher.hatch(geomSlice)

    # The layer height is set in integer increment of microns to ensure no rounding error during manufacturing
    layer.z = int(z*1000)
    layers.append(layer)

print('Completed Hatching')

# Plot the layer geometries using matplotlib
# Note: the use of python slices to get the arrays
pyslm.visualise.plotLayers(layers[0:-1:10])


