"""
A simple example showing how to use PySLM for generating a Stripe Scan Strategy across a single layer and
visualising the point exposures across the scan vectors and the 'effective' heat or exposure map of the layer.
"""
import numpy as np
import pyslm
import pyslm.visualise
from pyslm import hatching as hatching
import matplotlib.pyplot as plt

# Imports the part and sets the geometry to  an STL file (frameGuide.stl)
solidPart = pyslm.Part('myFrameGuide')
solidPart.setGeometry('../models/frameGuide.stl')

"""
Transform the part:
Rotate the part 30 degrees about the Z-Axis - given in degrees
Translate by an offset of (5,10) and drop to the platform the z=0 Plate boundary
"""
solidPart.origin = [5.0, 10.0, 0.0]
solidPart.rotation = np.array([0, 0, 30])
solidPart.dropToPlatform()

print(solidPart.boundingBox)

# Set te slice layer position
z = 23.

# Create a BasicIslandHatcher object for performing any hatching operations (
myHatcher = hatching.BasicIslandHatcher()
myHatcher.stripeWidth = 5.0

# Set the base hatching parameters which are generated within Hatcher
myHatcher.hatchAngle = 10 # [°] The angle used for the islands
myHatcher.volumeOffsetHatch = 0.08 # [mm] Offset between internal and external boundary
myHatcher.spotCompensation = 0.06 # [mm] Additional offset to account for laser spot size
myHatcher.numInnerContours = 2
myHatcher.numOuterContours = 1
myHatcher.hatchSortMethod = hatching.AlternateSort()

"""
Perform the slicing. Return coords paths should be set so they are formatted internally.
This is internally performed using Trimesh to obtain a closed set of polygons.
Further polygon simplification may be required to reduce excessive number of edges in the boundaries.
"""
geomSlice = solidPart.getVectorSlice(z)
layer = myHatcher.hatch(geomSlice)

# we have to assign a model and build style id to the layer geometry
for layerGeom in layer.geometry:
    layerGeom.mid = 1
    layerGeom.bid = 1

bstyle = pyslm.geometry.BuildStyle()
bstyle.bid = 1
bstyle.laserSpeed = 200.0 # [mm/s]
bstyle.laserPower = 200 # [W]#
bstyle.pointDistance = 60 # (60 microns)
bstyle.pointExposureTime = 30 #

model = pyslm.geometry.Model()
model.mid = 1
model.buildStyles.append(bstyle)

resolution = 0.2

"""
Generate the exposure points for the layer given the hatch layer geometry, and point exposure parameters (pointDistance)
 specific in the specific model buildstyle assigned to each layer geometry
"""
exposurePoints = pyslm.hatching.getExposurePoints(layer, model)


# Plot the heatmap based on the point exposure and the chosen resolution
# Currently the part and z-layer is required to generate a bitmap which covers the part geometry.
fig, ax = pyslm.visualise.plotHeatMap(solidPart, z, exposurePoints,  resolution)

# Plot all the exposure points
fig, ax = plt.subplots()
ax.axis('equal')

plt.scatter(exposurePoints[:,0], exposurePoints[:,1], marker='o', linestyle='None')

# Plot the exposure points
pyslm.visualise.plot(layer, plot3D=False, handle=(fig,ax))

# Plot the corresponding layers
pyslm.visualise.plot(layer, plot3D=False, plotOrderLine=True, plotArrows=False)

