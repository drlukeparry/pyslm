"""
A simple example showing how t use PySLM for generating a Stripe Scan Strategy across a single layer.
"""
import numpy as np
import pyslm
import pyslm.analysis.utils as analysis
from pyslm import hatching as hatching

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
z = 1.0

# Create a BasicIslandHatcher object for performing any hatching operations (
myHatcher = hatching.BasicIslandHatcher()
myHatcher.islandWidth = 3.0
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
The boundaries of the slice can be automatically simplified if desired. 
"""
geomSlice = solidPart.getVectorSlice(z, simplificationFactor=0.1)
layer = myHatcher.hatch(geomSlice)

"""
Note the hatches are ordered sequentially across the stripe. Additional sorting may be required to ensure that the
the scan vectors are processed generally in one-direction from left to right.
The stripes scan strategy will tend to provide the correct order per isolated region.
"""

"""
Plot the layer geometries using matplotlib
The order of scanning for the hatch region can be displayed by setting the parameter (plotOrderLine=True)
Arrows can be enables by setting the parameter plotArrows to True
"""

pyslm.visualise.plot(layer, plot3D=False, plotOrderLine=True, plotArrows=False)

"""
Before exporting or analysing the scan vectors, a model and build style need to be created and assigned to the 
LaserGeometry groups.

The user has to assign a model (mid)  and build style id (bid) to the layer geometry
"""

for layerGeom in layer.geometry:
    layerGeom.mid = 1
    layerGeom.bid = 1

bstyle = pyslm.geometry.BuildStyle()
bstyle.bid = 1
bstyle.laserSpeed = 200 # [mm/s]
bstyle.laserPower = 200 # [W]

model = pyslm.geometry.Model()
model.mid = 1
model.buildStyles.append(bstyle)

"""
Analyse the layers using the analysis module. The path distance and the estimate time taken to scan the layer can be
predicted.
"""
print('Total Path Distance: {:.1f} mm'.format(analysis.getLayerPathLength(layer)))
print('Total jump distance {:.1f} mm'.format(analysis.getLayerJumpLength(layer)))
print('Time taken {:.1f} s'.format(analysis.getLayerTime(layer, model)) )

