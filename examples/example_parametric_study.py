"""
A simple example showing how to use PySLM for generating a Design of Experiment (DOE) study for characterising the
material properties for selected laser / design paramters in Selective Laser Melting. Remember to install doepy
using pip install
"""

import numpy as np

import matplotlib.pyplot as plt

import doepy.build

import pyslm
import pyslm.geometry as geometry
import pyslm.visualise
import pyslm.analysis
import pyslm.analysis
from pyslm import hatching as hatching


"""
Imports the part and sets the geometry to  an STL file (inversePyramid.stl). This is used as the base object for
generating all the samples during the design of experiment
"""

solidPart = pyslm.Part('inversePyramid')
solidPart.setGeometry('../models/inversePyramid.stl')

solidPart.origin[0] = 5.0
solidPart.origin[1] = 2.5
solidPart.scaleFactor = 1.0
solidPart.rotation = [0, 0.0, 45]
solidPart.dropToPlatform()

# Create a Hatcher object for performing any hatching operations
myHatcher = hatching.Hatcher()

# Set the base hatching parameters which are generated within Hatcher
myHatcher.hatchAngle = 45
myHatcher.volumeOffsetHatch = 0.08
myHatcher.spotCompensation = 0.06
myHatcher.numInnerContours = 2
myHatcher.numOuterContours = 1
myHatcher.hatchSortMethod = hatching.AlternateSort()

# Set the layer thickness
layerThickness = 0.4 # [mm]

# Perform the slicing. Return coords paths should be set so they are formatted internally.
#myHatcher.layerAngleIncrement = 66.7

#Perform the hatching operations
print('Hatching Started')

"""
Generate the header
"""
header = geometry.Header()
header.filename = "MTT Layerfile"
header.version = (1, 2)
header.zUnit = 1000

layers = {}
models = []

"""
Generate test position matrix for the samples both spatially in the X and Y directions
"""
numSamples = (4,4)

"""
Create the design of experiment using a space filling curve. The design of experiment in this case use the latin
hypercube space-filling to cover the three variables (laser power, laser speed and hatch distance). This could 
alternatively be an exhaustive bruteforce search to cover the parameter space.
"""
doe3 = doepy.build.space_filling_lhs(
{
    'laserPower':[40,70],
    'laserSpeed':[200, 500],
    'hatchDistance':[0.1,0.4]
}, num_samples = numSamples[0] * numSamples[1])

# Convert explicitly the DOE array numpy array for all the laser parameters tested
laserParameters = doe3.to_numpy()

"""
Set the spatial positions for the samples in the DOE
"""
delta = (30,30) # offset between samples
offset = 1.2 # Offset applied across each row
buildPlateSize = np.array([[20,200],[20,200]])

# Generate the positions across the build plate
x,y = np.meshgrid(np.linspace(buildPlateSize[0,0], buildPlateSize[0,1], numSamples[0]),
                  np.linspace(buildPlateSize[1,0], buildPlateSize[1,1], numSamples[1]))

xOffset = np.arange(0, numSamples[1]) * 1.2 * solidPart.extents[0]
x += xOffset.reshape(-1,1)

modelId = 0

for i in range(numSamples[0]):
    for j in range(numSamples[1]):

        # Now iterating across each sample

        modelId += 1
        solidPart.origin[:2] = np.array([x[j,i],y[j,i]])

        print('Processing solid part [{:d}]'.format(modelId))
        solidPart.regenerate()

        # Create an individual part for each sample
        model = geometry.Model()
        model.mid = modelId
        model.name = "Sample {:d}".format(modelId)
        model.topLayerId = 0
        models.append(model)

        """
        A BuildStyle represents a set of laser parameters used for scanning a particular layer geometry. The set of parameters
        typically includes laser power, laser speed, exposure times and distances for Q-Switched laser sources. 
        Each BuildStyle should have a unique id (.bid) for each mid used.
        """
        bstyle = geometry.BuildStyle()

        # Get the laser parmaters from the Pandas Dataset
        laserParameters = doe3.iloc[modelId - 1]
        bstyle.bid = 1

        # Set the laser parameters for the Contour Build Style
        bstyle.laserPower = laserParameters.laserPower # W
        bstyle.laserSpeed = laserParameters.laserSpeed  # mm/s - Note this is used on some systems but should be set
        bstyle.laserFocus = 0.0  # mm - (Optional) Some new systems can modify the focus position real-time.
        bstyle.pointDistance = 50  # μm - Distance between exposure points
        bstyle.pointExposureTime = 80  # ms - Exposure time
        bstyle.laserId = 1
        bstyle.laserMode = geometry.LaserMode.Pulse  # Non-continious laser mode (=1), CW laser mode = (0)

        model.buildStyles.append(bstyle)

        """
        We can also alter the hatch parameters per sample
        """
        myHatcher.hatchDistance = laserParameters.hatchDistance

        # Counter for the layer id
        layerId = 0
        for z in np.arange(0, solidPart.boundingBox[5], layerThickness):

            # Typically the hatch angle is globally rotated per layer by usually 66.7 degrees per layer
            myHatcher.hatchAngle += 66.7
            # Slice the boundary
            geomSlice = solidPart.getVectorSlice(z)

            # Hatch the boundary using myHatcher
            layer = myHatcher.hatch(geomSlice)

            for geo in layer.geometry:
                geo.mid = model.mid
                geo.bid = bstyle.bid

            # The layer height is set in integer increment of microns to ensure no rounding error during manufacturing
            layer.z = int(layerThickness * 1000 * layerId)  # [mu m]
            layerId += 1
            layer.layerId = layerId

            """
            Create a new layer if this doesn't exist otherwise append the layer geometries generated into the previous
            generated layer.
            """
            if not layers.get(layer.z):
                layers[layer.z] = layer
            else:
                layers[layer.z].geometry.extend(layer.geometry)

        model.topLayerId = layerId

layerList = list(layers.values())

print('Completed Hatching')

def plotLaserPower(models, hatchGeom):
    buildStyle = pyslm.geometry.getBuildStyleById(models, hatchGeom.mid, hatchGeom.bid)
    return np.tile(buildStyle.laserPower, [int(len(hatchGeom.coords)/2),1])

"""
Plot the layer geometries using matplotlib
Note: the use of python slices to get the arrays
"""
fig, ax = pyslm.visualise.plot(layerList[-1], plot3D=False, plotOrderLine=True, plotArrows=False, plotColorbar=True,
                                              index=lambda hatchGeom :plotLaserPower(models, hatchGeom))

"""
Plot the text lables for each parameter in the DOE
"""
modelId = 0
for i in range(numSamples[0]):
    for j in range(numSamples[1]):
        ax.text(x[i,j],y[i,j]-20,
                "Model {:d} \n Laser Power({:.1f}) \n Laser Speed ({:.1f} \n Hatch Dist ({:.1f})".format(modelId+1,
                                                                                  doe3.iloc[modelId-1].laserPower,
                                                                                  doe3.iloc[modelId-1].laserSpeed,
                                                                                  doe3.iloc[modelId-1].hatchDistance),
                 fontsize=8, ha='center')
        modelId += 1

# Turn the grid on
ax.grid()

""" Validate the input model """
modelValidator = pyslm.geometry.ModelValidator()
modelValidator.validateBuild(models, layerList)

