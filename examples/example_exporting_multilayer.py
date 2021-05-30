"""
A simple example showing the basic structure and layout required for generating a machine build file for a SLM System.
This is automatically built into the hatching classes, but for simple experiments or simulations it can be useful to
create these structures manually.

The following example demonstrate the overall structure for creating a multi-layer build file, hatching a 3D model
geometry and using a Stripe Hatch Scan Strategy. The file is exported using the Renishaw .mtt translator and then
imported to visualise the layer.
"""

import pyslm
import pyslm.analysis
import pyslm.visualise
from pyslm import hatching as hatching
import numpy as np

from libSLM.translators import mtt

from pyslm import geometry

solidPart = pyslm.Part('nut')
solidPart.setGeometry('../models/nut.stl')
solidPart.origin = [5.0, 10.0, 0.0]
solidPart.dropToPlatform()

# Create a StripeHatcher object for performing any hatching operations
myHatcher = hatching.StripeHatcher()
myHatcher.stripeWidth = 5.0

# Set the base hatching parameters which are generated within Hatcher
myHatcher.hatchAngle = 10  # [°]
myHatcher.volumeOffsetHatch = 0.08  # [mm]
myHatcher.spotCompensation = 0.06  # [mm]
myHatcher.numInnerContours = 2
myHatcher.numOuterContours = 1

"""
Create the model:
A model represents a container with a unique ID (.mid) which has a set of BuildStyles. The combination of both the
Model and BuildStyle are assigned to various LayerGeometry features within the build file. 

:note:
    For each model, the top layer ID that contains a child buildstyle must be included. 
"""
model = geometry.Model()
model.mid = 1
model.name = "Nut"
model.topLayerId = 0

"""
Generate the header
"""
header = geometry.Header()
header.filename = "MTT Layerfile"
header.version = (1, 2)
header.zUnit = 1000

"""
A BuildStyle represents a set of laser parameters used for scanning a particular layer geometry. The set of parameters
typically includes laser power, laser speed, exposure times and distances for Q-Switched laser sources. 
Each BuildStyle should have a unique id (.bid) for each mid used.
"""
BuildStyle = geometry.BuildStyle()

"""
Note MTT Build Styles:
3 - Border
71 - Additional Border
12 - Fill Contour
11 - Fill Hatch
24 - Upskin fill hatch
"""

BuildStyle.bid = 1

# Set the laser parameters for the Contour Build Style
BuildStyle.laserPower = 200  # W
BuildStyle.laserSpeed = 500  # mm/s - Note this is used on some systems but should be set
BuildStyle.laserFocus = 0.0  # mm - (Optional) Some new systems can modify the focus position real-time.

"""
The point exposure parameters are specified for some systems (typically Q-Switch Pulse Lasers)
:Note:
    laser speed v = pointDistance / pointExposureTime
"""
BuildStyle.pointDistance = 50  # μm - Distance between exposure points
BuildStyle.pointExposureTime = 80  # ms - Exposure time
BuildStyle.laserId = 1
BuildStyle.laserMode = geometry.LaserMode.Pulse # Non-continious laser mode (=1), CW laser mode = (0)

model.buildStyles.append(BuildStyle)

layer_list = []
layerId = 0

for i in np.arange(0.0, solidPart.boundingBox[5], 0.5):
    myHatcher.hatchAngle += 66.7 # [deg]

    geomSlice = solidPart.getVectorSlice(i)
    layer = myHatcher.hatch(geomSlice)

    for geo in layer.geometry:
        geo.mid = model.mid
        geo.bid = BuildStyle.bid

    # The layer height is set in integer increment of microns to ensure no rounding error during manufacturing
    layer.z = int(30*i) # [mu m]
    layer.layerId = layerId
    layer_list.append(layer)

    layerId += 1


model.topLayerId = layerId - 1

""" Validate the input model """
pyslm.geometry.ModelValidator.validateBuild([model], layer_list)

from libSLM import mtt

"Create the initial object"
mttWriter = mtt.Writer()
mttWriter.setFilePath("build.mtt")
mttWriter.write(header, [model], layer_list)

""" Read the exported file to verify """
mttReader = mtt.Reader()
mttReader.setFilePath("build.mtt")
mttReader.parse()

readLayers = mttReader.layers
modelRead = mttReader.models

""" 
Plot the laser id used for each hatch vector used. A lambda function is used to plot this. 
"""
def plotLaserId(models, hatchGeom):
    buildStyle = pyslm.analysis.utils.getBuildStyleById(models, hatchGeom.mid, hatchGeom.bid)
    return np.tile(buildStyle.laserId, [int(len(hatchGeom.coords)/2),1])


(fig, ax) = pyslm.visualise.plot(readLayers[0], plot3D=False, plotOrderLine=True, plotArrows=False,
                                            index=lambda hatchGeom :plotLaserId([model], hatchGeom) )
