"""
A simple example showing how it is possible to import a file using libSLM, visualise and extract some information
about the file.

The following example works with a Renishaw .mtt file. The user needs to specify the file to analyse,.
"""

import pyslm
import pyslm.analysis
import pyslm.visualise

import numpy as np

"""
The structures necessary for creating a machine build file should be imported from the geometry submodule. 
Fallback python classes that are equivalent to those in libSLM are provided to ensure prototyping can take place.
"""
from pyslm import geometry as slm

"""
Import the MTT (Renishaw SLM) Exporter
"""
from libSLM import mtt

"Create the initial .mtt reader translator and set the file path "
mttReader = mtt.Reader()
mttReader.setFilePath("20201030_Cube_Luke_CW_M4.mtt")

"""
Parse and read the entire file.
Note: The entire build file is read, including all layers so this may take some time for very large files and are directly
transferred into memory
"""
#
mttReader.parse()


# Get layers (note a reference is always used in Python)
layers = mttReader.layers

# Get the models and  associated build styles
models = mttReader.models

""" Visualise the layer geometry for the first layer """
print('Total Path Distance: {:.1f} mm'.format(pyslm.analysis.getLayerPathLength(layers[0])))
print('Total Layer Scan Time: {:.1f} s'.format(pyslm.analysis.getLayerTime(layers[0], models)) )

"""
A wrapped lambda function is used to curry the existing models list and locate for each hatch geometry group
the laser id used to scan the hatch geometry.
"""
def plotLaserId(models, hatchGeom):
    buildStyle = pyslm.analysis.utils.getBuildStyleById(models, hatchGeom.mid, hatchGeom.bid)
    return np.tile(buildStyle.laserId, [int(len(hatchGeom.coords)/2),1])


(fig, ax) = pyslm.visualise.plot(layers[0], plot3D=False, plotOrderLine=True, plotArrows=False,
                                            index=lambda hatchGeom :plotLaserId(models, hatchGeom) )
