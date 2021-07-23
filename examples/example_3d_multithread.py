"""
A simple example showing how to use PySLM for generating slices across a 3D model.
THhs example takes advantage of the multi-processing module to run across multiple threads.
"""
import pyslm
import pyslm.visualise
from pyslm import hatching as hatching
import numpy as np
import time

from multiprocessing import Manager
from multiprocessing.pool import Pool
from multiprocessing import set_start_method

def calculateLayer(input):
    # Typically the hatch angle is globally rotated per layer by usually 66.7 degrees per layer
    d = input[0]
    zid= input[1]

    layerThickness = d['layerThickness']
    solidPart = d['part']

    # Create a StripeHatcher object for performing any hatching operations
    myHatcher = hatching.Hatcher()

    # Set the base hatching parameters which are generated within Hatcher
    layerAngleOffset = 66.7
    myHatcher.hatchAngle = 10 + zid * 66.7
    myHatcher.volumeOffsetHatch = 0.08
    myHatcher.spotCompensation = 0.06
    myHatcher.numInnerContours = 2
    myHatcher.numOuterContours = 1
    myHatcher.hatchSortMethod = hatching.AlternateSort()

    #myHatcher.hatchAngle += 10

    # Slice the boundary
    geomSlice = solidPart.getVectorSlice(zid*layerThickness)

    # Hatch the boundary using myHatcher
    layer = myHatcher.hatch(geomSlice)

    # The layer height is set in integer increment of microns to ensure no rounding error during manufacturing
    layer.z = int(zid*layerThickness * 1000)
    layer.layerId = int(zid)

    return layer

def main():
    set_start_method("spawn")

    # Imports the part and sets the geometry to  an STL file (frameGuide.stl)
    solidPart = pyslm.Part('inversePyramid')
    solidPart.setGeometry('../models/inversePyramid.stl')

    solidPart.origin[0] = 5.0
    solidPart.origin[1] = 2.5
    solidPart.scaleFactor = 1.0
    solidPart.rotation = [0, 0.0, 45]
    solidPart.dropToPlatform()
    print(solidPart.boundingBox)

    # Set the layer thickness
    layerThickness = 0.04 # [mm]

    #Perform the hatching operations
    print('Hatching Started')

    layers = []

    p = Pool(processes=1)

    d = Manager().dict()
    d['part'] = solidPart
    d['layerThickness'] = layerThickness

    # Rather than give the z position, we give a z index to calculate the z from.
    numLayers = solidPart.boundingBox[5] / layerThickness
    z = np.arange(0, numLayers).tolist()

    # The layer id and manager shared dict are zipped into a list of tuple pairs
    processList = list(zip([d] * len(z), z))

    startTime = time.time()

    # uncomment to test the time processing in single process
    #for pc in processList:
    #   calculateLayer(pc)

    layers = p.map(calculateLayer, processList)

    print('Multiprocessing time {:.1f} s'.format(time.time()-startTime))
    p.close()

    print('Completed Hatching')

    # Plot the layer geometries using matplotlib
    # Note: the use of python slices to get the arrays
    pyslm.visualise.plotLayers(layers[0:-1:10])


if __name__ == '__main__':
    main()