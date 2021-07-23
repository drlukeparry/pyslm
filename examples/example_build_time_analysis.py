"""
A simple example showing how to use PySLM for calculating the build time estimate.
THis example takes advantage of the multi-processing module to run across more threads.
"""
import pyslm
import shapely
from pyslm import hatching as hatching
import numpy as np
import time

from multiprocessing import Manager
from multiprocessing.pool import Pool
from multiprocessing import set_start_method

"""
Constants
"""
layerThickness = 0.03  # [mm]
rotation = [60, 0.0, 45]
layerRecoatTime = 30 # [s]
contourLaserScanSpeed = 250 # [mm/s]
hatchLaserScanSpeed = 1000 # [mm/s]
eos_m280_alsi10mg_brate = 4.8*3600/1000 # [cm3/hr]
hatchDistance = 0.16
numCountourOffsets = 1

def calculateLayer(input):
    # Typically the hatch angle is globally rotated per layer by usually 66.7 degrees per layer
    d = input[0]
    zid= input[1]

    layerThickness = d['layerThickness']
    solidPart = d['part']

    # Slice the boundary
    geomSlice = solidPart.getVectorSlice(zid*layerThickness, returnCoordPaths=False)
    #print(geomSlice)
    if len(geomSlice) > 0:
        return geomSlice
    else:
        return [shapely.geometry.Polygon()]

def main():
    set_start_method("spawn")

    # Imports the part and sets the geometry to  an STL file (frameGuide.stl)
    solidPart = pyslm.Part('FrameGuide')
    solidPart.setGeometry('../models/frameGuide.stl')

    solidPart.origin[0] = 5.0
    solidPart.origin[1] = 2.5
    solidPart.scaleFactor = 1.0
    solidPart.rotation = rotation
    solidPart.dropToPlatform()
    print(solidPart.boundingBox)

    # Create the multi-threaded map function  using the Python multiprocessing library
    layers = []

    p = Pool(processes=8)

    d = Manager().dict()
    d['part'] = solidPart
    d['layerThickness'] = layerThickness

    # Rather than give the z position, we give a z index to calculate the z from.
    numLayers = int(solidPart.boundingBox[5] / layerThickness)
    z = np.arange(0, numLayers).tolist()

    # The layer id and manager shared dict are zipped into a list of tuple pairs
    processList = list(zip([d] * len(z), z))

    startTime = time.time()

    print('Beginning Slicing')
    # uncomment to test the time processing in single process
    #for pc in processList:
     #   calculateLayer(pc)

    layers = p.map(calculateLayer, processList)
    p.close()

    print('\t Multiprocessing time', time.time() - startTime)
    print('Slicing Finished')

    polys = []
    for layer in layers:
        for poly in layer:
            polys.append(poly)

    layers = polys

    """
    Calculate total layer statistics:
    """
    totalHeight = solidPart.boundingBox[5]
    totalVolume = solidPart.volume
    totalPerimeter = np.sum([layer.length for layer in layers]) * numCountourOffsets
    totalArea = np.sum([layer.area for layer in layers])
    print('\nStatistics:')
    print('\tDiscretised volume {:.2f} cm3'.format(totalArea * layerThickness / 1e3))
    print("\tNum Layers {:d} Height: {:.2f}".format(numLayers, totalHeight))
    print("\tVolume: {:.2f} cm3, Area: {:.2f} mm2, Contour Perimeter: {:.2f} mm".format(totalVolume/1000, totalArea,totalPerimeter))

    """
    Calculate the time estimates:
    This calculates the total scan time using the layer slice approach 
    """
    hatchTimeEstimate = totalArea / hatchDistance / hatchLaserScanSpeed
    boundaryTimeEstimate = totalPerimeter / contourLaserScanSpeed
    scanTime = hatchTimeEstimate + boundaryTimeEstimate
    recoaterTimeEstimate = numLayers * layerRecoatTime

    totalTime = hatchTimeEstimate + boundaryTimeEstimate + recoaterTimeEstimate
    print('\nLayer Approach:')
    print("\tScan Time: {:.2f} hr, Recoat Time: {:.2f} hr, Total time: {:.3f} hr".format(scanTime / 3600, recoaterTimeEstimate/3600, totalTime/3600))

    """ 
    Calculate using a simplified approach
    Projected Area:
    Calculates the projected vertical area of the part 
    """
    print('\nApproximate Build Time Estimate:')
    # Calculate the vertical face angles
    v0 = np.array([[0., 0., 1.0]])
    v1 = solidPart.geometry.face_normals

    sin_theta = np.sqrt((1-np.dot(v0, v1.T)**2))
    triAreas = solidPart.geometry.area_faces * sin_theta
    projectedArea = np.sum(triAreas)
    print('\tProjected surface area: {:.3f}'.format(projectedArea))
    print('\tSurface area: {:.3f}'.format(solidPart.surfaceArea))

    approxScanTime = solidPart.volume/(hatchDistance * hatchLaserScanSpeed * layerThickness) + solidPart.surfaceArea / (contourLaserScanSpeed*layerThickness)
    approxProjectedScanTime = solidPart.volume / (hatchDistance * hatchLaserScanSpeed * layerThickness) + projectedArea / (
                contourLaserScanSpeed * layerThickness)
    print('\tApprox scan time *surface) {:.2f} hr'.format(approxScanTime/3600))
    print('\tApprox scan time (using projected area):  {:.2f} hr'.format(approxProjectedScanTime/3600))


if __name__ == '__main__':
    main()