"""
A simple reference example showing how to use Trimesh functions to convert vector slices obtained from a part into a
bitmap (binary) image of the current slice at very high resolutions.

This internally uses Trimesh's capability, essentially built on the Pillow library, so much credit goes to there, however,
given it's utility generally in AM, it is a valuable function to show how this can be very quickly and conveniently
generated.

This method is very unlikely to be as fast as a dedicated voxeliser method to generate 3D volumes, but is reasonable
technique to generate a stack of bitmap images for DLP, BJF, Inkjet processes.
"""

import numpy as np
import pyslm

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
solidPart.scaleFactor = 2.0
solidPart.dropToPlatform()

# Note the resolution units are [mm/px], DPI = [px/inch]
dpi = 300.0
resolution = 25.4 / dpi

# Return the Path2D object from Trimesh by setting second argument to False
slice = solidPart.getTrimeshSlice(14.0)

# Rasterise and cast to a numpy array
# The origin is set based on the minium XY bounding box of the part. Depending on the platform the user may
sliceImage = slice.rasterize(pitch = resolution, origin= solidPart.boundingBox[:2])
sliceImage = np.array(sliceImage)

# For convenience, the same function above is available directly from the Part class
slice = solidPart.getBitmapSlice(14.0, resolution)

fig = plt.figure()
ax = plt.imshow(sliceImage,  cmap='gray', origin='lower')
