"""
A simple example showing how to generate slices and scan for implicit fields, which are typically used for generating
lattice structures.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import measure

import pyslm
import pyslm.visualise
import pyslm.analysis
from pyslm import hatching as hatching

""" Specify the target resolution  and the size of the lattice to generate """
res = 0.1

Lx = 40
Ly = 40
Lz = 10

""" Number of lattice unit cells"""
kx = 8
ky = 8
kz = 8

""" Create the computational grid - note np operates with k(z) numerical indexing unlike the default matlab equivalent"""
z, y, x = np.meshgrid(np.arange(0, Lz, res),
                      np.arange(0, Ly, res),
                      np.arange(0, Lx, res), indexing='ij')

nz, ny, nx = x.shape

""" 
Calculating the Gyroid TPMS
"""
T = 0.7

U = ( np.cos(kx*2*np.pi*(x/Lx))*np.sin(ky*2*np.pi*(y/Ly))
    + np.cos(ky*2*np.pi*(y/Ly))*np.sin(kz*2*np.pi*(z/Lz))
    + np.cos(kz*2*np.pi*(z/Lz))*np.sin(kx*2*np.pi*(x/Lx)) )**2 - T**2

""" Generate a sphere to fill the domain"""
sphere_rad = Lx/2
sphere = np.sqrt((x-Lx/2)**2 + (y-Ly/2)**2  + (z-Lz/2)**2) < sphere_rad

# Generate a narrow-band levelset
#sphere = sphere * np.logical_and(sphere > sphere_rad -3, sphere < sphere_rad +3)
#sphere = (sphere - sphere_rad) / 3

""" Note plotting the image"""
plt.figure()
plt.imshow(sphere[int(nz/2)])

""" Merge the space to create final implicit field representing a gyroid sphere and plot the field. The background
space is set to >0 to isolate the lattice sphere. """
field = U
field[sphere != 1] = 1

""" 
Having obtained the field, the slicing operations can begin. THe boundaries across the slice are directly extracted from
the implicit field. The boundary extracted is simply the marching squares algorithm applied to a 2D slice extracted from
the 3D numpy array.
"""


"""
Note we are simply extracting a 2D XY slice from the array, which is dependent on the resolution chosen. This can be 
improved further by extracting neighbouring slices and interpolation. The transpose is required, due to x,y coordinates
being swapped.
"""
z_slice_pos = int(3.04 / res)
slice = field[z_slice_pos]

plt.figure()
plt.imshow(slice.T)

contours = measure.find_contours(slice, -0.00001)

""" Simplify the boundaries.  Internally this is simply using skimage.measure.approximate_polygon"""
contours = hatching.simplifyBoundaries(contours, 0.2)

""" Scale the coordinates back to the original coordinate systems"""
contours = [contour * res for contour in contours]

""" We have to scale the coordinates back to the original resolution"""

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
The boundaries of the slice can be automatically simplified if desired. 
"""
print('Hatching started')
layer = myHatcher.hatch(contours)
print('Hatching finished')

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

pyslm.visualise.plot(layer, plot3D=False, plotOrderLine=False, plotArrows=False)

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
bstyle.laserSpeed = 200.0 # [mm/s]
bstyle.laserPower = 200.0 # [W]

model = pyslm.geometry.Model()
model.mid = 1
model.buildStyles.append(bstyle)

"""
Analyse the layers using the analysis module. The path distance and the estimate time taken to scan the layer can be
predicted.
"""
print('Total Path Distance: {:.1f} mm'.format(pyslm.analysis.getLayerPathLength(layer)))
print('Total jump distance {:.1f} mm'.format(pyslm.analysis.getLayerJumpLength(layer)))
print('Time taken {:.1f} s'.format(pyslm.analysis.getLayerTime(layer, [model])))

