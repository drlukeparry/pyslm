"""
A simple example showing how to use PySLM for generating a sinusoidal scanning strategy across a single layer.
"""

import numpy as np
import pyslm

import pyslm.visualise
import pyslm.analysis
from pyslm import hatching as hatching
from pyslm.geometry import Layer, LayerGeometry, ContourGeometry, HatchGeometry
from pyslm.hatching import BaseHatcher
from typing import Optional

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

class WavyHatcher(pyslm.hatching.Hatcher):

    def __init__(self):
        super().__init__()

        self._amplitude = 1
        self._frequency = 0.1
        self._discretisation = 1.5

    @property
    def amplitude(self) -> float:
        return self._amplitude

    @amplitude.setter
    def amplitude(self, value):
        self._amplitude = value

    @property
    def frequency(self):
        return self._frequency

    @frequency.setter
    def frequency(self, value):
        self._frequency = value

    @property
    def discretisation(self):
        return self._discretisation

    @discretisation.setter
    def discretisation(self, value):
        self._discretisation = value

    def __str__(self):
        return 'StripeHatcher'

    def hatch(self, boundaryFeature):
        """
        Generates a series of contour or boundary offsets along with a basic full region internal hatch.
        """

        if len(boundaryFeature) == 0:
            return

        layer = Layer(0, 0)
        # First generate a boundary with the spot compensation applied

        offsetDelta = 0.0
        offsetDelta -= self._spotCompensation

        for i in range(self._numOuterContours):
            offsetDelta -= self._contourOffset
            offsetBoundary = self.offsetBoundary(boundaryFeature, offsetDelta)

            for poly in offsetBoundary:
                for path in poly:
                    contourGeometry = ContourGeometry()
                    contourGeometry.coords = np.array(path)[:, :2]
                    contourGeometry.subType = "outer"
                    layer.geometry.append(contourGeometry)  # Append to the layer

        # Repeat for inner contours
        for i in range(self._numInnerContours):

            offsetDelta -= self._contourOffset
            offsetBoundary = self.offsetBoundary(boundaryFeature, offsetDelta)

            for poly in offsetBoundary:
                for path in poly:
                    contourGeometry = ContourGeometry()
                    contourGeometry.coords = np.array(path)[:, :2]
                    contourGeometry.subType = "inner"
                    layer.geometry.append(contourGeometry)  # Append to the layer

        # The final offset is applied to the boundary

        offsetDelta -= self._volOffsetHatch

        curBoundary = self.offsetBoundary(boundaryFeature, offsetDelta)

        scanVectors = []

        if True:
            paths = curBoundary

            # Hatch angle will change per layer
            # TODO change the layer angle increment
            layerHatchAngle = np.mod(self._hatchAngle + self._layerAngleIncrement, 180)

            # The layer hatch angle needs to be bound by +ve X vector (i.e. -90 < theta_h < 90 )
            if layerHatchAngle > 90:
                layerHatchAngle = layerHatchAngle - 180

            # Generate the un-clipped hatch regions based on the layer hatchAngle and hatch distance
            hatches = self.generateHatching(paths, self._hatchDistance, layerHatchAngle)

            """
            A significant difference in previous implementations is that the paths are clipped separatly as open paths.
            These are provided as a list of separated paths rather than the normal procedure which clips single
            discrete hatch lines. This ensures paths are separated.
            """

            clippedPaths = self.clipContourLines(paths, hatches)
            geomIds = [coord[0][2] for coord in clippedPaths]

            """
            Sort the sinusoidal vectors based on the 1st coordinate's sort id (column 3). This only sorts individual paths
            rather than the contours internally.            
            """
            clippedPaths = sorted(clippedPaths, key=lambda x: x[0][2])

            # Merge the lines together
            if len(clippedPaths) > 0:
                for path in clippedPaths:
                    clippedLines = np.vstack(path) #BaseHatcher.clipperToHatchArray(clippedPaths)

                    clippedLines = clippedLines[:,:2]
                    # Uncomment to turn to use hatch geometry
                    #clippedLines = np.concatenate([clippedLines[:-1, :2], clippedLines[1:,:2]], axis=1)
                    #contourGeom = HatchGeometry()

                    contourGeom = ContourGeometry()
                    contourGeom.coords = clippedLines.reshape(-1, 2)

                    layer.geometry.append(contourGeom)



        else:
            # Iterate through each closed polygon region in the slice. The currently individually sliced.
            for contour in curBoundary:
                # print('{:=^60} \n'.format(' Generating hatches '))

                paths = contour

                # Hatch angle will change per layer
                # TODO change the layer angle increment
                layerHatchAngle = np.mod(self._hatchAngle + self._layerAngleIncrement, 180)

                # The layer hatch angle needs to be bound by +ve X vector (i.e. -90 < theta_h < 90 )
                if layerHatchAngle > 90:
                    layerHatchAngle = layerHatchAngle - 180

                # Generate the un-clipped hatch regions based on the layer hatchAngle and hatch distance
                hatches = self.generateHatching(paths, self._hatchDistance, layerHatchAngle)

                # Clip the hatch fill to the boundary
                clippedPaths = self.clipLines(paths, hatches)

                # Merge the lines together
                if len(clippedPaths) == 0:
                    continue

                clippedLines = self.clipperToHatchArray(clippedPaths)

                # Extract only x-y coordinates and sort based on the pseudo-order stored in the z component.
                clippedLines = clippedLines[:, :, :3]
                id = np.argsort(clippedLines[:, 0, 2])
                clippedLines = clippedLines[id, :, :]

                scanVectors.append(clippedLines)


        return layer

    def generateHatching(self, paths, hatchSpacing: float, hatchAngle: Optional[float] = 90.0) -> np.ndarray:
        """
        Generates un-clipped sinusoidal hatches which is guaranteed to cover the entire polygon region base on the
        maximum extent  of the polygon bounding box
        """

        # Hatch angle
        theta_h = np.radians(hatchAngle)  # 'rad'

        # Get the bounding box of the paths
        bbox = self.boundaryBoundingBox(paths)

        # print('bounding box bbox', bbox)
        # Expand the bounding box
        bboxCentre = np.mean(bbox.reshape(2, 2), axis=0)

        # Calculates the diagonal length for which is the longest
        diagonal = bbox[2:] - bboxCentre
        bboxRadius = np.sqrt(diagonal.dot(diagonal))

        # Construct a square which wraps the radius

        #y = np.array([-bboxRadius, bboxRadius])

        dx = self._discretisation # num points per mm
        numPoints = 2*bboxRadius * dx

        x = np.arange(-bboxRadius, bboxRadius, hatchSpacing, dtype=np.float32).reshape(-1, 1)
        hatches = x.copy()

        """
        Generate the sinusoidal curve along the local coordinate system x' and y'. These will be later tiled and then
        transformed across the entire coordinate space.
        """
        xDash = np.linspace(-bboxRadius, bboxRadius, int(numPoints))
        yDash = self._amplitude * np.sin(2.0*np.pi * self._frequency * xDash)

        """
        We replicate and transform the sine curve along adjacent paths and transform along the y-direction
        """
        y = np.tile(yDash, [x.shape[0], 1])
        y += x

        x = np.tile(xDash, [x.shape[0],1]).flatten()
        y = y.ravel()

        z = np.arange(0, x.shape[0] ).astype(np.int64)

        # Seperate the z-order index per group
        inc = np.arange(0, 10000*(xDash.shape[0]), 10000).astype(np.int64).reshape(-1,1)
        zInc = np.tile(inc, [1,hatches.shape[0]]).flatten()
        z += zInc

        coords = np.hstack([x.reshape(-1, 1),
                            y.reshape(-1, 1),
                            z.reshape(-1, 1)])

        # Create the 2D rotation matrix with an additional row, column to preserve the hatch order
        c, s = np.cos(theta_h), np.sin(theta_h)
        R = np.array([(c, -s, 0),
                      (s, c, 0),
                      (0, 0, 1.0)])

        # Apply the rotation matrix and translate to bounding box centre
        coords = np.matmul(R, coords.T)
        coords = coords.T + np.hstack([bboxCentre, 0.0])

        """        
        The transformed coordinate group needs to be split into seperate open paths since they will be clipped as 
        discrete paths using PyClipper
        """

        print('Hatch Pattern Generated')
        return np.split(coords, hatches.shape[0])


# Set te slice layer position
z = 1.0

# Create a BasicIslandHatcher object for performing any hatching operations (
myHatcher = WavyHatcher()
myHatcher.islandWidth = 3.0
myHatcher.stripeWidth = 5.0
myHatcher.hatchDistance = 1.0
myHatcher.amplitude = 1 # The amplitude of the sine curve
myHatcher.frequency = 2 # The frequency / periodicity of the sine curve
myHatcher.discretisation = 20 # Number of points per unit distance for the sinusoidal curve

# Set the base hatching parameters which are generated within Hatcher
myHatcher.hatchAngle = 120 # [°] The angle used for the islands
myHatcher.volumeOffsetHatch = 0.06 # [mm] Offset between internal and external boundary
myHatcher.spotCompensation = 0.06 # [mm] Additional offset to account for laser spot size
myHatcher.numInnerContours = 2
myHatcher.numOuterContours = 1


"""
Perform the slicing. Return coords paths should be set so they are formatted internally.
This is internally performed using Trimesh to obtain a closed set of polygons.
The boundaries of the slice can be automatically simplified if desired. 
"""
geomSlice = solidPart.getVectorSlice(z, simplificationFactor=0.1)
layer = myHatcher.hatch(geomSlice)


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
bstyle.laserSpeed = 200 # [mm/s]
bstyle.laserPower = 200 # [W]

model = pyslm.geometry.Model()
model.mid = 1
model.buildStyles.append(bstyle)

"""
Analyse the layers using the analysis module. The path distance and the estimate time taken to scan the layer can be
predicted.
"""
print('Total Path Distance: {:.1f} mm'.format(pyslm.analysis.getLayerPathLength(layer)))
print('Total jump distance {:.1f} mm'.format(pyslm.analysis.getLayerJumpLength(layer)))
print('Time taken {:.1f} s'.format(pyslm.analysis.getLayerTime(layer, [model])) )

