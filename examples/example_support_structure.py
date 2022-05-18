"""
Support generation script - Shows how to generate basic block supports using PySLM
Support Generation currently requires compiling the `Cork library <https://github.com/gilbo/cork> and then providing
the path to the compiled executable`
"""




import numpy as np
from vispy import app

app.use_app('pyqt5') # Set backend

from matplotlib import pyplot as plt
from pyslm.core import Part
import pyslm.support

import trimesh
import trimesh.creation
import logging

logging.getLogger().setLevel(logging.INFO)

## CONSTANTS ####
CORK_PATH = '/home/lparry/Development/src/external/cork/bin/cork'

pyslm.support.BlockSupportGenerator.CORK_PATH = CORK_PATH

OVERHANG_ANGLE = 55 # deg - Overhang angle


img = np.zeros([1000,1000])
x,y = np.meshgrid(np.arange(0,img.shape[1]), np.arange(0, img.shape[0]))

solid = np.sqrt((x-500)**2 + (y-500)**2) < 400
bound =   np.sqrt((x-500)**2 + (y-500)**2)
#solid = img[:,:,3].astype(np.float64)
orient = np.array([1.0,1.0])
orient = orient / np.linalg.norm(orient)
perp = np.array((orient[1], orient[0]))


dotProd = np.dot(x, orient[0])+np.dot(y,orient[1])
solid2 = solid*( np.sin(0.2*dotProd))
bound * solid2


"""
Set the Geometry for the Example
"""
myPart = Part('myPart')
myPart.setGeometry("../models/bracket.stl", fixGeometry=True)
#myPart.scaleFactor = 4.0
myPart.rotation = [62.0, 50.0, -0.0] #[62.0, 50.0, -40.0] #[10, 70, 30] #[62.0, 50.0, -40.0]  #[-70.0, 50.0, -30.0] #[62.0, 50.0, -40.0]
#myPart.rotation = [76,35,-13] #[-25,0,5] #[76,35,-13]#[62,50,-40.0]

myPart.scaleFactor = 1.0
myPart.dropToPlatform(10)

""" Extract the overhang mesh - don't explicitly split the mesh"""
overhangMesh = pyslm.support.getOverhangMesh(myPart, OVERHANG_ANGLE,
                                             splitMesh=False, useConnectivity=False)
overhangMesh.visual.face_colors = [254.0, 0., 0., 254]


"""
Generate the geometry for the supports (Point and Edge Over Hangs)
"""
# First generate point and edge supports
pointOverhangs = pyslm.support.BaseSupportGenerator.findOverhangPoints(myPart)
overhangEdges = pyslm.support.BaseSupportGenerator.findOverhangEdges(myPart)

"""
Generate block supports for the part.
The GridBlockSupportGenerator class is initialised and the parameters below are specified
"""
supportGenerator = pyslm.support.GridBlockSupportGenerator()
supportGenerator.rayProjectionResolution = 0.1 # [mm] - The resolution of the grid used for the ray projection
supportGenerator.innerSupportEdgeGap = 0.2 # [mm] - Inner support offset used between adjacent support distances
supportGenerator.outerSupportEdgeGap = 0.2 # [mm] - Outer support offset used for the boundaries of overhang regions
supportGenerator.simplifyPolygonFactor = 0.5 #  - Factor used for simplifying the overall support shape
supportGenerator.triangulationSpacing = 2.0 # [mm] - Used for triangulating the extruded polygon for the bloc
supportGenerator.minimumAreaThreshold = 0.1 # Minimum area threshold to not process support region'
supportGenerator.triangulationSpacing = 4
supportGenerator.supportBorderDistance = 1.0
supportGenerator.splineSimplificationFactor = 10
#supportGenerator.gridSpacing = [20,20]


# Generate a list of  Grid Block Supports (trimesh objects currently)
supportBlockRegions = supportGenerator.identifySupportRegions(myPart, OVERHANG_ANGLE)


for block in supportBlockRegions:
    block.trussWidth = 1.0

blockSupports = [block.supportVolume for block in supportBlockRegions]

"""
Generate the edges for visualisation
"""
edges = myPart.geometry.edges_unique
meshVerts = myPart.geometry.vertices
centroids = myPart.geometry.triangles_center

if True:
    """ Visualise Edge Supports"""
    edgeRays = np.vstack([meshVerts[edge] for edge in overhangEdges])
    visualize_support_edges = trimesh.load_path((edgeRays).reshape(-1, 2, 3))
    colorCpy = visualize_support_edges.colors.copy()
    colorCpy[:] = [254, 0, 0, 254]
    visualize_support_edges.colors = colorCpy

    edge_supports = []
    for edge in overhangEdges:
        coords = np.vstack([meshVerts[edge,:]]*2)
        coords[2:,2] = 0.0

        extrudeFace = np.array([(0, 1, 3), (3, 2, 0)])
        edge_supports.append(trimesh.Trimesh(vertices=coords, faces=extrudeFace))

    """  Visualise Point Supports """

    point_supports = []
    cylinder_rad = 0.5 # mm
    rays = []
    for pnt in pointOverhangs:
        coords = np.zeros((2,3))
        coords[:,:] = meshVerts[pnt]
        coords[1,2] = 0.0

        point_supports += trimesh.creation.cylinder(radius = cylinder_rad, segment =coords)
        rays.append(coords)


    # Alternatively can be visualised by lines
    rays = np.hstack([meshVerts[pointOverhangs]]*2).reshape(-1, 2, 3)
    rays[:, 1, 2] = 0.0
    visualize_support_pnts = trimesh.load_path(rays)


import trimesh.creation


# Make the normal part transparent
myPart.geometry.visual.vertex_colors = [80,80,80, 125]

"""
Visualise all the support geometry
"""
""" Identify the sides of the block extrudes """
s1 = trimesh.Scene([myPart.geometry, overhangMesh] + blockSupports) # , overhangMesh] + supportExtrudes)

with open('overhangSupport.glb', 'wb') as f:
    f.write(trimesh.exchange.gltf.export_glb(s1, include_normals=True))

DISPLAY_BLOCK_VOLUME = True

if DISPLAY_BLOCK_VOLUME:
    s2 = trimesh.Scene([myPart.geometry, overhangMesh,
                        point_supports, edge_supports,
                        blockSupports])
    s2.show()

#
"""
Merge the support geometry together into a single mesh
"""

meshSupports = []

for supportBlock in supportBlockRegions:
    supportBlock.mergeMesh = False
    meshSupports.append(supportBlock.geometry())

s2 = trimesh.Scene([overhangMesh, myPart.geometry,
                   ] + meshSupports)

s2.show()

isectMesh += blockSupportSides

isectMesh = blockSupportMesh + myPart.geometry

# Obtain the 2D Planar Section at this Z-position
sections = isectMesh.section(plane_origin=[0.0, 0, 10.0], plane_normal=[0, 0, 1])
blockSupportMesh
transformMat = np.array(([1.0, 0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.0, 1.0]), dtype=np.float32)

planarSection, transform = sections.to_planar(transformMat, normal=[1,0,0])
sections.show()

