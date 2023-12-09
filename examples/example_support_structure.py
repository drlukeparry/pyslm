"""
Support generation script - Shows how to generate basic block supports using PySLM
"""

import numpy as np
import logging

from matplotlib import pyplot as plt
from pyslm.core import Part
import pyslm.support

import vispy
import trimesh
import trimesh.creation

"""
Uncomment the line below to provide debug messages for OpenGL - if issues arise.
"""
# vispy.set_log_level('debug')

logging.getLogger().setLevel(logging.INFO)

## CONSTANTS ####
OVERHANG_ANGLE = 55 # deg - Overhang angle

"""
Set the Geometry for the Example using a complicated topology optimised bracket geometry
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
                                             splitMesh=False, useConnectivity=True)

overhangMesh.visual.face_colors = [254.0, 0., 0., 254]

"""
Generate the geometry for the supports (Point and Edge Over Hangs)
"""
# First generate point and edge supports
pointOverhangs = pyslm.support.BaseSupportGenerator.findOverhangPoints(myPart)
overhangEdges  = pyslm.support.BaseSupportGenerator.findOverhangEdges(myPart)

"""
Generate block supports for the part.

The GridBlockSupportGenerator class is initialised and the parameters below are specified as a reasonable starting
defaults for the algorithm. The GridBlockSupport generator overrides the BlockSupportGenerator class and provides
additional methods for generating a grid-truss structure from the support volume.
"""
supportGenerator = pyslm.support.GridBlockSupportGenerator()
supportGenerator.rayProjectionResolution = 0.05 # [mm] - The resolution of the grid used for the ray projection
supportGenerator.innerSupportEdgeGap = 0.3      # [mm] - Inner support offset used between adjacent support distances
supportGenerator.outerSupportEdgeGap = 0.3      # [mm] - Outer support offset used for the boundaries of overhang regions
supportGenerator.simplifyPolygonFactor = 0.5    #      - Factor used for simplifying the overall support shape
supportGenerator.triangulationSpacing = 2.0     # [mm] - Used for triangulating the extruded polygon for the bloc
supportGenerator.minimumAreaThreshold = 0.1     # Minimum area threshold to not process support region'
supportGenerator.triangulationSpacing = 4       # [mm^2] - Internal parameter used for generating the mesh of the volume
supportGenerator.supportBorderDistance = 1.0    # [mm]

# Support teeth parameters
supportGenerator.useUpperSupportTeeth = True
supportGenerator.useLowerSupportTeeth = True
supportGenerator.supportTeethTopLength = 0.1        # [mm] - The length of the tab for the support teeth
supportGenerator.supportTeethHeight = 1.5           # [mm] - Length of the support teeth
supportGenerator.supportTeethBaseInterval = 1.5     # [mm] - The interval between the support teeth
supportGenerator.supportTeethUpperPenetration = 0.2 # [mm] - The penetration of the support teeth into the part

supportGenerator.splineSimplificationFactor = 10 # - Specify the smoothing factor using spline interpolation for the support boundaries
supportGenerator.gridSpacing = [5,5] # [mm] The Grid

"""
Generate a list of Grid Block Supports (trimesh objects currently). The contain the support volumes and other generated
information identified from the support surfaces identified on the part based on the choice of overhang angle.
"""
supportBlockRegions = supportGenerator.identifySupportRegions(myPart, OVERHANG_ANGLE, True)

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
    """ Visualise Edges potentially requiring support"""
    edgeRays = np.vstack([meshVerts[edge] for edge in overhangEdges])
    visualize_support_edges = trimesh.load_path((edgeRays).reshape(-1, 2, 3))

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

# Make the normal part transparent
myPart.geometry.visual.vertex_colors = [80,80,80, 125]

"""
Visualise all the support geometry
"""

""" Identify the sides of the block extrudes """
s1 = trimesh.Scene([myPart.geometry] + blockSupports)

"""
The following section exports the group of support structures from the trimesh scene. 
"""

with open('overhangSupport.glb', 'wb') as f:
    f.write(trimesh.exchange.gltf.export_glb(s1, include_normals=True))

"""
Show only the volume block supports generated
"""
DISPLAY_BLOCK_VOLUME = False

if DISPLAY_BLOCK_VOLUME:
    s2 = trimesh.Scene([myPart.geometry, overhangMesh,
                        blockSupports])
    s2.show()

"""
The following section generates the grid-truss structure by calling the geometry method. As a summary, the process 
takes a multiple cross-sections across the support volume and extracts the faces of the volume boundary projected onto
an equivalent 2D area. Within the 2D regions a series of lines are offset and intersected to produce a grid structure.
The polygon is converted to a triangular mesh and the boundary truss is mapped back onto the original 3D boundary.
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

