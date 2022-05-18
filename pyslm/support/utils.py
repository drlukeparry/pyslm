import logging
from typing import Optional, Tuple, List

import networkx as nx

import numpy as np
import trimesh

from . import geometry
from ..core import Part


def getAdjacentFaces(mesh: trimesh.Trimesh):
    """
    Returns a list of connected faces
    :param mesh: A Trimesh mesh for locating the connected faces

    :return: Returns a list of connectivity for each face
    """
    graph = nx.Graph()
    graph.add_edges_from(mesh.face_adjacency)

    adjacentFaces = {node: list(graph.neighbors(node)) for node in graph.nodes}
    return adjacentFaces


def getSupportAngles(part: Part, unitNormal: np.ndarray = None, useConnectivity: Optional[bool] = True) -> np.ndarray:
    """
    Returns the support angles for each triangular face normal. This is mainly used for the benefit of visualising the
    support angles for a part.

    :param part: The :class:`Part` to calculate the support or overhang angles
    :param unitNormal: The up-vector direction used to calculate the angle against
    :param useConnectivity: Use face connectivity to get the interpolated support angle
    :return: The support angles across the whole part geometry
    """

    # Upward vector for support angles
    v0 = np.array([[0., 0., -1.0]]) if unitNormal is None else np.asanyarray(unitNormal)

    # Identify Support Angles
    v1 = part.geometry.face_normals
    theta = np.arccos(np.clip(np.dot(v0, v1.T), -1.0, 1.0))
    theta = np.degrees(theta).flatten()

    if useConnectivity:
        thetaAvg = theta.copy()
        adjacencyList = getAdjacentFaces(part.geometry)
        for face in adjacencyList.keys():
            conFaces = [face] + adjacencyList[face]
            thetaAvg[face] = np.mean(theta[conFaces])

        return thetaAvg
    else:
        return theta


def getFaceZProjectionWeight(mesh: trimesh.Trimesh,
                             useConnectivity: Optional[bool] = False) -> np.ndarray:
    """
    Utility which returns the inverse projection of the faces relative to the +ve Z direction in order to isolate side
    faces. This could be considered the inverse component of the overhang angle. It is calculated by using the
    following trigonometric identify :math:`\sin(\\theta) = \sqrt{1-\cos^2(\\theta)`.

    :param mesh: The mesh to identify the projection weights
    :param useConnectivity: Uses mesh connectivity to interpolate the surface normals across
    """

    v0 = np.array([[0., 0., 1.0]])
    v1 = mesh.face_normals

    sin_theta = np.sqrt((1 - np.dot(v0, v1.T) ** 2)).reshape(-1)

    if useConnectivity:
        sinThetaAvg = sin_theta.copy()
        adjacencyList = getAdjacentFaces(mesh)
        for face in adjacencyList.keys():
            conFaces = [face] + adjacencyList[face]
            sinThetaAvg[face] = np.mean(sinThetaAvg[conFaces])
        return sinThetaAvg
    else:
        return sin_theta


def getOverhangMesh(part: Part, overhangAngle: float,
                    splitMesh: Optional[bool] = False,
                    useConnectivity: Optional[bool] = False) -> trimesh.Trimesh:
    """
    Gets the overhang mesh from a :class:`Part`. If the individual regions for the overhang mesh require separating,
    the parameter :code:`splitMesh` should be set to `True`. This will split mesh regions by their facial connectivity
    using Trimesh.

    :param part: The part to extract the overhang mesh from
    :param overhangAngle: The overhang angle in degrees
    :param splitMesh: If the overhang mesh should be split into separate Trimesh entities by network connectivity
    :param useConnectivity: Uses mesh connectivity to interpolate the surface normals across
    :return: The extracted overhang mesh
    """

    # Upward vector for support angles
    v0 = np.array([[0., 0., 1.0]])

    theta = getSupportAngles(part, unitNormal=v0, useConnectivity=useConnectivity)

    supportFaceIds = np.argwhere(theta > 180 - overhangAngle).flatten()

    overhangMesh = trimesh.Trimesh(vertices=part.geometry.vertices,
                                   faces=part.geometry.faces[supportFaceIds])

    if splitMesh:
        return overhangMesh.split(only_watertight=False)
    else:
        return overhangMesh


def approximateSupportMomentArea(part: Part, overhangAngle: float) -> float:
    """
    The support moment area is a metric, which projects the distance from the base-plate (:math:`z=0`) for
    each support surface multiplied by the area. It gives a two parameter component cost function for the support area.

    .. note::
        This is an approximation that does not account for any self-intersections. It does not use ray queries to
        project the distance towards the mesh, therefore is more useful estimating the overall cost of the support
        structures, during initial support optimisation.

    :param part: The part to analyse
    :param overhangAngle: The overhang angle in degrees

    :return: The approximate cost function
    """
    overhangMesh = getOverhangMesh(part, overhangAngle)

    zHeights = overhangMesh.triangles_center[:, 2]

    # Use the projected area by flattening the support faces
    overhangMesh.vertices[:, 2] = 0.0
    faceAreas = overhangMesh.area_faces

    return float(np.sum(faceAreas * zHeights))


def getApproximateSupportArea(part: Part, overhangAngle: float, projected: Optional[bool] = False) -> float:
    """
    The support area is a metric of the total area of support surfaces, including the flattened or projected area.

    .. note::
        This is an approximation that does not account for any self-intersections. It does not use ray queries to
        project the distance towards the mesh, therefore is more useful estimating the overall cost of the support
        structures, during initial support optimisation.

    :param part: The part to analyse
    :param overhangAngle: The overhang angle in degrees
    :param projected: If True, the projected area is used
    :return: The approximate cost function
    """
    overhangMesh = getOverhangMesh(part, overhangAngle)

    zHeights = overhangMesh.triangles_center[:,2]

    # Use the projected area by flattening the support faces
    if projected:
        overhangMesh.vertices[:, 2] = 0.0

    faceAreas = overhangMesh.area_faces

    return faceAreas


def approximateSupportMapByCentroid(part: Part, overhangAngle: float,
                                    includeTriangleVertices: Optional[bool] = False) -> Tuple[np.ndarray]:
    """
    This method to approximate the surface area, projects  a single ray :math:`(0,0,-1)`, form each triangle in the
    overhang mesh -originating from the centroid or optionally each triangle vertex by setting the
    :code:`includeTriangleVertices` parameter. A self-intersection test with the mesh is performed  and this is used to
    calculate the distance from the hit location or if no intersection is made the base-plate (:math:`z=0.0`),
    which may be used later to generate a support heightmap.

    :param part: The :class:`Part` to analyse
    :param overhangAngle: The desired overhang angle in degrees
    :param includeTriangleVertices: Optional parameter projects also from the triangular vertices
    :return: A tuple with the support map
    """

    overhangMesh = getOverhangMesh(part, overhangAngle)

    coords = overhangMesh.triangles_center

    if includeTriangleVertices:
        coords = np.vstack([coords, overhangMesh.vertices])

    ray_dir = np.tile(np.array([[0., 0., -1.0]]), (coords.shape[0], 1))

    # Find the first intersection hit of rays project from the triangle.
    hitLoc, index_ray, index_tri = part.geometry.ray.intersects_location(ray_origins=coords,
                                                                         ray_directions=ray_dir,
                                                                         multiple_hits=False)

    heightMap = np.zeros((coords.shape[0], 1), dtype=np.float)
    heightMap[index_ray] = hitLoc[:, 2].reshape(-1, 1)
    
    heightMap = np.abs(heightMap - coords[:, 2])

    return heightMap


def approximateProjectionSupportCost(part: Part, overhangAngle: float,
                                     includeTriangleVertices: Optional[bool] = False) -> float:
    """
    Provides a support structure cost using ray projection from the overhang regions which allows for self-intersection
    checks.

    :param part: The part to determine the overall approximate support cost
    :param overhangAngle: The overhang angle in degree
    :param includeTriangleVertices: Optional parameter projects also from the triangular vertices
    :return: The cost function for support generation
    """

    overhangMesh = getOverhangMesh(part, overhangAngle)

    heightMap = approximateSupportMapByCentroid(part, overhangAngle, includeTriangleVertices)

    # Project the overhang area
    overhangMesh.vertices[:, 2] = 0.0
    faceAreas = overhangMesh.area_faces

    return np.sum(faceAreas * heightMap), heightMap


def generateHeightMap(mesh: trimesh.Trimesh,
                       upVec = [0,0,1.0],
                       resolution: Optional[float] = 0.5,
                       offsetPoly: Optional[trimesh.path.Path2D] = None) -> Tuple[np.ndarray]:
    """
    Generates the height map of the upper and lower depths. This is done by projecting rays at a resolution
    (attr:`~BlockSupportGenerator.rayProjectionResolution`) across the entire polygon region (offsetPoly) in both
    vertical directions (+z, -z) and are intersected with the upper and lower support surface. A sequence of
    height maps are generated from these ray intersections.

    :param mesh: The upper surface (typically overhang surface region)
    :param upVec: The projection vector (default is pointing vertically in the +Z direction)
    :param resolution: The resolution of the height map to generate
    :param offsetPoly: The polygon region defining the support region
    :return: A tuple containing various height maps
    """

    if not offsetPoly:
        # Generate a polgyon covering the part's bouding box
        offsetPoly = trimesh.load_path(geometry.generatePolygonBoundingBox(mesh.bounds.reshape(2,3)))


    # Rasterise the surface of overhang to generate projection points
    supportArea = np.array(offsetPoly.rasterize(resolution, offsetPoly.bounds[0, :])).T

    coords = np.argwhere(supportArea).astype(np.float32) * resolution
    coords += offsetPoly.bounds[0, :] + 1e-5  # An offset is required due to rounding error

    logging.info('\t - start projecting rays')
    logging.info('\t - number of rays with resolution ({:.3f}): {:d}'.format(resolution, len(coords)))

    """
    Project upwards to intersect with the upper surface
    """
    # Set the z-coordinates for the ray origin
    coords = np.insert(coords, 2, values=-1e5, axis=1)
    rays = np.repeat([upVec], coords.shape[0], axis=0)

    # Find the first location of any triangles which intersect with the part
    hitLoc, index_ray, index_tri = mesh.ray.intersects_location(ray_origins=coords,
                                                                ray_directions=rays,
                                                                multiple_hits=False)
    logging.info('\t - finished projecting rays')

    # Create a height map of the projection rays
    heightMap = np.ones(supportArea.shape) * -1.0

    if len(hitLoc) > 0:
        hitLocCpy = hitLoc.copy()
        hitLocCpy[:, :2] -= offsetPoly.bounds[0, :]
        hitLocCpy[:, :2] /= resolution

        hitLocIdx = np.ceil(hitLocCpy[:, :2]).astype(np.int32)

        # Assign the heights
        heightMap[hitLocIdx[:, 0], hitLocIdx[:, 1]] = hitLoc[:, 2]

    return heightMap


def generateHeightMap2(mesh: trimesh.Trimesh,
                       upVec = [0,0,1.0],
                       resolution: Optional[float] = 0.5,
                       offsetPoly: Optional[trimesh.path.Path2D] = None) -> Tuple[np.ndarray]:
    """
    Generates the height map of the upper and lower depths. This is done by projecting rays at a resolution
    (attr:`~BlockSupportGenerator.rayProjectionResolution`) across the entire polygon region (offsetPoly) in both
    vertical directions (+z, -z) and are intersected with the upper and lower support surface. A sequence of
    height maps are generated from these ray intersections.

    :param mesh: The upper surface (typically overhang surface region)
    :param upVec: The projection vector (default is pointing vertically in the +Z direction)
    :param resolution: The resolution of the height map to generate
    :param offsetPoly: The polygon region defining the support region
    :return: A tuple containing various height maps
    """

    if not offsetPoly:
        # Generate a polgyon covering the part's bouding box
        offsetPoly = trimesh.load_path(geometry.generatePolygonBoundingBox(mesh.bounds.reshape(2,3)))


    # Rasterise the surface of overhang to generate projection points
    supportArea = np.array(offsetPoly.rasterize(resolution, offsetPoly.bounds[0, :])).T

    coords = np.argwhere(supportArea).astype(np.float32) * resolution
    coords += offsetPoly.bounds[0, :] + 1e-5  # An offset is required due to rounding error

    logging.info('\t - start projecting rays')
    logging.info('\t - number of rays with resolution ({:.3f}): {:d}'.format(resolution, len(coords)))

    """
    Project upwards to intersect with the upper surface
    """
    # Set the z-coordinates for the ray origin
    coords = np.insert(coords, 2, values=-1e5, axis=1)
    rays = np.repeat([upVec], coords.shape[0], axis=0)

    # Find the first location of any triangles which intersect with the part
    hitLoc, index_ray, index_tri = mesh.ray.intersects_location(ray_origins=coords,
                                                                ray_directions=rays,
                                                                multiple_hits=False)
    logging.info('\t - finished projecting rays')

    # Create a height map of the projection rays
    heightMap = np.ones(supportArea.shape) * -1.0

    if len(hitLoc) > 0:
        hitLocCpy = hitLoc.copy()
        hitLocCpy[:, :2] -= offsetPoly.bounds[0, :]
        hitLocCpy[:, :2] /= resolution

        hitLocIdx = np.ceil(hitLocCpy[:, :2]).astype(np.int32)

        # Assign the heights
        heightMap[hitLocIdx[:, 0], hitLocIdx[:, 1]] = hitLoc[:, 2]

    return heightMap
