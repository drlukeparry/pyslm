"""
Provides supporting functions to generate geometry for support structures
"""

import logging
import collections

import numpy as np
from typing import Any, List, Optional, Tuple

import trimesh
from trimesh import grouping

import shapely.geometry.polygon
from shapely.geometry import Polygon

# Triangulation Libraries
from mapbox_earcut import triangulate_float64, triangulate_float32
from triangle import triangulate

import pycork

from pyslm import pyclipper


def extrudeFace(extrudeMesh: trimesh.Trimesh,
                height: Optional[float] = None,
                heightArray: Optional[np.ndarray] = None) -> trimesh.Trimesh:
    """
    Extrudes a set of connected triangle faces into a prism. This is based on a constant height - or an optional
    height array for corresponding to extrusions to be added to each triangular facet.

    :param extrudeMesh: A mesh consisting of *n* triangular faces for extrusion
    :param height: A constant height to use for the prism extrusion
    :param heightArray: Optional array consisting of *n* heights to extrude each triangular facet
    :return: The extruded prism mesh
    """
    faceMesh = extrudeMesh.copy()

    # Locate boundary nodes/edges of the support face
    interiorEdges = faceMesh.face_adjacency_edges
    aset = set([tuple(x) for x in faceMesh.edges])
    bset = set([tuple(x) for x in interiorEdges])  # Interior edges
    # cset = aset.difference(bset)
    # boundaryEdges = np.array([x for x in aset.difference(bset)])

    # Deep copy the vertices from the face mesh
    triVertCpy = faceMesh.vertices.copy()

    if height is not None:
        triVertCpy[:, 2] = height
    elif heightArray is not None:
        triVertCpy[:, 2] = heightArray
    else:
        triVertCpy[:, 2] = -0.1

    meshInd = np.array([]).reshape((0, 3))
    meshVerts = np.array([]).reshape((0, 3))

    # Count indicator increases the triangle index upon each loop iteration
    cnt = 0

    meshInd = []
    meshVerts = []
    # All projected faces are guaranteed to intersect with face
    for i in range(0, faceMesh.faces.shape[0]):
        # extrude the triangle based on the ray length
        fid = faceMesh.faces[i, :]
        tri_verts = np.array(faceMesh.vertices[fid, :])

        # Create a tri from intersections
        meshVerts.append(tri_verts)
        meshVerts.append(triVertCpy[fid, :])

        # Always create the bottom and top faces
        triInd = np.array([(0, 1, 2),  # Top Face
                           (4, 3, 5)  # Bottom Face
                           ])

        edgeA = {(fid[0], fid[1]), (fid[1], fid[0])}
        edgeB = {(fid[0], fid[2]), (fid[2], fid[0])}
        edgeC = {(fid[1], fid[2]), (fid[2], fid[1])}

        if len(edgeA & bset) == 0:
            triInd = np.vstack([triInd, ((0, 3, 4), (1, 0, 4))])  # Side Face (A)

        if len(edgeB & bset) == 0:
            triInd = np.vstack([triInd, ((0, 5, 3), (0, 2, 5)) ]) # Side Face (B)

        if len(edgeC & bset) == 0:
            triInd = np.vstack([triInd, ((2, 1, 4), (2, 4, 5))])  # Side Face (C)

        triInd += cnt * 6
        cnt += 1

        meshInd.append(triInd)

    meshInd = np.vstack(meshInd)
    meshVerts = np.vstack(meshVerts)

    # Generate the extrusion
    extMesh = trimesh.Trimesh(vertices=meshVerts, faces=meshInd, validate=True, process=True)
    extMesh.fix_normals()

    return extMesh


def boolUnion(meshA: trimesh.Trimesh, meshB: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Performs a Boolean CSG union operation using the `pycork <https://github.com/drlukeparry/pycork>`_ library  between
    two meshes.

    .. note::
        The meshes provided should ideally be watertight (manifold) and have no-self intersecting faces to ensure that
        the underlying Cork Library can correctly perform the operation. The resultant mesh is processed natively
        using Trimesh to merge coincident vertices and remove degenerate faces.


    :param meshA: Mesh A
    :param meshB: Mesh B
    :return: The Boolean union between Mesh A and Mesh B.
    """
    vertsOut, facesOut = pycork.union(meshA.vertices, meshA.faces, meshB.vertices, meshB.faces)

    return trimesh.Trimesh(vertices=vertsOut, faces=facesOut, process=True)


def boolIntersect(meshA: trimesh.Trimesh, meshB: trimesh.Trimesh):
    """
      Performs a Boolean CSG intersection operation using the `pycork <https://github.com/drlukeparry/pycork>`_ library
      between two meshes.

      .. note::
          The meshes provided should ideally be watertight (manifold) and have no-self intersecting faces to ensure that
          the underlying Cork Library can correctly perform the operation. The resultant mesh is processed natively
          using Trimesh to merge coincident vertices and remove degenerate faces.


      :param meshA: Mesh A
      :param meshB: Mesh B
      :return: The Boolean intersection between Mesh A and Mesh B.
      """
    vertsOut, facesOut = pycork.intersection(meshA.vertices, meshA.faces, meshB.vertices, meshB.faces)

    return trimesh.Trimesh(vertices=vertsOut, faces=facesOut, process=True)


def boolDiff(meshA: trimesh.Trimesh, meshB: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Performs a Boolean CSG difference operation using the `pycork <https://github.com/drlukeparry/pycork>`_ library
    between two meshes.

    .. note::
        The meshes provided should ideally be watertight (manifold) and have no-self intersecting faces to ensure that
        the underlying Cork Library can correctly perform the operation. The resultant mesh is processed natively
        using Trimesh to merge coincident vertices and remove degenerate faces.


    :param meshA: Mesh A
    :param meshB: Mesh B
    :return: The Boolean difference between Mesh A and Mesh B.
    """
    vertsOut, facesOut = pycork.difference(meshA.vertices, meshA.faces, meshB.vertices, meshB.faces)

    return trimesh.Trimesh(vertices=vertsOut, faces=facesOut, process=True)


def resolveIntersection(meshA: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Resolves all self-intersections within a meshn using the `pycork <https://github.com/drlukeparry/pycork>`_ library.

    .. note::
        The meshes provided should ideally be watertight (manifold) and have no-self intersecting faces to ensure that
        the underlying Cork Library can correctly perform the operation. The resultant mesh is processed natively
        using Trimesh to merge coincident vertices and remove degenerate faces.


    :param meshA: Mesh A
    :return: Mesh with all intersections resolved
    """
    vertsOut, facesOut = pycork.resolveIntersection(meshA.vertices, meshA.faces)

    return trimesh.Trimesh(vertices=vertsOut, faces=facesOut, process=True)


def createPath2DfromPaths(paths: List[np.ndarray]) -> trimesh.path.Path2D:
    """
    A static helper function that converts PyClipper Paths into a single :class:`trimesh.path.Path2D` object. This
    function is not used for performance reasons, but provides additional capability if required.

    :param paths: A list of paths generated from ClipperLib paths.
    :return: A Path2D object containing a list of Paths.
    """
    loadedLinePaths = [trimesh.path.exchange.misc.lines_to_path(path) for path in list(paths)]
    loadedPaths = [trimesh.path.Path2D(**path, process=False) for path in loadedLinePaths]

    sectionPath = trimesh.path.util.concatenate(loadedPaths)
    return sectionPath


def path2DToPathList(shapes: List[shapely.geometry.polygon.Polygon]) -> List[np.ndarray]:
    """
    Returns the list of paths and coordinates from a cross-section (i.e. :class:`Trimesh.path.Path2D` objects).
    This is required to be done for performing boolean operations and offsetting with the internal PyClipper package.

    :param shapes: A list of Shapely Polygons representing a cross-section or container of
                    closed polygons
    :return: A list of paths (Numpy Coordinate Arrays) describing fully closed and oriented paths.
    """

    paths = []

    for poly in shapes:
        coords = np.array(poly.exterior.coords)
        paths.append(coords)

        for path in poly.interiors:
            coords = np.array(path.coords)
            paths.append(coords)

    return paths


def sortExteriorInteriorRings(polyNode,
                              closePolygon: Optional[bool] = False) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    A recursive function that sorts interior and exterior rings or paths from PyClipper (:class:`pylcipper.PyPolyNode`)`
    objects.

    :param polyNode: The :class:`pyclipper.PyPolyNode` tree defining the polygons and interior holes
    :param closePolygon: If `True`, the contours passed are closed
    :return: A tuple consisting of exterior and interior rings
    """

    import pyslm.hatching

    exteriorRings = []
    interiorRings = []

    if polyNode.Contour:

        contour = pyslm.hatching.BaseHatcher.scaleFromClipper(polyNode.Contour)

        if closePolygon:
            contour.append(contour[0])

        contour = np.array(contour)[:, :2]

        if polyNode.IsHole:
            interiorRings.append(contour)
        else:
            exteriorRings.append(contour)

    for node in polyNode.Childs:

        exteriorChildRings, interiorChildRings = sortExteriorInteriorRings(node, closePolygon)

        exteriorRings += exteriorChildRings
        interiorRings += interiorChildRings

    return exteriorRings, interiorRings


def triangulateShapelyPolygon(polygon: shapely.geometry.Polygon,
                        triangle_args: Optional[str]=None,
                        **kwargs):
    """
    Triangulate a Shapely Polygon  using a python interface to `triangle.c`.

    .. code-block:: bash
        pip install triangle

    :param polygon: Shapely Polygon object to be triangulated
    :param triangle_args: Passed to triangle.triangulate i.e: '`p`', '`pq30`'
    :param kwargs:
    :return: Returns a tuple of vertices and faces
    """

    # set default triangulation arguments if not specified
    if triangle_args is None:
        triangle_args = 'p'

    # turn the polygon in to vertices, segments, and hole points
    arg = _polygon_to_kwargs(polygon)

    # run the triangulation
    result = triangulate(arg, triangle_args)

    return result['vertices'], result['triangles']

def triangulatePolygonFromPaths(exterior: np.ndarray, interiors: List[np.ndarray],
                                triangle_args: str = None,
                                **kwargs):
    """
    Given a list of exterior and interiors triangulation using a
    python interface to `triangle.c`

    .. code-block:: bash
        pip install triangle

    :param exterior: Exterior boundaries of polygon
    :param interiors: Interior boundaries of polygon
    :param triangle_args: Passed to triangle.triangulate i.e: 'p', 'pq30'
    :param kwargs:
    :return: Returns a tuple of vertices and faces
    """

    # set default triangulation arguments if not specified
    if triangle_args is None:
        triangle_args = 'p'

    # turn the polygon in to vertices, segments, and hole points
    arg = _polygon_to_kwargs2(exterior, interiors)

    # run the triangulation
    result = triangulate(arg, triangle_args)

    return result['vertices'], result['triangles']


def _polygon_to_kwargs2(exterior: np.ndarray, interiors:  List[np.ndarray]):
    """
    Given both exterior and interior boundaries, create input for the
    the triangle mesh generator. This is version #2 which has been  adapted from
    the Trimesh library to be more efficient for the specific use cases

    :param exterior: List of exterior paths
    :param interiors: List of interior paths
    :return:  Has keys: vertices, segments, holes
    """

    #if not polygon.is_valid:
    #    raise ValueError('invalid shapely polygon passed!')

    def round_trip2(start, length):
        """
        Given a start index and length, create a series of (n, 2) edges which
        create a closed traversal.

        Examples
        ---------
        start, length = 0, 3
        returns:  [(0,1), (1,2), (2,0)]
        """
        tiled = np.tile(np.arange(start, start + length).reshape((-1, 1)), 2)
        tiled = tiled.reshape(-1)[1:-1].reshape((-1, 2))
        tiled = np.vstack((tiled, [tiled[-1][-1], tiled[0][0]]))
        return tiled


    def add_boundary2(boundary, start, clean=True):
        # TODO THIS SECTION IS A VERY BIG BOTTLENECK FOR TRIANGULATION OF POLYGONS

        # coords is an (n, 2) ordered list of points on the polygon boundary
        # the first and last points are the same, and there are no
        # guarantees on points not being duplicated (which will
        # later cause meshpy/triangle to shit a brick)

        # find indices points which occur only once, and sort them
        # to maintain order
        if clean:
            coords = boundary
            unique = np.sort(grouping.unique_rows(coords)[0])
            cleaned = coords[unique]

            vertices.append(cleaned)
            facets.append(round_trip2(start, len(cleaned)))
            test = Polygon(cleaned)
            lenCoords = len(cleaned)
        else:
            coords = boundary
            coords = boundary[:-1,:]
            vertices.append(coords)
            facets.append(round_trip2(start, len(coords)))
            test = Polygon(coords)
            lenCoords = len(coords)

        repPnt = test.representative_point()
        holes.append(np.array(repPnt.coords)[0])

        return lenCoords

    # sequence of (n,2) points in space
    vertices = collections.deque()
    # sequence of (n,2) indices of vertices
    facets = collections.deque()
    # list of (2) vertices in interior of hole regions
    holes = collections.deque()

    start = add_boundary2(exterior, 0)
    for interior in interiors:
        try:
            start += add_boundary2(interior, start, clean=True)
        except BaseException:
            logging.warning('invalid interior, continuing')
            continue

    # create clean (n,2) float array of vertices
    # and (m, 2) int array of facets
    # by stacking the sequence of (p,2) arrays
    vertices = np.vstack(vertices)
    facets = np.vstack(facets).tolist()

    # shapely polygons can include a Z component
    # strip it out for the triangulation
    if vertices.shape[1] == 3:
        vertices = vertices[:, :2]

    result = {'vertices': vertices,
              'segments': facets}
    # holes in meshpy lingo are a (h, 2) list of (x,y) points
    # which are inside the region of the hole
    # we added a hole for the exterior, which we slice away here
    holes = np.array(holes)[1:]
    if len(holes) > 0:
        result['holes'] = holes

    return result


def _polygon_to_kwargs(polygon):
    """
    Given a Shapely Polygon creates the input for the
    the triangle mesh generator. This is version #2 which has been  adapted from
    the Trimesh library to be more efficient for the specific use cases

    :param polygon: Shapely Polygon to proces
    :return:  Has keys: vertices, segments, holes
    """

    #if not polygon.is_valid:
    #    raise ValueError('invalid shapely polygon passed!')

    def round_trip(start, length):
        """
        Given a start index and length, create a series of (n, 2) edges which
        create a closed traversal.

        Examples
        ---------
        start, length = 0, 3
        returns:  [(0,1), (1,2), (2,0)]
        """
        tiled = np.tile(np.arange(start, start + length).reshape((-1, 1)), 2)
        tiled = tiled.reshape(-1)[1:-1].reshape((-1, 2))
        tiled = np.vstack((tiled, [tiled[-1][-1], tiled[0][0]]))
        return tiled

    def add_boundary(boundary, start, clean=True):
        # coords is an (n, 2) ordered list of points on the polygon boundary
        # the first and last points are the same, and there are no
        # guarantees on points not being duplicated (which will
        # later cause meshpy/triangle to shit a brick)

        # find indices points which occur only once, and sort them
        # to maintain order
        if clean:
            coords = np.array(boundary.coords)
            unique = np.sort(grouping.unique_rows(coords)[0])
            cleaned = coords[unique]

            vertices.append(cleaned)
            facets.append(round_trip(start, len(cleaned)))
            test = Polygon(cleaned)
            lenCoords = len(cleaned)
        else:
            coords = np.array(boundary.coords)[:-1,:]
            vertices.append(coords)
            facets.append(round_trip(start, len(coords)))
            test = Polygon(coords)
            lenCoords = len(coords)

        repPnt = test.representative_point()
        holes.append(np.array(repPnt.coords)[0])

        return lenCoords

    # sequence of (n,2) points in space
    vertices = collections.deque()
    # sequence of (n,2) indices of vertices
    facets = collections.deque()
    # list of (2) vertices in interior of hole regions
    holes = collections.deque()

    start = add_boundary(polygon.exterior, 0)
    for interior in polygon.interiors:
        try:
            start += add_boundary(interior, start, clean=False)
        except BaseException:
            logging.warning('invalid interior, continuing')
            continue

    # create clean (n,2) float array of vertices
    # and (m, 2) int array of facets
    # by stacking the sequence of (p,2) arrays
    vertices = np.vstack(vertices)
    facets = np.vstack(facets).tolist()

    # shapely polygons can include a Z component
    # strip it out for the triangulation
    if vertices.shape[1] == 3:
        vertices = vertices[:, :2]

    result = {'vertices': vertices,
              'segments': facets}
    # holes in meshpy lingo are a (h, 2) list of (x,y) points
    # which are inside the region of the hole
    # we added a hole for the exterior, which we slice away here
    holes = np.array(holes)[1:]
    if len(holes) > 0:
        result['holes'] = holes

    return result


def triangulatePolygon(section,
                       closed: Optional[bool] = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function triangulates polygons generated natively by PyClipper, from :class:`pyclipper.PyPolyNode` objects.
    This is specifically used to optimally generate the polygon triangulations using an external triangulation
    library Mapbox using the Ear Clipping algorithm - see `Mapbox <https://github.com/mapbox/earcut.hpp>`_ and
    the `Ear-Cut <https://pypi.org/project/mapbox-earcut/>`_ PyPi package .

    By using the :class:`pyclipper.PyPolyNode` object, ClipperLib automatically generates a polygon hierarchy tree for
    separating both external contours and internal holes, which can be passed directly to the earcut algorithm.
    Otherwise, this requires passing all paths and sorting these to identify interior holes.

    :param section: A :class:`pyclipper.PyPolyNode` object containing a collection of polygons
    :param closed: If the polygo is already closed
    :return: A tuple of vertices and faces generated from the triangulation
    """

    vertIdx = 0
    meshFaces = []
    meshVerts = []

    """
    For multiple polygons, we know the exteriors must not overlap therefore they can be treat as independent meshes
    when they sorted    
    """
    for polygon in section.Childs:

        exterior, interior = sortExteriorInteriorRings(polygon)

        exteriorPath2D = np.array(exterior[0])[:, :2]

        interiorPath2D = []
        for path in interior:
            if closed:
                coords = np.array(path)[:-1, :2]
            else:
                coords = np.array(path)[:, :2]
            interiorPath2D.append(coords)

        # get vertices as sequence where exterior is the first value
        vertices = [np.array(exteriorPath2D)]
        vertices.extend(np.array(i) for i in interiorPath2D)

        # record the index from the length of each vertex array
        rings = np.cumsum([len(v) for v in vertices])

        # stack vertices into (n, 2) float array
        triVerts = np.vstack(vertices)

        # run triangulation
        triangFaces = triangulate_float32(triVerts, rings).reshape( (-1, 3)) #.astype(np.int64).reshape((-1, 3))
        triFaces = triangFaces + vertIdx

        meshFaces.append(triFaces)
        meshVerts.append(triVerts)
        vertIdx += len(triVerts)

        # Save below as a reference in the future if required
        #vy = np.insert(triVerts, 2, values=0.0, axis=1)
        #meshes.append(trimesh.Trimesh(vertices=vy, faces=triangFaces))

    meshFaces = np.vstack(meshFaces)
    meshVerts = np.vstack(meshVerts)

    #mesh = trimesh.util.concatenate(meshes)
    return meshVerts, meshFaces


def generatePolygonBoundingBox(bbox: np.ndarray) -> shapely.geometry.Polygon:
    """
    Generates a Shapely Polygon based on the extents of the bounding box of the object passed

    :param bbox: The bounding box representing the gemometry
    :return: A 2D polygon representing the bounding box
    """

    bx = bbox[:, 0]
    by = bbox[:, 1]
    bz = bbox[:, 2]

    a = [np.min(bx), np.max(bx)]
    b = [np.min(by), np.max(by)]

    # Create a closed polygon representing the transformed slice geometry
    bboxPoly = shapely.geometry.polygon.Polygon([[a[0], b[0]],
                                                 [a[0], b[1]],
                                                 [a[1], b[1]],
                                                 [a[1], b[0]],
                                                 [a[0], b[0]]])

    return bboxPoly
