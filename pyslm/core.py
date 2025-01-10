import warnings
import itertools
from abc import ABC
from typing import Any, Iterable, List, Optional, Union
import logging
import sys

import numpy as np
import networkx as nx

import trimesh

import shapely.geometry
from shapely.geometry import Polygon
from shapely.ops import unary_union

from scipy.spatial.qhull import ConvexHull


class DocumentObject(ABC):

    def __init__(self, name):
        self._name = name
        self._label = 'Document Object'
        self._attributes = []

    # Attributes are those links to other document objects or properties
    @property
    def attributes(self) -> List[Any]:
        return self._attributes

    @property
    def label(self) -> str:
        return self._label

    @label.setter
    def label(self, label: str) -> None:
        self._label = label

    @property
    def name(self) -> str:
        return self._name

    def _setAttributes(self, attributes: List[Any]):
        self._attributes = attributes

    def setName(self, name):
        self._name = name

    def boundingBox(self):  # const
        raise NotImplementedError('Abstract method should be implemented in derived class')

    def extents(self):
        raise NotImplementedError('Abstract method should be implemented in derived class')


class Document:

    def __init__(self):
        logging.info('Initialising the Document Graph')

        # Create a direct acyclic graph using NetworkX
        self._graph = nx.DiGraph()

    def addObject(self, obj: DocumentObject):

        if not issubclass(type(obj), DocumentObject):
            raise ValueError('Feature {:s} is not a Document Object'.format(obj))

        self._graph.add_node(obj)

        for attr in obj.attributes:
            # Add the sub-features if they do not already exist in the document graph
            if attr is None:
                continue

            self.addObject(attr)

            # Add the dependency link between parent and it's child attributes
            self._graph.add_edge(attr, obj)

        # Update the document accordingly
        self.recalculateDocument()

    def getObjectsByType(self, objType: Any) -> List[DocumentObject]:
        objs = []

        for node in list(self._graph):

            # Determine if the document object requires boundary layers in calculation
            if type(node) is objType:
                objs.append(node)

        return objs

    def recalculateDocument(self) -> None:

        for node in list(nx.dag.topological_sort(self._graph)):

            # Determine if the document object requires boundary layers in calculation
            if type(node).usesBoundaryLayers():

                for childNode in list(nx.dag.ancestors(self._graph, node)):
                    childNode.setRequiresBoundaryLayers()

    @property
    def head(self) -> DocumentObject:
        graphList = list(nx.dag.topological_sort(self._graph))
        return graphList[-1]

    @property
    def parts(self) -> List[DocumentObject]:

        objs = list(self._graph)
        parts = []

        for obj in objs:
            if issubclass(type(obj), Part):
                parts.append(obj)

        return parts

    @property
    def extents(self) -> np.ndarray:
        # Method for calculating the total bounding box size of the document
        bbox = self.boundingBox
        return np.array([bbox[3] - bbox[0],
                         bbox[4] - bbox[1],
                         bbox[5] - bbox[2]])

    @property
    def partExtents(self) -> np.ndarray:
        bbox = self.partBoundingBox
        return np.array([bbox[3] - bbox[0],
                         bbox[4] - bbox[1],
                         bbox[5] - bbox[2]])

    def getDependencyList(self):
        return list(nx.dag.topological_sort(self._graph))

    @property
    def partBoundingBox(self):
        """
        A (nx6) array containing the bounding box for all the parts. This is needed for calculating the grid
        """
        pbbox = np.vstack([part.boundingBox for part in self.parts])
        return np.hstack([np.min(pbbox[:, :3], axis=0), np.max(pbbox[:, 3:], axis=0)])

    @property
    def boundingBox(self) -> np.ndarray:

        graphList = list(nx.dag.topological_sort(self._graph))
        graphList.reverse()
        return graphList[0].boundingBox

    def drawNetworkGraph(self):
        import networkx.drawing
        nodeLabels = [i.name for i in self._graph]
        networkLabels = dict(zip(self._graph, nodeLabels))
        networkx.drawing.draw(self._graph, labels=networkLabels)
        # networkx.drawing.draw_graphviz(self._graph, labels=networkLabels)



class Part(DocumentObject):
    """
    Part represents a solid geometry within the document object tree. Currently, this just represents a single part
    that will eventually be later sliced as part of a document tree structure.

    The part can be transformed and has a position (:attr:`origin`), rotation (:attr:`rotation`) and additional scale
    factor (:attr:`scaleFactor`), which are collectively applied to the geometry in its local coordinate system
    :math:`(x,y,z)`. Changing the geometry using :meth:`setGeometryByMesh` or :meth:`setGeometry` along with any of
    the transformation attributes will set the part dirty and forcing the transformation and geometry to be
    re-computed on the next call in order to obtain the :attr:`geometry`.

    The part is currently based on using a faceted mesh, internally building on capabilities of the Trimesh.

    Generally for AM and 3D printing the following function :meth:`getVectorSlice` is the most useful. This method
    provides the user with a slice for a given Z-plane containing the boundaries consisting of a series of polygons.
    The output from this function is either a list of closed paths (coordinates) or a list of
    :class:`shapely.geometry.Polygon`. A bitmap slice can alternatively be obtained for certain AM process using
    :meth:`~Part.getBitmapSlice` in similar manner.
    """

    _partType = 'Part'
    """ The part type is a static class attribute used for classifying the part when used in the document tree. """

    POLYGON_FIX_EPSILON = 1e-3
    """
    Constant value used for repairing invalid/broken polygon regions obtained using :meth:`getVectorSlice`
    Default value is equivalent to 1 micron.
    """

    def __init__(self, name):

        super().__init__(name)

        self._geometry = None
        self._geometryCache = None

        self._bbox = np.zeros((1, 6))

        self._rotation = np.array((0.0, 0.0, 0.0))
        self._scaleFactor = np.array((1.0, 1.0, 1.0))
        self._origin = np.array((0.0, 0.0, 0.0))
        self._dirty = True

    def __str__(self):
        return 'Part <{:s}>'.format(self.name)

    @property
    def partType(self) -> str:
        """
        The Part type. This will be used in future for the document tree.
        """

        return self._partType
    def isDirty(self) -> bool:
        """
        When a transformation or the geometry object has been changed via methods in the part, the state is toggled
        dirty and the transformation matrix must be re-applied to generate a new internal representation of the
        geometry , which is then cached for future use.

        :return: The current state of the geometry
        """

        return self._dirty

    @property
    def rotation(self) -> np.ndarray:
        """
        The part rotation is a 1x3 numpy array representing the rotations :math:`(\\alpha, \\beta, \\gamma)`
        in degrees about X, Y, Z, applied sequentially in that order.
        """
        return self._rotation

    @rotation.setter
    def rotation(self, rotation: Iterable) -> None:

        rotation = np.asanyarray(rotation)

        if len(rotation) != 3:
            raise ValueError('Rotation value should be 1x3 Numpy array')

        self._rotation = rotation
        self._dirty = True

    @property
    def origin(self) -> np.ndarray:
        """ The origin or the translation of the part"""
        return self._origin

    @origin.setter
    def origin(self, origin: Iterable) -> None:

        origin = np.asanyarray(origin)

        if len(origin) != 3:
            raise ValueError('Origin value should be 1x3 Numpy array')

        self._origin = origin
        self._dirty = True

    @property
    def scaleFactor(self) -> np.ndarray:
        """
        The scale factor is a 1x3 matrix :math:`(s_x, s_y, s_z)` representing the scale factor of the part
        """
        return self._scaleFactor

    @scaleFactor.setter
    def scaleFactor(self, sf: Iterable):

        self._scaleFactor = np.asanyarray(sf).flatten()

        if len(self._scaleFactor) == 1:
            self._scaleFactor = self._scaleFactor * np.ones([3,])

        self._dirty = True

    def dropToPlatform(self, zPos: Optional[float] = 0.0) -> None:
        """
        Drops the part at a set height (`zPos`) from its lowest point from the platform (assumed :math:`z=0`).

        :param zPos: The position the bottom of the part should be suspended above :math:`z=0`
        """

        self.origin[2] = -1.0 * self.boundingBox[2] + zPos
        self._dirty = True

    def getTransform(self) -> np.ndarray:
        """
        Returns the transformation matrix (3x3 numpy matrix) used for the :class:`Part` consisting of a translation
        (:attr:`origin`), a :attr:`rotation` and a :attr:`scaleFactor`
        """

        Sx = trimesh.transformations.scale_matrix(factor=self._scaleFactor[0], direction=[1, 0, 0])
        Sy = trimesh.transformations.scale_matrix(factor=self._scaleFactor[1], direction=[0, 1, 0])
        Sz = trimesh.transformations.scale_matrix(factor=self._scaleFactor[2], direction=[0, 0, 1])
        S = Sx*Sy*Sz
        T = trimesh.transformations.translation_matrix(self._origin)

        alpha, beta, gamma = np.deg2rad((self._rotation))

        R_e = trimesh.transformations.euler_matrix(alpha, beta, gamma, 'rxyz')

        M = trimesh.transformations.concatenate_matrices(T, R_e, S)

        return M

    def setGeometry(self, geometry: Any,
                    fixGeometry: Optional[bool] = True,
                    mergeVertices: Optional[bool] = True) -> None:
        """
        Sets the Part geometry based on either a :class:`trimesh.Trimesh` object or a  file path to a mesh that is
        importable by Trimesh.

        :param geometry: The geometry (can be a trimesh or filename to load from)
        :param fixGeometry: Use Trimesh's utilities to fix the mesh: Default = `True`
        :param mergeVertices:  Merges the vertices of the mesh: Default = `True`
        """

        if isinstance(geometry, trimesh.Trimesh):
            self._geometry = geometry
        else:
            logging.info('Geometry information <{:s}> - [{:s}]'.format(self.name, geometry))
            self._geometry = trimesh.load_mesh(geometry, process=False, use_embree=False, Validate_faces=False)

        if mergeVertices:
            self._geometry.merge_vertices()

        if fixGeometry:
            self._geometry.process(validate=True)
            self._geometry.fix_normals()

        logging.info('\t Bounds: [{:.3f},{:.3f},{:.3f}], [{:.3f},{:.3f},{:.3f}]'.format(*self._geometry.bounds.ravel()))
        logging.info('\t Extent: [{:.3f},{:.3f},{:.3f}]'.format(*self._geometry.extents))

        self.checkGeometry()
        self._dirty = True

    def checkGeometry(self) -> bool:

        if not self.geometry.is_watertight:
            logging.warning('The geometry for {:s} is not watertight'.format(self.name))
            return False
        else:
            return True

    def setGeometryByMesh(self, mesh: trimesh.Trimesh) -> None:
        """
         Sets the Part geometry based on an existing Trimesh object.

         :param mesh: The trimesh object loaded
         """
        self._geometry = mesh
        self._dirty = True

    def getProjectedHull(self, returnPoly: Optional[bool] = False) -> Union[shapely.geometry.Polygon, np.ndarray]:
        """
        The convex hull of the part projected in the Z-direction. This is for convenience when trying to find the
        approximate boundary of the part when used for optimising the layout of parts.

        :param returnPoly: Returns a Shapely Polygon if `True` otherwise a numpy array of the convex hull vertices
        :return: The convex hull of the part
        """

        coords = self.geometry.vertices[:, :2]

        chull = ConvexHull(coords)

        hullCoords = coords[chull.vertices]

        if returnPoly:
            hullCoords = np.append(hullCoords, hullCoords[0, :].reshape(-1, 2), axis=0)
            return Polygon(hullCoords)
        else:
            return hullCoords

    def getProjectedArea(self) -> shapely.geometry.Polygon:
        """
        The resultant projected area of the part projected on the z-axis.

        :return: A Shapely Polygon representing the projected area of the part
        """

        facesCpy = self.geometry.faces

        shapes = self.geometry.vertices[facesCpy, :2]

        triPolys = []

        for face in shapes:
            faceCpy = np.append(face, face[0, :].reshape(-1, 2), axis=0)
            triPolys.append(Polygon(faceCpy))

        return unary_union(triPolys)

    @property
    def boundingBox(self) -> np.ndarray:
        """
        The bounding box of the geometry transformed in the global coordinate frame :math:`(X,Y,Z)`. The bounding
        box is a 1x6 a numpy array consisting of the minimum coordinates followed by the maximum coordinates for the
        corners of the bounding box.
        """

        if not self.geometry:
            raise ValueError('Geometry was not set')
        else:
            return  self.geometry.bounds.flatten()

    @property
    def extents(self) -> np.ndarray:
        """
        The extents the geometry transformed in the global coordinate frame :math:`(X,Y,Z)`. The extents is a 1x3
        numpy array consisting of the linear Cartesian dimensions of the part.
        """

        if not self.geometry:
            raise ValueError('Geometry was not set')

        bbox = self.boundingBox

        return np.array([bbox[3] - bbox[0],
                         bbox[4] - bbox[1],
                         bbox[5] - bbox[2]])

    @property
    def volume(self) -> float:
        """
        The volume of the part geometry

        .. note::

            The volume is calculated using the Trimesh volume property therefore the part must be water-tight to
            obtain correct volumes.
        """
        if not self.geometry.is_volume:
            raise ValueError('Part is not a valid volume')

        return self.geometry.volume

    @property
    def surfaceArea(self) -> float:
        """
        The total surface area of the part geometry
        """
        return float(self.geometry.area)

    @property
    def geometry(self) -> Union[trimesh.Trimesh, None]:
        """
        The geometry of the part with the transformations applied.
        """
        if not self._geometry:
            return None

        if self.isDirty():
            self.regenerate()

        return self._geometryCache

    def regenerate(self) -> None:
        """
        Regenerate the geometry
        """
        logging.debug('Updating {:s} Geometry Representation'.format(self.label))
        self._geometryCache = self._geometry.copy()
        self._geometryCache.apply_transform(self.getTransform())
        self._dirty = False


    def getTrimeshSlice(self, z: float) -> trimesh.path.Path2D:
        """
        The vector slice is created by using `trimesh` to slice the mesh into a polygon - returns a Path2D object

        :param z: The slice's z-position
        :return: The vector slice at the given z level

        """
        if not self.geometry:
            raise ValueError('Geometry was not set')

        if z < self.boundingBox[2] or z > self.boundingBox[5]:
            return []

        transformMat = np.array(([1.0, 0.0, 0.0, 0.0],
                                 [0.0, 1.0, 0.0, 0.0],
                                 [0.0, 0.0, 1.0, 0.0],
                                 [0.0, 0.0, 0.0, 1.0]), dtype=np.float32)

        # Obtain the section through the STL polygon using Trimesh Algorithm (Shapely)
        sections = self.geometry.section(plane_origin=[0.0, 0.0, z],
                                         plane_normal=[0.0, 0.0, 1.0])

        if sections is None:
            return []

        # Obtain the 2D Planar Section at this Z-position
        planarSection, transform = sections.to_planar(transformMat)

        if not planarSection.is_closed:
            # Needed in case there are any holes in the stl mesh
            # Repairs the polygon boundary using a merge function built into Trimesh
            planarSection.fill_gaps(planarSection.scale / 100.0)

        return planarSection

    def getVectorSlice(self, z: float,
                       fixPolygons: Optional[bool] = True,
                       simplificationFactor: Optional[float] = None,
                       simplificationPreserveTopology: Optional[bool] = True,
                       simplificationFactorMode: Optional[str] = 'absolute') -> Any:
        """
        The vector slice is created by using `trimesh` to slice the mesh into a polygon

        :param z: The slice's z-position
        :param returnCoordPaths: If `True` returns a list of closed paths representing the polygon, otherwise Shapely Polygons
        :param fixPolygons: Fixes any polygons during slicing by offset by epsilon value
        :param simplificationFactor:  Simplification factor used for the boundary
        :param simplificationPreserveTopology:  Preserves the slice's topology when using simplification algorithm
        :param simplificationFactorMode: Set mode ('absolute', 'line') for the simplification tolerance calculation

        :return: The vector slice at the given z level
        """
        planarSection = self.getTrimeshSlice(z)

        if not planarSection:
            return []

        # Obtain a closed list of shapely polygons
        polygons = planarSection.polygons_full

        if simplificationFactor:

            if simplificationFactorMode == 'absolute':
                simpFactor = simplificationFactor
            elif simplificationFactorMode == 'bound':
                meanLen = np.mean(planarSection.extents)
                simpFactor = simplificationFactor * meanLen
            elif simplificationFactorMode == 'line':
                pass
            else:
                raise Exception('simplification mode invalid')

            simpPolys = []

            for polygon in polygons:

                if simplificationFactorMode == 'line':
                    coords = np.vstack([polygon.exterior.xy[0], polygon.exterior.xy[1]]).T
                    delta = np.diff(coords, axis=0)
                    dist = np.sqrt(delta[:, 0] * delta[:, 0] + delta[:, 1] * delta[:, 1])
                    simpFactor = np.mean(dist) * simplificationFactor

                simpPolys.append(polygon.simplify(simpFactor, preserve_topology=simplificationPreserveTopology))

            polygons = simpPolys

        # fix polygon
        if fixPolygons:
            fixPolys = []
            for polygon in polygons:
                fixPolys.append(polygon.buffer(Part.POLYGON_FIX_EPSILON))
            polygons = fixPolys

        if returnCoordPaths:
            return Part.path2DToPathList(polygons)
        else:
            return polygons

    @staticmethod
    def path2DToPathList(shapes: List[Polygon]) -> List[np.ndarray]:
        """
        Returns the list of paths and coordinates from a cross-section (i.e. Trimesh Path2D). This is required to be
        done for performing boolean operations and off`setting with the internal PyClipper package.

        :param shapes: A list of :class:`shapely.geometry.Polygon` representing a cross-section or container of
                        closed polygons
        :return: A list of paths (Numpy Coordinate Arrays) describing fully closed and oriented paths.
        """

        # deprecated function
        warnings.warn('This function is deprecated and will be removed in future versions', DeprecationWarning)

        paths = []

        for poly in shapes:
            coords = np.array(poly.exterior.coords)
            paths.append(coords)

            for path in poly.interiors:
                coords = np.array(path.coords)
                paths.append(coords)

        return paths

    def getBitmapSlice(self, z: float, resolution: float,  origin: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Returns a bitmap (binary) image of the slice at position :math:`z` position. The resolution parameter
        can change the required definition for rasterising the slice layer.

        :param z: The z-position to take the slice from
        :param resolution: The resolution of the bitmap to generate [pixels/length unit]
        :param origin: The offset for (0,0) in the bitmap image - defaults to the bounding box minimum (optional)

        :return: A bitmap image for the current slice at position
        """

        if self._geometry is None:
            raise RuntimeError(f"Geometry was not set for Part ({self.name})")

        vectorSlice = self.getTrimeshSlice(z)

        bitmapOrigin = self.boundingBox[:2] if origin is None else origin

        sliceImage = vectorSlice.rasterize(pitch=resolution, origin=bitmapOrigin)

        return np.array(sliceImage)
