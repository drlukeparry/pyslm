import abc
import time
from typing import List, Optional, Union
import itertools

import numpy as np

import trimesh.intersections

from .utils import mesh_plane
from ..hatching import utils as hatchUtils


def binarySort(z_slices: List[float], mesh: trimesh.Trimesh):
    """
    Efficiently assigns triangles to z-height layers using binary search.

    :param z_slices: A list of z-positions used to identify intersecting triangless
    :param mesh: A trimesh to slice from
    :return: A list of sorted triangle indices corresponding to each z-height layers
    """
    # Get triangle z-bounds
    triangles = mesh.triangles
    z_values = triangles[:, :, 2]
    tri_min = np.min(z_values, axis=1)
    tri_max = np.max(z_values, axis=1)

    # Binary search to find which layers each triangle spans
    min_idx = np.searchsorted(z_slices, tri_min, side='left')
    max_idx = np.searchsorted(z_slices, tri_max, side='left')

    # Clip to valid range
    min_idx = np.clip(min_idx, 0, len(z_slices) - 1)
    max_idx = np.clip(max_idx, 0, len(z_slices) - 1)

    # Assign triangles to layers
    layers = [[] for _ in range(len(z_slices))]
    for tri_id in range(len(tri_min)):
        for layer_id in range(min_idx[tri_id], max_idx[tri_id] + 1):
            layers[layer_id].append(tri_id)

    # Convert to sorted numpy arrays
    return [np.sort(np.array(layer, dtype=np.int64)) for layer in layers]

class Slicer(abc.ABC):

    def __init__(self, mesh):

        self._mesh = mesh

    @property
    def bounds(self):
        return self._mesh.bounds

    @property
    def mesh(self):
        return self._mesh

    @abc.abstractmethod
    def getSlice(self):
        return NotImplementedError()

    @abc.abstractmethod
    def getSlices(self):
        return NotImplementedError()

class BasicUniformSlicer(Slicer):

    POLYGON_FIX_EPSILON = 1e-3
    """
    Constant value used for repairing invalid/broken polygon regions obtained using :meth:`getVectorSlice`
    Default value is equivalent to 1 micron.
    """

    def __init__(self, mesh):

        super().__init__(mesh)

        self._init = False

        # Common Flags
        self._simplificationFactorMode = 'absolute'
        self._simplificationFactor = 1.0
        self._simplificationPreserveTopology = True
        self._returnCoordPaths = True
        self._processPaths = False

    @property
    def layerThickness(self):
        return self._layerThickness

    @property
    def returnCoordPaths(self) -> bool:
        return self._returnCoordPaths

    @returnCoordPaths.setter
    def returnCoordPaths(self, state: bool):
        self._returnCoordPaths = state

    @property
    def processPaths(self) -> bool:
        return self._processPaths

    @processPaths.setter
    def processPaths(self, state):
        self._processPaths = state

    def init(self, zMin: float, zMax: float,  zThickness: float):

        self._layerThickness = zThickness

        # Generate a list of slices for each layer across the entire mesh

        self._zMin = self.mesh.bounds[0,2] if zMin is None else zMin
        self._zMax = self.mesh.bounds[2, 2] if zMin is None else zMax

        self._init = True

        return self

    def getSlice(self, z, process=False, returnCoordPaths=True):

        if z < self._mesh.bounds[0,2] or z > self._mesh.bounds[1,2]:
            return None

        transformMat = np.array(([1.0, 0.0, 0.0, 0.0],
                                 [0.0, 1.0, 0.0, 0.0],
                                 [0.0, 0.0, 1.0, 0.0],
                                 [0.0, 0.0, 0.0, 1.0]), dtype=np.float32)

        from trimesh.intersections import mesh_plane

        # Calculate the slice at this layer
        sections = mesh_plane(mesh=self._mesh,
                              plane_origin=[0.0, 0.0, z], plane_normal=[0.0, 0.0, 1.0])

        # otherwise load the line segments into a Path2D object
        planarSection = trimesh.load_path(sections)

        if planarSection is None:
            return None

        # Obtain the 2D Planar Section at this Z-position
        planarSection, transform = sections.to_planar(transformMat)

        if not planarSection.is_closed:
            # Needed in case there are any holes in the stl mesh
            # Repairs the polygon boundary using a merge function built into Trimesh
            planarSection.fill_gaps(planarSection.scale / 100.0)

        paths = planarSection.polygons_full
        paths = self.simplifyPaths(paths, process=process, returnCoordPaths=returnCoordPaths)

        return planarSection

    def getSlices(self, process: bool=True, returnCoordPaths: bool=True):

        heights = np.arange(0.0, self._mesh.bounds[1,2], self._layerThickness)

        from trimesh.intersections import mesh_multiplane

        # Calculate the slice at this layer
        sections, to3D, faceIds  = mesh_multiplane(mesh=self._mesh,
                                    plane_origin=[0.0, 0.0, 0], plane_normal=[0.0, 0.0, 1.0],
                                    heights=heights)

        planarPaths = []
        for i,section in enumerate(sections):
            print(i)
            # otherwise load the line segments into a Path2D object
            planarSection = trimesh.load_path(section)

            if sections is None:
                planarPaths.append([])

            if not planarSection.is_closed:
                # Needed in case there are any holes in the stl mesh
                # Repairs the polygon boundary using a merge function built into Trimesh
                planarSection.fill_gaps(planarSection.scale / 100.0)

            paths = planarSection.polygons_full

            paths = self.simplifyPaths(paths, process=process, returnCoordPaths=returnCoordPaths)
            planarPaths.append(paths)

        return planarPaths

    @property
    def simplificationFactorMode(self) -> str:
        return self._simplificationFactorMode

    @simplificationFactorMode.setter
    def simplificationFactorMode(self, simplificationFactorMode: str):

        if simplificationFactorMode == 'absolute':
            pass
        elif simplificationFactorMode == 'bound':
            pass
        elif simplificationFactorMode == 'line':
            raise NotImplementedError('Line simplification mode not implemented')
        elif simplificationFactorMode == None:
            pass
        else:
            raise ValueError(f"Simplification mode ({simplificationFactorMode} invalid")

        self._simplificationFactorMode = simplificationFactorMode

    @property
    def simplificationPreserveTopology(self):
        return self._simplificationPreserveTopology

    @simplificationPreserveTopology.setter
    def simplificationPreserveTopology(self, state: bool):
        self._simplificationPreserveTopology = state
    @property
    def simplificationFactor(self) -> float:
        return self._simplificationFactor

    @simplificationFactor.setter
    def simplificationFactor(self, factor: float):
        self._simplificationFactor = factor

    def processSegs(self, segs: list[np.ndarray]):

        # if the section didn't hit the mesh return None
        if len(segs) == 0:
            return None

        # flatten the dimensions  to 2d for segs
        segs = [seg[:,:2] for seg in segs]

        # otherwise load the line segments into a Path3D object
        path = trimesh.load_path(segs)

        if path is None:
            return []

        # Obtain a closed list of shapely polygons
        polygons = path.polygons_full

        return polygons

    def simplifyPaths(self, polygons, process: bool = True, returnCoordPaths: bool = False):

        if self._simplificationFactorMode == 'absolute':
            simpFactor = self._simplificationFactor
        elif self._simplificationFactorMode == 'bound':
            meanLen = np.mean(polygons.extents)
            simpFactor = self._simplificationFactor * meanLen
        elif self._simplificationFactorMode == 'line':
            raise NotImplementedError('Line simplification mode not implemented')
        elif self._simplificationFactorMode == None:
            pass
        else:
            raise ValueError(f"Simplification mode ({self._simplificationFactorMode} invalid")

        simpPolys = []

        for polygon in polygons:


            if self._simplificationFactor == 'line':
                coords = np.vstack([polygon.exterior.xy[0], polygon.exterior.xy[1]]).T
                delta = np.diff(coords, axis=0)
                dist = np.sqrt(delta[:, 0] * delta[:, 0] + delta[:, 1] * delta[:, 1])
                simpFactor = np.mean(dist) * simpFactor

            if self._simplificationFactorMode is not None:
                simpPolys.append(polygon.simplify(simpFactor, preserve_topology=self._simplificationPreserveTopology))
            else:
                simpPolys.append(polygon)

        polygons = simpPolys

        # fix polygon
        if process:
            polygons = [poly.buffer(BasicUniformSlicer.POLYGON_FIX_EPSILON) for poly in polygons]

        if returnCoordPaths:
            return list(itertools.chain.from_iterable([hatchUtils.poly2Paths(poly) for poly in polygons]))
        else:
            return polygons



class UniformSlicer(BasicUniformSlicer):

    def __init__(self, mesh):

        super().__init__(mesh)

        self._init = False


    def init(self, zMin: float, zMax: float, zThickness: float):

        super().init(zMin, zMax, zThickness)

        zBox = np.arange(self._zMin, self._zMax, self._layerThickness)

        # store z positions and count for iterator support
        self._zBox = zBox
        self._num_layers = len(zBox)

        k = len(zBox) # number of slices

        # Attach the corresponding presorted triangles into the
        self._layers = binarySort(zBox, self._mesh)

        # The iterative part for assigning potential triangles for intersection on a triangle are performed here
        # Note: Process is very inefficient in native Python O(n*k)

        self._vertexDots = np.dot(self._mesh.vertices - np.array([0, 0, 0.0]),  np.array([0, 0, 1.0]))

        # Toggle the flag to indiciate initialised
        self._init = True

        # Return self as an iterator
        return self


    def _getSegments(self, z: float, intersectTris: np.ndarray):

        new_dots = self._vertexDots[self._mesh.faces[intersectTris]] - z

        # Calculate the slice at this layer
        segs = mesh_plane(mesh=self._mesh,
                          plane_origin=[0.0, 0.0, z], plane_normal=[0.0, 0.0, 1.0],
                          local_faces=intersectTris, cached_dots=new_dots,
                          tris=np.array(self._mesh.faces))

        if segs is None:
            segs = []

        return segs

    def getSlice(self, z, process: bool=True, returnCoordPaths=True):

        if not self._init:
            raise Exception('Slicer not initialised!')

        raw = (z - self._zMin) / self._layerThickness
        idx = int(np.floor(raw))

        intersectTris = self._layers[idx]

        fixPolygons = process

        segs = self._getSegments(z, intersectTris)
        paths = self.processSegs(segs)
        paths = self.simplifyPaths(paths, process=process, returnCoordPaths=returnCoordPaths)

        return paths


    def getSlices(self, process: bool=True, returnCoordPaths: bool=True):

        if not self._init:
            raise Exception('Slicer not initialised!')

        slices = []

        for i, intersectTris in enumerate(self._layers):

            # Obtain the interpolated z-position
            zMin, zMax = self._mesh.bounds[:, 2]
            z = i*self._layerThickness + zMin

            segs = self._getSegments(z, intersectTris)
            paths = self.processSegs(segs)
            paths = self.simplifyPaths(paths, process=process, returnCoordPaths=returnCoordPaths)

            slices.append(paths)

        return slices

    def __len__(self):
        if not self._init:
            return 0
        return int(getattr(self, '_num_layers', len(self._layers)))

    def __iter__(self):
        if not self._init:
            raise Exception('Slicer not initialised!')
        # initialize an iterator pointer
        self._iter_index = 0
        return self

    def __next__(self):
        if not self._init:
            raise Exception('Slicer not initialised!')

        if getattr(self, '_iter_index', 0) >= len(self):
            raise StopIteration

        idx = self._iter_index
        self._iter_index += 1

        z = float(self._zBox[idx])
        paths = self.getSlice(z, process=self._processPaths, returnCoordPaths=self._returnCoordPaths)
        return paths

    def __getitem__(self, key):
        """Support indexing and slicing like a sequence: slicer[i] or slicer[start:stop:step]."""
        if isinstance(key, int):
            z = float(self._zBox[key])
            return self.getSlice(z, process=self._processPaths, returnCoordPaths=self._returnCoordPaths)
        if isinstance(key, slice):
            indices = range(*key.indices(len(self)))
            zPos = [ float(self._zBox[key]) for key in indices ]
            return [self.getSlice(z, process=self._processPaths, returnCoordPaths=self._returnCoordPaths) for z in zPos]
        raise TypeError('Indices must be integers or slices')

    def iter(self):
        """Generator yielding (index, z, paths) for each layer without exhausting the object iterator.

        Example:
            for idx, z, paths in slicer.iter_slices():
                ...
        """
        if not self._init:
            raise Exception('Slicer not initialised!')

        for i in range(len(self)):
            z = float(self._zBox[i])
            paths = self.getSlice(z, process=self._processPaths, returnCoordPaths=self._returnCoordPaths)
            yield i, z, paths
