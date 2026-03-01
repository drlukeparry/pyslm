"""
Provides classes  and methods for the creation of support structures in Additive Manufacturing.
"""

try:
    import triangle
except Exception:
    raise Exception("Lib Triangle is required to use support.geometry submodule")

try:
    import mapbox_earcut
except Exception:
    raise Exception("Mapbox earcut is required to use the support.geometry submodule")

try:
    import wgpu
except Exception:
    raise Exception("Wgpu is required to use the support.geometry submodule")

import abc

from typing import Any, Optional, List, Tuple, Union
import logging
import time
import warnings

import scipy.ndimage.filters
import skimage.measure

import shapely.geometry
import shapely.affinity

import numpy as np
import trimesh
import trimesh.path
import trimesh.path.traversal

from ..core import Part
from ..hatching import utils as hatchingUtils

from . import geometry
from . import render
from . import utils


class SupportStructure(abc.ABC):
    """
    The Support Structure is the base class representing all definitions of support structures used in PySLM.
    This includes attributes that link to the source part and the original support faces or features of an object and
    shared utility methods for the calculation of useful properties.

    It stores properties that describe the type of support identified during its generation and stores relevant data
    connecting this such as the original support surface (:attr:`supportSurface`) and
    the support object or part (:attr:`supportObject`) and if the support self-intersects with
    the part (:attr:`supportObject`).
    """
    def __init__(self,
                 supportObject: Optional[Part] = None,
                 supportVolume: Optional[trimesh.Trimesh] = None,
                 supportSurface: Optional[trimesh.Trimesh] = None,
                 intersectsPart: bool = False):

        self._supportVolume = supportVolume
        self._supportObject = supportObject
        self._supportSurface = supportSurface
        self._intersectsPart = intersectsPart

    def __str__(self) -> str:
        return 'SupportStructure'

    @abc.abstractmethod
    def geometry(self) -> trimesh.Trimesh:
        """
        Returns the geometry representing the support structure
        """
        raise NotImplementedError('Geometry property is an abstract method')

    @staticmethod
    def flattenSupportRegion(region: trimesh.Trimesh) -> shapely.geometry.Polygon:
        """
        The function takes a support surfaces and flattens this as a projected polygon.

        :param region: The support surface as a :class:`trimesh.Trimesh` mesh
        :return: The 2D Polygon of the flattened surface
        """

        # [FIX] - should this return multiple polygon regions

        supportRegion = region.copy()

        """ Extract the outline of the overhang mesh region"""
        poly = supportRegion.outline()

        """ Convert the line to a 2D polygon"""
        poly.vertices[:, 2] = 0.0

        flattenPath, polygonTransform = poly.to_planar()
        flattenPath.process()

        flattenPath.apply_translation(polygonTransform[:2, 3])  # np.array([polygonTransform[0, 3],

        # flattenPath = flattenPath.simplify_spline(smooth=1000)
        polygon = flattenPath.polygons_full[0]

        return polygon

    @property
    @abc.abstractmethod
    def volume(self) -> float:
        """
        Returns the  volume of the Support Geometry
        """
        raise NotImplementedError('Support Volume property is an abstract method')

    def projectedSupportArea(self) -> float:
        """
        Convenience function returns the total projected surface area of the support region.

        .. note::
            The area returned is unit-less

        :return: The total projected (flattened) surface support area
        """
        if self._supportSurface:
            return float(self.flattenSupportRegion(self._supportSurface).area)
        else:
            return 0.0

    def supportArea(self) -> float:
        """
        Convenience function returns the total surface area of the support region.

        :return: The total surface area of the support
        """

        return self._supportSurface.area if self._supportSurface else 0.0

    @property
    def intersectsPart(self) -> bool:
        """
        Indicates if the projected support structure intersect with the originating part
        """
        return self._intersectsPart

    @intersectsPart.setter
    def intersectsPart(self, state: bool) -> None:
        self._intersectsPart = state

    @property
    def supportSurface(self) -> Union[trimesh.Trimesh, None]:
        """
        The support surface identified on the originating part
        """
        return self._supportSurface

    @supportSurface.setter
    def supportSurface(self, surface: trimesh.Trimesh) -> None:
        self._supportSurface = surface

    @property
    def supportObject(self) -> Union[Any, None]:
        """
        The originating object that the support structure is generated for.
        """
        return self._supportObject

    @supportObject.setter
    def supportObject(self, obj: Any) -> None:
        self._supportObject = obj


class BlockSupportBase(SupportStructure):
    """
    The BlockSupportBase is a base class representing **a single support volume** region constructed by an extruded
    overhang surface region, that may intersect with the build platform (:math:`z=0`) or self-intersect with the
    original mesh of the :class:`Part`.

    These are generated externally in the :class:`BlockSupportGenerator` and other derived generator classes that
    build upon this. Objects represent the data structure for the support structure rather than the methods for
    generating themselves.

    The support volume (:attr:`supportVolume`) is a generic 3D volume body or mesh that enables
    differentiation of support structures to be generated by creating a derived class that re-defines the
    abstract method :meth:`SupportStructure.geometry`.
    """

    def __init__(self,
                 supportObject: Optional[Part] = None,
                 supportVolume: Optional[trimesh.Trimesh] = None,
                 supportSurface: Optional[trimesh.Trimesh] = None,
                 intersectsPart: bool = False):

        super().__init__(supportObject, supportVolume, supportSurface, intersectsPart)

    def __str__(self) -> str:
        return 'BlockSupportBase'

    def geometry(self) -> trimesh.Trimesh:
        """
        Returns the geometry representing the support structure.
        """
        return self._supportVolume

    @property
    def volume(self) -> float:
        """
        The calculated volume of the support volume region.
        """
        return float(self._supportVolume.volume)

    @property
    def supportVolume(self) -> trimesh.Trimesh:
        """
        The support volume stores the 3D mesh geometry representing an extruded geometry projected onto either the
        part surface or build-plate (:math:`z=0`). This is generated externally in :class:`BlockSupportGenerator` and
        the resultant block 3D geometry is stored in this property.
        """
        return self._supportVolume

    @supportVolume.setter
    def supportVolume(self, supportVolume: trimesh.Trimesh) -> None:
        self._supportVolume = supportVolume

    @property
    def supportBoundary(self) -> trimesh.Trimesh:
        """
        The boundary or vertical walls constructed from the extruded support volume. These are identified by
        taking using :meth:`utils.getFaceZProjectionWeight` and then using a default threshold value in
        the private static attribute :attr:`BlockSupportBase._supportSkinSideTolerance`

        .. note::
            Any self-intersections with the object geometry that are steep (~90 degrees) may potentially be included.
        """

        blockSupportSides = self._supportVolume.copy()
        sin_theta = utils.getFaceZProjectionWeight(blockSupportSides)

        blockSupportSides.update_faces(sin_theta > (1.0-1e-4))
        blockSupportSides.remove_unreferenced_vertices()

        return blockSupportSides

    @staticmethod
    def triangulateSections(sections: list[trimesh.path.Path2D]) -> trimesh.Trimesh:
        """
        A static method to take a collection of section slice or cross-section and triangulate them into a combined
        mesh. The triangulated meshed are then transformed based on the original transformation generated internally
        when using :meth:`trimesh.Trimesh.section`.

        :param sections: The sections to triangulate into a mesh
        :return: A mesh containing the concatenated triangulated polygon sections
        """
        sectionMesh = trimesh.Trimesh()

        for section in sections:
            if section is None:
                continue

            v, f = section.triangulate()

            if len(v) == 0:
                continue

            v = np.insert(v, 2, values=0.0, axis=1)
            sec = trimesh.Trimesh(vertices=v, faces=f)
            sec.apply_transform(section.metadata['to_3D'])
            sectionMesh += sec

        return sectionMesh


class BaseSupportGenerator:
    """
    The BaseSupportGeneration class provides common methods used for generating the support structures
    (:class:`SupportStructure`) typically used in Additive Manufacturing.

    This class provides the base methods used for identifying geometrical unsupported features
    such as vertices and edges within a part.
    """

    PYCLIPPER_SCALEFACTOR = 1e4
    """
    The scaling factor used for polygon clipping and offsetting in `pyclipr <https://github.com/drlukeparry/pyclipr>`_
    for the decimal component of each polygon coordinate. This should be set to inverse of the required decimal
    tolerance i.e. `0.01` requires a minimum scale factor of `1e2`. Default is `1e4`.
    """

    POINT_OVERHANG_TOLERANCE = 0.05
    """
    The point overhang tolerance is used for determining if adjacent connected vertices in the mesh lies above,
    which indicates that this vertex requires an additional point support generating.
    """

    def __init__(self) -> None:
        pass

    def __str__(self) -> str:
        return 'BaseSupportGenerator'

    @staticmethod
    def findOverhangPoints(part: Part) -> np.ndarray:
        """
        Identifies vertices that require additional support based on their connectivity with adjacent vertices.

        :param part: The part to locate unsupported vertices
        :return: Identified points that require additional support
        """
        meshVerts = part.geometry.vertices
        vAdjacency = part.geometry.vertex_neighbors

        pointOverhangs = []
        for i in range(len(vAdjacency)):

            # Find the edge deltas between the points
            v = meshVerts[i]
            neighborVerts = meshVerts[vAdjacency[i], :]
            delta = neighborVerts - v
            # mag = np.sqrt(np.sum(delta * delta, axis=1))
            # theta = np.arcsin(delta[:,2]/mag)
            # theta = np.rad2deg(theta)
            # if np.all(theta > -0.001):
            # pointOverhang.append(i)

            """
            If all neighbouring connected vertices lie above the point, this indicates the vertex lies below and 'may'
            not have underlying connectivity. There are two cases that exist: on upwards or downwards pointing surface.
            """
            if np.all(delta[:, 2] > -BaseSupportGenerator.POINT_OVERHANG_TOLERANCE):

                # Check that the vertex normal is pointing downwards (-ve Z) showing that the no material is underneath
                if part.geometry.vertex_normals[i][2] < 0.0:
                    pointOverhangs.append(i)

        return pointOverhangs

    @staticmethod
    def findOverhangEdges(part: Part,
                          overhangAngle: float = 45.0,
                          edgeOverhangAngle: float = 10.0) -> List[Tuple[int,int]]:
        """
        Identifies edges which requires additional support based on both the support surface and support edge angle.

        :param part: The part to be analysed
        :param overhangAngle: The support surface overhang angle (degrees)
        :param edgeOverhangAngle: The edge overhang angle (degrees)

        :return: A list of edge tuples.
        """

        mesh = part.geometry
        edges = mesh.edges_unique
        edgeVerts = mesh.vertices[edges]

        """
        Calculate the face angles with respect to the +z vector  and the inter-face angles
        """
        theta = utils.getSupportAngles(part, np.array([[0., 0., 1.0]]))
        adjacentFaceAngles = np.rad2deg(mesh.face_adjacency_angles)

        overhangEdges = []

        # Iterate through all the edges in the model
        for i in range(len(edgeVerts)):

            """
            Calculate the 'vertical' angle of the edge pointing in the z-direction by using the z component.
            First calculate vector, magnitude and the vertical angle of the vector
            """
            edge = edgeVerts[i].reshape(2, 3)
            delta = edge[0] - edge[1]
            mag = np.sqrt(np.sum(delta * delta))
            ang = np.rad2deg(np.arcsin(delta[2] / mag))

            # Identify if the vertical angle of the edge is less than the edgeOverhangAngle irrespective of the actual
            # direction of the vector (bidirectional)
            if np.abs(ang) < edgeOverhangAngle:

                """
                Locate the adjacent faces in the model using the face-adjacency property to identify if the edge
                belongs to a sharp corner which tends to be susceptible areas. This is done by calculating the angle
                between faces.
                """
                adjacentFaces = mesh.face_adjacency[i]

                if adjacentFaceAngles[i] > overhangAngle and np.all(theta[adjacentFaces] > 89):
                    overhangEdges.append(edges[i])

        return overhangEdges


class BlockSupportGenerator(BaseSupportGenerator):
    """
    The BlockSupportGenerator class provides common methods used for generating the 'support' structures typically used
    in Additive Manufacturing. Derived classes can build directly upon this by either using existing BlockSupports
    generated or redefining the overall support geometry created underneath overhang regions.

    After passing the geometry and setting the required parameters, the user is required to call
    :meth:`identifySupportRegions` in order to generate the support volumes.

    In summary, the technique identifies first overhang surface regions on the mesh geometry provided based on the
    chosen :attr:`overhangAngle`. From these identified overhang surfaces,  extruded prisms are generated and are
    then intersected with the original part in both :math:`+Z` and :math:`-Z` directions using the Boolean CSG
    library (`manifold <https://github.com/elalish/manifold>`_). This method provides the means to approximately isolate
    regions and fundamentally decide if the supports are self-intersecting (:attr:`SupportStructure.intersectsPart`).
    Non-intersecting regions are connected to the build-plate only and are excluded from further processing.

    Each of these regions are then ray traced using an OpenGL depth technique to identify the support regions using the
    private method :meth:`_identifySelfIntersectionHeightMap`.

    Intersecting regions with the part are identified and these can be more smartly separated based on a tolerance
    :meth:`gradThreshold` calculated from the equivalent ray or rasterised projection resolution
    :attr:`rayProjectionResolution` and :attr:`overhangAngle` previously defined. Regions identified are
    simplified from a calculated heightMap image and then approximate support extrusions are generated that intersect
    with the originating part by adding Z-offsets (:attr:`lowerProjectionOffset` and :attr:`upperProjectionOffset`).

    Finally, these extruded regions are intersected with the part using the CSG library to produce the final
    :class:`BlockSupportBase` that precisely conforms the boundary of the part if there are self-intersections.
    """

    _supportSkinSideTolerance = 1.0 - 1e-3
    """
    The support skin side tolerance is used for masking the extrusions side faces when generating the polygon region
    for creating the surrounding support skin.

    By masking the regions, the upper and lower surfaces of the extruded volume are separated and their 3D boundaries
    can be extracted.
    """

    _intersectionVolumeTolerance = 50
    """
    An internal tolerances used to determine if the projected volume intersects with the part
    """

    _gaussian_blur_sigma = 1.0
    """
    The internal parameter is used for blurring the calculated depth field to smooth out the boundaries. Care should
    be taken to keep this low as it will artificially offset the boundary of the support
    """

    def __init__(self) -> None:

        super().__init__()

        self._minimumAreaThreshold = 5.0  # mm^2 (default = 5.0)
        self._rayProjectionResolution = 'auto'  # mm (default = 0.2)
        self._splineSimplificationFactor = 'auto' # default = auto

        self._lowerProjectionOffset = 0.05 # mm
        self._upperProjectionOffset = 0.05 # mm

        self._innerSupportEdgeGap = 0.2  # mm (default = 0.1)
        self._outerSupportEdgeGap = 0.5  # mm  - offset between part supports and baseplate supports
        self._simplifyMeshPolygonFactor = 0.5 ## default = 0.5 (used for simplifying the polygon following isocurve extraction

        self._triangulationSpacing = 2.0  # mm (default = 1)

        self._overhangAngle = 45.0  # [deg]

        self._useApproxBasePlateSupport = False  #
        self._depthRenderer = None

    def __str__(self) -> str:
        return 'BlockSupportGenerator'

    @staticmethod
    def gradThreshold(rayProjectionDistance: float, overhangAngle: float) -> float:
        """
        A static method which defines the threshold  applied to the gradient generated from the support
        depth map, which separates each support volume region. This is based on a combination of the ray projection
        resolution, the overhang angle and an arbitrary constant to ensure discrete regions are isolated.

        :param rayProjectionDistance: The ray projection resolution used
        :param overhangAngle: The overhang angle [degrees]
        :return: The gradient threshold used.

        """
        return 5.0 * np.tan(np.deg2rad(overhangAngle)) * rayProjectionDistance

    @property
    def splineSimplificationFactor(self) -> Union[float,str]:
        """
        The simplification factor using a spline approximation approach for smoothening the support volume boundary.

        This can either be a float value or the string 'auto' which automatically selects a value based on the
        polygonal cross-sectional area and perimeter of each region.
        """
        return self._splineSimplificationFactor

    @splineSimplificationFactor.setter
    def splineSimplificationFactor(self, value: Union[float,str]) -> None:
        self._splineSimplificationFactor = value

    @property
    def overhangAngle(self) -> float:
        """ The overhang angle (degrees) used for identifying support surfaces on the :class:`Part` """
        return self._overhangAngle

    @overhangAngle.setter
    def overhangAngle(self, angle: float) -> None:
        self._overhangAngle = angle

    @property
    def upperProjectionOffset(self) -> float:
        """
        An internal parameter used for defining an offset applied to the upper projection used to provide a clean
        intersection when performing the final boolean intersection between the original geometry and the extruded
        support volume geometry.
        """
        return self._upperProjectionOffset

    @upperProjectionOffset.setter
    def upperProjectionOffset(self, offset: float) -> None:
        self._upperProjectionOffset = offset

    @property
    def lowerProjectionOffset(self) -> float:
        """
        The offset applied to the lower projection used to provide a clean intersection when performing the final
        boolean intersection between the original geometry and the extruded support volume geometry.
        """
        return self._lowerProjectionOffset

    @lowerProjectionOffset.setter
    def lowerProjectionOffset(self, offset: float) -> None:
        self._lowerProjectionOffset = offset

    @property
    def outerSupportEdgeGap(self) -> float:
        """ The offset applied to the  projected boundary of the support volume."""
        return self._outerSupportEdgeGap

    @outerSupportEdgeGap.setter
    def outerSupportEdgeGap(self, spacing: float) -> None:
        self._outerSupportEdgeGap = spacing

    @property
    def innerSupportEdgeGap(self) -> float:
        """
        The inner support gap is the distance between adjacent supports regions that are identified as separated by a
        significant vertical extent.
        """
        return self._innerSupportEdgeGap

    @innerSupportEdgeGap.setter
    def innerSupportEdgeGap(self, spacing: float) -> None:
        self._innerSupportEdgeGap = spacing

    @property
    def minimumAreaThreshold(self) -> float:
        """
        The minimum support area threshold (:math:`mm^2`) used to identify disconnected support regions.
        Support regions with a smaller area will be excluded and not generated.
        """
        return self._minimumAreaThreshold

    @minimumAreaThreshold.setter
    def minimumAreaThreshold(self, areaThresholdValue: float) -> None:
        self._minimumAreaThreshold = areaThresholdValue

    @property
    def simplifyMeshPolygonFactor(self) -> float:
        """
        The simplification factor used for simplifying the boundary polygon generated from the rasterisation process.
        It is factor applied in-conjunction with self.rayProjectionResolution when extracting the isocurve from the height map.
        """
        return self._simplifyMeshPolygonFactor

    @simplifyMeshPolygonFactor.setter
    def simplifyMeshPolygonFactor(self, value: float) -> None:
        self._simplifyMeshPolygonFactor = value

    @property
    def triangulationSpacing(self) -> float:
        """ The spacing factor used whilst triangulating the support polygon region."""
        return self._triangulationSpacing

    @triangulationSpacing.setter
    def triangulationSpacing(self, spacing: float) -> None:
        self._triangulationSpacing = spacing

    @property
    def rayProjectionResolution(self) -> Union[float, str]:
        """
        The equivalent ray projection resolution used to discretise the projected support region using the OpenGL
        rasterisation. This can be adjusted accordingly depending on the overall scale and size of the part mesh,
        although this is mostly insignificant due to the relatively high performance using OpenGL.

        The resolution should be selected to appropriately capture the complexity of the features within the part.

        .. note::
            There is a restriction on the maximum size based on the framebuffer memory available during GPU
             rendering, which is dependent on the Operating System and drivers.

        Alternatively, the option 'auto' can be selected which automatically selects a resolution based on the areal
        size of each individual region,  rather than a fixed resolution for all regions. This is useful when the
        order of the overhang surface feature size varies significantly across the part.

        """
        return self._rayProjectionResolution

    @rayProjectionResolution.setter
    def rayProjectionResolution(self, resolution: Union[float, str]) -> None:

        if isinstance(resolution, str):
            if resolution != 'auto':
                raise ValueError('Ray Projection Resolution is not "auto" or numerical value')

        self._rayProjectionResolution = resolution

    def filterSupportRegion(self, region):
        """ Not implemented """
        raise NotImplementedError('Not Implemented')

    def generateIntersectionHeightMap(self) -> None:
        """ Not implemented """
        raise NotImplementedError('Not Implemented')


    @staticmethod
    def _identifySelfIntersectionHeightMap(subregion: trimesh.Trimesh,
                                           rayProjectionResolution: float,
                                           cutMesh: trimesh.Trimesh,
                                           bbox: np.ndarray) -> Tuple[np.ndarray]:
        """
        Generates the height map of the upper and lower depths. This is done by projecting rays at a resolution
        (attr:`~BlockSupportGenerator.rayProjectionResolution`) across the entire polygon region (offsetPoly) in both
        vertical directions (+z, -z) and are intersected with the upper and lower support surface. A sequence of
        height maps are generated from these ray intersections.

        :param subregion: The upper surface (typically overhang surface region)
        :param rayProjectionResolution: The ray projection resolution
        :param cutMesh: The lower intersecting surfaces which potentially intersect with the polygon region
        :return: A tuple containing various height maps
        """

        warnings.warn('This function is deprecated and will be removed in the future',
                      DeprecationWarning, stacklevel=2)

        logging.info('\tGenerated support height map (OpenGL Version)')

        # Extend the bounding box extents in the Z direction
        bboxCpy = bbox.copy()
        bboxCpy[0, 2] -= 1
        bboxCpy[1, 2] += 1

        if len(subregion.triangles) == 0:
            raise Exception('Cut mesh has no triangles!')

        upperImg = render.projectHeightMap(subregion, rayProjectionResolution, False, bboxCpy)

        if len(cutMesh.triangles) == 0:
            lowerImg = np.zeros(upperImg.shape)
        else:

            # Cut mesh is lower surface
            lowerImg = render.projectHeightMap(cutMesh, rayProjectionResolution, True, bboxCpy)
            lowerImg = np.flipud(lowerImg)

        # Generate the difference between upper and lower ray-traced intersections
        heightMap2 = upperImg.copy()
        mask = lowerImg > 1.01
        heightMap2[mask] = lowerImg[mask]

        return heightMap2.T, upperImg, lowerImg


    def _identifyOccludedHeightMap(self,
                                   subregion: trimesh.Trimesh,
                                   rayProjectionResolution: float,
                                   partMesh: trimesh.Trimesh,
                                   bbox: np.ndarray = None) -> Tuple[np.ndarray]:
        """
        Optimised version of :meth:`BlockSupportGenerator_identifySelfIntersectionHeightMap` using
        meth:`render3.LowLevelDepthRenderer`.

        The renderer is stored as an object attribute (self._depthRenderer) and is lazily initialized
        on first use. For subsequent calls, only the mesh data is updated (fast GPU upload).

        :param subregion: The upper surface (typically overhang surface region)
        :param rayProjectionResolution: The ray resolution
        :param part: The lower intersecting surfaces which potentially intersect with the polygon region
        :param bbox: The bounding box for the region
        :return: A tuple containing various height maps (heightMap2.T, upperImg, lowerImg)

        """
        from . import occlusionRenderer

        logging.info('\tGenerated support height map - occluded version (WGPU Version)')

        # Extend the bounding box extents in the Z direction
        if bbox is None:
            bboxCpy = subregion.bounds.copy()
        else:
            bboxCpy = bbox.copy()

        bboxCpy[0, 2] -= 1
        bboxCpy[1, 2] += 1
        #bboxCpy[:,2] += BlockSupportGenerator.TOL_OFFSET

        if len(subregion.triangles) == 0:
            raise Exception('Subregion mesh has no triangles!')

        # Initialize renderer lazily on first use
        if self._depthRenderer is None:
            logging.info('\tInitialising reusable WGPU depth renderer')

            # Initialize renderer once
            self._depthRenderer = occlusionRenderer.OverhangMaskRenderer(
                mesh=subregion,
                overhangMesh=subregion,
                bbox=subregion.bounds,
                resolution=rayProjectionResolution,
                flipDir=True
            )


        # Reuse existing renderer - just update mesh and resolution
        self._depthRenderer.setMesh(mesh=subregion, overhangMesh=subregion, bbox=subregion.bounds, resolution=rayProjectionResolution)
        self._depthRenderer.flipDir = True

        # Render upper surface
        self._depthRenderer.renderWithoutOverhangMask()
        upperImg = self._depthRenderer.heightMap

        # Update renderer for lower surface (flip direction)
        self._depthRenderer.setMesh(mesh=partMesh, overhangMesh=subregion, bbox=subregion.bounds, resolution=rayProjectionResolution)
        self._depthRenderer.flipDir = True
        self._depthRenderer.renderWithOverhangMask()

        lowerImg = self._depthRenderer.heightMap
        upperImg = np.flipud(upperImg)
        lowerImg = np.flipud(lowerImg)

        # Generate the difference between upper and lower ray-traced intersections
        heightMapDelta = upperImg.copy()
        mask = lowerImg > 1.01
        heightMapDelta[mask] = lowerImg[mask]

        return heightMapDelta.T, upperImg, lowerImg


    def _identifySelfIntersectionHeightMapRayTracing(self, subregion: trimesh.Trimesh,
                                                     offsetPoly: trimesh.path.Path2D,
                                                     cutMesh: trimesh.Trimesh) -> Tuple[np.ndarray]:
        """
        Deprecated: Generates the height map of the upper and lower depths. This is done by projecting rays at a
        resolution (attr:`~BlockSupportGenerator.rayProjectionResolution`) across the entire polygon region (
        offsetPoly) in both vertical directions (+z, -z) and are intersected with the upper and lower support
        surface. A sequence of height maps are generated from these ray intersections.

        :param subregion: The upper surface (typically overhang surface region)
        :param offsetPoly: The polygon region defining the support region
        :param cutMesh: The lower intersecting surfaces which potentially intersect with the polygon region
        :return: A tuple containing various height maps
        """

        warnings.warn('This function is deprecated and will be removed in the future',
                      DeprecationWarning, stacklevel=2)

        # Rasterise the surface of overhang to generate projection points
        supportArea = np.array(offsetPoly.rasterize(self.rayProjectionResolution, offsetPoly.bounds[0, :])).T

        coords = np.argwhere(supportArea).astype(np.float32) * self.rayProjectionResolution
        coords += offsetPoly.bounds[0, :] + 1e-5  # An offset is required due to rounding error

        logging.info('\t - start projecting rays')
        logging.info('\t - number of rays with resolution ({:.3f}): {:d}'.format(self.rayProjectionResolution, len(coords)))

        """
        Project upwards to intersect with the upper surface
        """
        # Set the z-coordinates for the ray origin
        coords = np.insert(coords, 2, values=-1e5, axis=1)
        rays = np.repeat([[0., 0., 1.]], coords.shape[0], axis=0)

        # Find the first location of any triangles which intersect with the part
        hitLoc, index_ray, index_tri = subregion.ray.intersects_location(ray_origins=coords,
                                                                         ray_directions=rays,
                                                                         multiple_hits=False)
        logging.info('\t - finished projecting rays')

        coords2 = coords.copy()

        coords2[index_ray, 2] = 1e7
        rays[:, 2] = -1.0

        # If any vertices in triangle there is an intersection
        # Find the first location of any triangles which intersect with the part
        hitLoc2, index_ray2, index_tri2 = cutMesh.ray.intersects_location(ray_origins=coords2,
                                                                          ray_directions=rays,
                                                                          multiple_hits=False)

        logging.info('\t - finished projecting rays')

        # Create a height map of the projection rays
        heightMap = np.ones(supportArea.shape) * -1.0

        heightMapUpper = np.zeros(supportArea.shape)
        heightMapLower = np.zeros(supportArea.shape)

        if len(hitLoc) > 0:
            hitLocCpy = hitLoc.copy()
            hitLocCpy[:, :2] -= offsetPoly.bounds[0, :]
            hitLocCpy[:, :2] /= self.rayProjectionResolution

            hitLocIdx = np.ceil(hitLocCpy[:, :2]).astype(np.int32)

            # Assign the heights
            heightMap[hitLocIdx[:, 0], hitLocIdx[:, 1]] = hitLoc[:, 2]
            heightMapUpper[hitLocIdx[:, 0], hitLocIdx[:, 1]] = hitLoc[:, 2]

        if len(hitLoc2) > 0:
            hitLocCpy2 = hitLoc2.copy()
            # Update the xy coordinates
            hitLocCpy2[:, :2] -= offsetPoly.bounds[0, :]
            hitLocCpy2[:, :2] /= self.rayProjectionResolution
            hitLocIdx2 = np.ceil(hitLocCpy2[:, :2]).astype(np.int32)
            # Assign the heights based on the lower projection
            heightMap[hitLocIdx2[:, 0], hitLocIdx2[:, 1]] = hitLoc2[:, 2]
            heightMapLower[hitLocIdx2[:, 0], hitLocIdx2[:, 1]] = hitLoc2[:, 2]

        logging.info('\tgenerated support height map')

        return heightMap, heightMapUpper, heightMapLower


    def pointsToSplineEntity(self, points, smooth=None, count=None):

        from scipy.interpolate import splprep

        if count is None:
            count = len(points)
        if smooth is None:
            smooth = 0.002

        points = np.asanyarray(points, dtype=np.float64)
        closed = np.linalg.norm(points[0] - points[-1]) < 1e-8

        if closed:
            interp_point = (points[0] + points[-1]) / 2.0
            #print(points[-1], points[0])
            #points = np.vstack([points[:-1], interp_point])
            #points = points[:-1]


        knots, control, degree = splprep(points.T, s=smooth)[0]

        control = np.transpose(control)
        index = np.arange(len(control))

        if closed:
            control[0] = control[[0, -1]].mean(axis=0)
            control = control[:-1]
            index[-1] = index[0]

        entity = trimesh.path.entities.BSpline(points=index,
                                  knots=knots,
                                  closed=closed)

        return entity, control

    def simplifySpline(self, path, smooth=None, verbose=False):
        """
        Simplify a Path2D object using spline fitting.

        :param path: The path to simplify
        :param smooth: The smoothness factor used internally by the scipy spline fitting function
        :param verbose:
        :return:
        """
        newVertices = []
        newEntities = []
        scale = path.scale

        for discrete in path.discrete:

            # entities for this path
            entity, vertices = self.pointsToSplineEntity(discrete, smooth=smooth)
            # reindex returned control points
            entity.points += len(newVertices)
            # save entity and vertices
            newVertices.extend(vertices)
            newEntities.append(entity)

        # create the Path2D object for the result
        simplified = type(path)(entities=newEntities,
                                vertices=newVertices)

        return simplified

    def projectSupportVolume(self, poly: shapely.geometry.Polygon,
                             bbox: np.ndarray, rayProjectionRes: float,
                             heightMapUpper: np.ndarray, heightMapLower: np.ndarray,
                             findSelfIntersectingSupport: bool = True) -> Union[trimesh.Trimesh, bool]:
        """
        Projects the support polygon using sample heights from heightMapUpper and heightMapLower and extrudes the
        polygon  non-uniformly across the lower and upper height maps to produce a support volume regions.

        :param poly: The support region polygon to extrude
        :param bbox: The bounding box for the polygon region
        :param rayProjectionRes: The ray projection resolution used for the creation of the height maps
        :param heightMapUpper: The previously identified height map (upper-surface)
        :param heightMapLower:The previously identified height map (lower-surface)
        :return: A Tuple of the trimesh mesh representing the extruded support volume, and a boolean indicating if the support
                 intersects with the part.
        """

        meanThickness, minThickness, maxThickness = geometry.calculatePolygonThickness(poly, rayProjectionRes)

        """
        Triangulate the polygon into a planar mesh
        """
        triangleSpacing = min(meanThickness / 3.0, rayProjectionRes)
        triangleSpacing = rayProjectionRes
        logging.info(f"  \t - Projecting Support Volume")
        logging.info(f"  \t   - Mean Thickness: {meanThickness:.3f} mm")
        logging.info(f"  \t   - Triangle Spacing {triangleSpacing:.3f}")
        poly_tri = trimesh.creation.triangulate_polygon(poly,
                                                        triangle_args='pa{:.3f}'.format(triangleSpacing),
                                                        engine='triangle')

        # Convert polygon vertices to heightMap indices
        verticesOffset = poly_tri[0] - bbox[0, :2]
        verticesIndices = (verticesOffset / rayProjectionRes).astype(np.int32)

        # Add padding offset
        verticesIndices += 2

        totalBooleanTime = 0.0
        # Clamp indices to valid range
        verticesIndices[:, 0] = np.clip(verticesIndices[:, 0], 0, heightMapUpper.shape[0] - 1)
        verticesIndices[:, 1] = np.clip(verticesIndices[:, 1], 0, heightMapUpper.shape[1] - 1)

        # Sample heights from both heightMaps
        upperHeights = heightMapUpper[verticesIndices[:, 0], verticesIndices[:, 1]] - BlockSupportGenerator.TOL_OFFSET
        lowerHeights = heightMapLower[verticesIndices[:, 0], verticesIndices[:, 1]] - BlockSupportGenerator.TOL_OFFSET


        intersectsPart = np.any(lowerHeights > 1e-5)

        if not findSelfIntersectingSupport and intersectsPart:
            return trimesh.Trimesh(), intersectsPart

        # Create upper surface vertices
        coords2 = np.column_stack([poly_tri[0], upperHeights + self.upperProjectionOffset])

        # Create lower surface vertices
        coords3 = np.column_stack([poly_tri[0], lowerHeights - self.lowerProjectionOffset])

        # Handle invalid heights (baseplate support case)
        mask_invalid_lower = lowerHeights < 1e-5
        if np.any(mask_invalid_lower):
            coords3[mask_invalid_lower, 2] = 0.0

        mask_invalid_upper = upperHeights < -1
        if np.any(mask_invalid_upper):
            coords2[mask_invalid_upper, 2] = 0.0
            raise Exception('Projections are not matching (Upper Surface) - please report a bug')


        # set the z coordinates for the lower surface
        coords1 = np.column_stack([poly_tri[0], np.array(0) * np.ones(len(poly_tri[0]))])

        # Create the upper surface mesh
        surf2 = trimesh.Trimesh(vertices=coords1, faces=poly_tri[1], process=True)

        # Perform a simple 2D prismatic extrusion on the mesh
        supportVolume = BlockSupportGenerator.extrudeTriangulation(surf2.vertices[:, :2], surf2.faces, 100)

        if True:
            """ Section for coloring the bottom and upper surfaces for visualisation purposes """
            # Colour-code faces based on metadata['face_type'] (if available).
            # Initialize default face colours (RGBA)
            default_color = np.array([200, 200, 200, 255], dtype=np.uint8)
            supportVolume.visual.face_colors = np.tile(default_color, (len(supportVolume.faces), 1))

            faceType = supportVolume.metadata.get('face_type') if hasattr(supportVolume, 'metadata') else None
            if faceType is not None:
                ft = np.array(faceType, dtype=object)
                # Attempt string labels first, fallback to numeric codes
                lowerFaces = np.array([], dtype=int)
                upperFaces = np.array([], dtype=int)
                try:
                    lowerFaces = np.where(ft == 'bottom')[0]
                    upperFaces = np.where(ft == 'top')[0]
                except Exception:
                    try:
                        lowerFaces = np.where(ft == 0)[0]
                        upperFaces = np.where(ft == 2)[0]
                    except Exception:
                        pass

                if lowerFaces.size:
                    supportVolume.visual.face_colors[lowerFaces, :3] = np.array([255, 128, 0], dtype=np.uint8)  # bottom: orange
                if upperFaces.size:
                    supportVolume.visual.face_colors[upperFaces, :3] = np.array([0, 128, 255], dtype=np.uint8)  # top: cyan

        # Identify the upper and lower surfaces based on the prismatic extrusion
        lowerIdx = supportVolume.vertices[:, 2] < 1
        upperIdx = supportVolume.vertices[:, 2] > 1

        # Assign the coordinates for the upper and lower surface
        supportVolume.vertices[lowerIdx] = coords2
        supportVolume.vertices[upperIdx] = coords3

        supportVolume.fix_normals()
        supportVolume.merge_vertices()

        #ab.show(smooth=False)
        return supportVolume, intersectsPart


            """
            Create an extrusion at the vertical extent of the part and perform self-intersection test
            """
            extruMesh2Flat = subregion.copy()
            extruMesh2Flat.vertices[:, 2] = 0.0

            extruMesh2 = trimesh.creation.extrude_triangulation(extruMesh2Flat.vertices[:, :2], extruMesh2Flat.faces, 100)

            # Position the upper-surface of the mesh just below the upper surface (1e-2) to avoid self-intersection
            eMesh2Idx = extruMesh2.vertices[:, 2] > 1.0
            extruMesh2.vertices[eMesh2Idx, 2] = subregion.vertices[:, 2] - 0.01
            extruMesh = extruMesh2
            #extruMesh = extrudeFace(subregion, 0.0)
            #extruMesh.vertices[:, 2] = extruMesh.vertices[:, 2] - 0.01

            timeIntersect = time.time()

            logging.info('\t - start intersecting mesh')

            bbox = extruMesh.bounds
            cutMesh = geometry.boolIntersect(part.geometry, extruMesh)
            logging.info('\t\t - Mesh intersection time using manifold: {:.3f}s'.format(time.time() - timeIntersect))
            logging.info('\t -  Finished intersecting mesh')
            totalBooleanTime += time.time() - timeIntersect

            # Note this a hard tolerance
            if cutMesh.volume < BlockSupportGenerator._intersectionVolumeTolerance:

                if self._useApproxBasePlateSupport:
                    """
                    Create a support structure that extends to the base plate (z=0)

                    NOTE - not currently used - edge smoothing cannot be performed despite this being a
                    quicker methods, it suffer sever quality issues with jagged edges so should be avoided.
                    """
                    logging.info('Creating Approximate Base-Plate Support')

                    extruMesh.visual.face_colors[:, :3] = np.random.randint(254, size=3)

                    # Create a support block object
                    baseSupportBlock = BlockSupportBase(supportObject=part,
                                                        supportVolume=extruMesh,
                                                        supportSurface=subregion)

                    supportBlockRegions.append(baseSupportBlock)

                    continue  # No self intersection with the part has taken place with the support
            elif not findSelfIntersectingSupport:
                continue

            v0 = np.array([[0., 0., 1.0]])

            # Identify Support Angles
            v1 = cutMesh.face_normals
            theta = np.arccos(np.clip(np.dot(v0, v1.T), -1.0, 1.0))
            theta = np.degrees(theta).flatten()

            cutMeshUpper = cutMesh.copy()
            cutMeshUpper.update_faces(theta < 89.95)
            cutMeshUpper.remove_unreferenced_vertices()

            # Toggle to use full intersecting mesh
            TOL_OFFSET = 1000
            cutMeshUpperCpy = cutMeshUpper.copy()
            cutMeshUpperCpy.vertices[:, 2] += TOL_OFFSET

            subregionCpy = subregion.copy()
            subregionCpy.vertices[:, 2] += TOL_OFFSET

            # Use a ray-tracing approach to identify self-intersections. This provides a method to isolate regions that
            # either are self-intersecting or not.
    def processOutline(self, outline: shapely.geometry.Polygon,
                       rayProjectionResolution: float,
                       useSplineSimplification: Optional[bool] = True) -> List[shapely.geometry.Polygon]:

        """
        Process the polygon by creating a shapely polygon and offsetting the boundary before creating the support
        volume region.

        :param outline: The identified polygon outline to use.
        :param rayProjectionResolution: The Ray Projection Resolution used for rasterisation of the support region
        :param useSplineSimplification:  If `True` (default) spline simplification is applied to the polygon boundary
        :return:
        """

        mergedPoly = trimesh.load_path(outline, process=True)

            logging.info('\t - finished generated support height map')
        if useSplineSimplification:

            heightMap = np.pad(heightMap, ((2, 2), (2, 2)), 'constant', constant_values=((1, 1), (1, 1)))
            heightMapUpper = np.pad(heightMapUpper.T, ((2, 2), (2, 2)), 'constant', constant_values=((1, 1), (1, 1)))
            # round the factor up to the nearest base 10 value
            initial_factor = mergedPoly.area / mergedPoly.length
            logging.debug(f' \t - Spline simplification factor (area/length): {initial_factor}')
            factor = 10 ** np.ceil(np.log10(initial_factor))

            # See the disucssion page (https://github.com/scipy/scipy/issues/11916)  regarding auto mode
            fudge_factor = 1.0
            factor = fudge_factor * len(mergedPoly.discrete[0]) * rayProjectionResolution ** 2

            logging.debug(f'Final spline simplification factor: {factor}')

            # mergedPoly = self.simplify_spline(mergedPoly,factor)
            if self._splineSimplificationFactor == 'auto':
                mergedPoly = self.simplifySpline(mergedPoly, smooth=factor)
            elif self._splineSimplificationFactor is not None:
                mergedPoly = mergedPoly.simplify_spline(self._splineSimplificationFactor)

        try:
            outPolygons = mergedPoly.polygons_full

        except:
            return []
            raise Exception('Issued occurred processing outline for support - please submit a bug report')

        if not mergedPoly.is_closed or len(outPolygons) == 0 or outPolygons[0] is None:
            return []

        if len(outPolygons) > 1:
            raise Exception('MultiPolygon error please submit a bug report')

        # Simplify the polygon boundary
        processedBoundaries = []
        for poly in outPolygons:

            simplifiedPoly = poly.simplify(self.simplifyMeshPolygonFactor * rayProjectionResolution, preserve_topology=True)

            if isinstance(self.innerSupportEdgeGap, str):
                if self.innerSupportEdgeGap == 'auto':
                    innerSupportEdgeGap = 2.0 * rayProjectionResolution
                else:
                    raise ValueError('innerSupportEdgeGap string value not recognised')
            else:
                innerSupportEdgeGap = self.innerSupportEdgeGap


            bufferPoly = simplifiedPoly.buffer(-innerSupportEdgeGap)

            if isinstance(bufferPoly, shapely.geometry.MultiPolygon):
                processedBoundaries += bufferPoly.geoms
            else:
                processedBoundaries.append(bufferPoly)

        polygons = []
        for poly in processedBoundaries:
            if poly.area > self.minimumAreaThreshold:
                polygons.append(poly)

        return polygons


    def identifySupportRegions(self, part: Part,
                               overhangAngle: float,
                               findSelfIntersectingSupport: bool = True) -> List[BlockSupportBase]:
        """
        Extracts the overhang mesh and generates block regions given a part and target overhang angle. The algorithm
        uses a combination of boolean operations and ray intersection/projection to discriminate support regions.
        If :code:`findSelfIntersectingSupport` is to set :code:`True` (default), the algorithm will process and
        separate overhang regions that by downward projection self-intersect with the part.

        This provides more refined behavior than simply projected support material downwards into larger support
        block regions and separates an overhang surface between intersecting and non-intersecting regions.

        :param part: Part for generating support structures for
        :param overhangAngle: Overhang angle (degrees)
        :param findSelfIntersectingSupport: Generates supports that intersect with the part

        :return: A list of BlockSupports
        """

        overhangSubregions = utils.getOverhangMesh(part, overhangAngle, splitMesh=True)
        supportBlockRegions = []

        totalBooleanTime = 0.0

        """ Process sub-regions"""
        for i, subregion in enumerate(overhangSubregions):

            logging.info(f"Processing Subregion [{i:d}/{len(overhangSubregions):d}] - area: {subregion.area :.3f} mm^2")

            if subregion.area < self.minimumAreaThreshold:
                logging.info(f' - Note: skipping shape (area ({subregion.area:.4f} < {self.minimumAreaThreshold})')
                continue

            if self.rayProjectionResolution == 'auto':
                # get the subregion size and set the ray projection resolution accordingly. 0.01 mm corresponds to 10 microns
                bboxSize = subregion.bounding_box.extents
                rayProjectionRes = max(BlockSupportGenerator._min_ray_projection_resolution, max(bboxSize) / 500.0)
                logging.info(f"\t - Region ray projection resolution (auto): {rayProjectionRes:.3f}")
            else:
                rayProjectionRes = self.rayProjectionResolution

            """
            Create an extrusion at the vertical extent of the part and perform self-intersection test
            """
            bbox = subregion.bounds.copy()

            subregionCpy = subregion.copy()
            subregionCpy.vertices[:, 2] -= 0.05

            # Use a ray-tracing approach to identify self-intersections. This provides a method to isolate regions that
            # either are self-intersecting or not.

            logging.info('\t - Starting extraction of support height map')

            try:
                heightMap, heightMapUpper, heightMapLower = self._identifyOccludedHeightMap(subregionCpy,
                                                                                            rayProjectionRes,
                                                                                            partMesh=part.geometry)
            except Exception as e:

                print(bbox)
                print('subregion cpy', subregionCpy)
                print('error', e)
                raise Exception(e)

            logging.info('\t - Finished generated support height map')

            heightMap = np.pad(heightMap, ((2, 2), (2, 2)), 'constant', constant_values=((1, 1), (1, 1)))
            heightMapUpper = np.pad(heightMapUpper.T, ((2, 2), (2, 2)), 'constant', constant_values=((1, 1), (1, 1)))
            heightMapLower = np.pad(heightMapLower.T, ((2, 2), (2, 2)), 'constant', constant_values=((1, 1), (1, 1)))

            """ Calculate the Gradients within the Height Map"""
            vx, vy = np.gradient(heightMap)
            grads = np.sqrt(vx ** 2 + vy ** 2)
            grads = scipy.ndimage.filters.gaussian_filter(grads, sigma=BlockSupportGenerator._gaussian_blur_sigma)

            """
            Find the outlines of any regions of the height map which deviate significantly
            This is used to separate both self-intersecting supports and those which are simply connected
            to the base-plate.
            """
            gradThreshold = self.gradThreshold(rayProjectionRes, overhangAngle)

            outlines = skimage.measure.find_contours(grads, gradThreshold,
                                                     mask=heightMap > (BlockSupportGenerator.TOL_OFFSET-1))

            # Transform the outlines from image to global coordinates system
            outlinesTrans = []
            for outline in outlines:
                outlinesTrans.append(outline * rayProjectionRes + bbox[0, :2])

            # Convert outlines into closed polygons
            outlinePolygons = hatchingUtils.pathsToClosedPolygons(outlinesTrans)

            extrudedBlocks = []

            # Process the outlines found from the contours
            for outline in outlinePolygons:

                blocks = []
                try:
                    polygons = self.processOutline(outline, rayProjectionRes, useSplineSimplification=True)
                    blocks = [self.projectSupportVolume(poly, bbox, rayProjectionRes, heightMapUpper, heightMapLower,
                                                        findSelfIntersectingSupport) for poly in polygons]

                except:
                    # Attempt without spline simplification
                    print('Error processing outline - attempting without spline simplification')
                    polygons = self.processOutline(outline, rayProjectionRes, useSplineSimplification=False)
                    blocks = [self.projectSupportVolume(poly, bbox, rayProjectionRes, heightMapUpper, heightMapLower,
                                                        findSelfIntersectingSupport) for poly in polygons]

                # Remove any empty block meshes
                blocks = [block for block in blocks if not block[0].is_empty]

                extrudedBlocks += blocks

            for extrudedBlock, isIntersecting in extrudedBlocks:

                """
                Take the near net-shape support and obtain the difference with the original part to get clean
                boundaries for the support.
         
                Note: Previous mesh was used in Version 0.5,6. This was not necessarily required, but offers the most robust
                implementation dealing with self-intersections
                """

                if False:
                    logging.info('\t\t Boolean Difference Time: {:.3f}\n'.format(time.time() - timeDiff))
                    blockSupportMesh = geometry.boolDiff(extrudedBlock, part.geometry)
                    totalBooleanTime += time.time() - timeDiff

                else:
                    blockSupportMesh = extrudedBlock


                # Draw the support structures generated
                blockSupportMesh.visual.face_colors[:,:3] = np.random.randint(254, size=3)

                # Create a BlockSupport Object
                baseSupportBlock = BlockSupportBase(supportObject=part,
                                                    supportVolume=blockSupportMesh,
                                                    supportSurface=subregion,
                                                    intersectsPart=isIntersecting)

                supportBlockRegions.append(baseSupportBlock)

            logging.info('\t - processed support face\n')

        logging.info('Total boolean time: {:.3f}\n'.format(totalBooleanTime))

        return supportBlockRegions
