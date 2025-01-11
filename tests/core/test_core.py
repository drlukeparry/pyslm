import sys
import math

import pyclipr
import pytest
import pyslm
import trimesh
import numpy as np

from shapely.geometry import Polygon, MultiPolygon

class TestBasic:

    NUM_ORIENTATIONS = 36
    NUM_ORIGINS = 10
    NUM_SCALEFACTORS = 10

    def test_vector_slicing_cube(self):

        # load the mesh into a pyslm Part
        part = pyslm.Part('cube')
        part.setGeometry(trimesh.creation.box(extents=[1, 1, 1]))

        # slice the part
        polys = part.getVectorSlice(0.0, returnCoordPaths=False, fixPolygons=False)

        # check on polygon is returned
        assert len(polys) == 1

        # check the polygon is a shapely polygon
        assert isinstance(polys[0], Polygon)

        # Check the area of the polygon is correct
        assert polys[0].area - 1.0 < sys.float_info.epsilon

    def test_vector_slicing_paths(self):

        # load the mesh into a pyslm Part
        part = pyslm.Part('cube')
        part.setGeometry(trimesh.creation.box(extents=[1, 1, 1]))

        # slice the part
        paths = part.getVectorSlice(0.0, returnCoordPaths=True)

        assert len(paths) == 1

    def test_vector_slicing_path_fix(self):

        # slice the part
        part = pyslm.Part('cube')
        part.setGeometry(trimesh.creation.box(extents=[1, 1, 1]))

        polys = part.getVectorSlice(0.0, returnCoordPaths=False, fixPolygons=True)

        area = math.pow(1.0 + 2e-3,2) - float(polys[0].area)
        area_abs = math.fabs(area)
        assert area_abs < 1e-6

    @pytest.fixture
    def orientations(self):
        rng = np.random.default_rng(seed=15)
        orientations = [rng.uniform(low=-180.0,high=180.0, size=(3,1)) for _ in range(self.NUM_ORIENTATIONS)]
        return orientations

    @pytest.fixture
    def origins(self):
        rng = np.random.default_rng(seed=15)
        origins = [rng.uniform(low=-180.0, high=180.0, size=(3,1)) for _ in range(self.NUM_ORIGINS)]
        return origins

    @pytest.fixture
    def reg_log_sf(self):
        scaleFactors = np.linspace(-2,2,self.NUM_SCALEFACTORS)
        return (10**sf for sf in scaleFactors)

    @pytest.fixture()
    def random_cube_generator(self, reg_log_sf, origins, orientations):

        from itertools import product

        class Cube:
            def __init__(self):
                self.part = pyslm.Part('cube')
                self.part.setGeometry(trimesh.creation.box(extents=[1, 1, 1]))

            def __iter__(self):
                for sf, origin, orientation in product(reg_log_sf, origins, orientations):

                    self.part.scaleFactor = sf
                    self.part.origin = origin
                    self.part.rotation = orientation
                    yield self.part

        return Cube()

    def test_drop_part_cube(self, random_cube_generator):

        for i, part in enumerate(random_cube_generator):

            # drop the part
            part.dropToPlatform()
            print(i, part.rotation)
            # check the part is dropped
            assert float(np.abs(part.boundingBox[2])) < 1e-8


    def test_vector_slicing_complex_path(self):

        # slice the part
        part = pyslm.Part('cube')
        part.setGeometry('./models/frameGuide.stl')

        # slice the part
        polys = part.getVectorSlice(0.0, returnCoordPaths=False)
        assert len(polys) == 2
        
        polys = part.getVectorSlice(0.0, returnCoordPaths=False, fixPolygons=True)
        assert len(polys) == 2 # 2 polygons expected

        paths = part.getVectorSlice(0.0, returnCoordPaths=True)
        assert len(paths) == 4  # 2 exterior and 2 interior paths are expected in total

        paths = part.getVectorSlice(0.0, returnCoordPaths=True, fixPolygons=True)
        assert len(paths) == 4 # 2 exterior and 2 interior paths are expected that are merged into a list

    def test_bitmap_slicing(self):

        RESOLUTION = 1e-4
        AREAL_ERROR = RESOLUTION * 1e2

        # slice the part
        part = pyslm.Part('cube')
        part.setGeometry(trimesh.creation.box(extents=[1, 1, 1]))

        img = part.getBitmapSlice(0.0, RESOLUTION)

        area = np.sum(img.ravel()) * math.pow(RESOLUTION,2)
        area_delta = abs(float(area) - 1.0)
        assert area_delta < AREAL_ERROR


    def test_bitmap_slicing_complex(self):

        RESOLUTION = 5e-3
        AREAL_ERROR = RESOLUTION * 1e2

        # slice the part
        part = pyslm.Part('frameguide')
        part.setGeometry('./models/frameGuide.stl')

        # slice the part
        polys = part.getVectorSlice(0.0, returnCoordPaths=False)

        poly_area = sum([poly.area for poly in polys])

        img = part.getBitmapSlice(0.0, RESOLUTION)

        area = np.sum(img.ravel()) * math.pow(RESOLUTION,2)
        area_delta = abs(float(area) - poly_area)
        assert area_delta < AREAL_ERROR

    def test_check_part_attributes(self):

        # load the mesh into a pyslm Part
        part = pyslm.Part('cube')

        with pytest.raises(RuntimeError):

            # Mesh is not available so attributes should not be available
            part.boundingBox

        with pytest.raises(RuntimeError):
            # Mesh is not available so attributes should not be available
            part.extents

        with pytest.raises(RuntimeError):
            # Mesh is not available so attributes should not be available
            part.volume

        with pytest.raises(RuntimeError):
            # Mesh is not available so attributes should not be available
            part.volume

        with pytest.raises(RuntimeError):
            # Mesh is not available so attributes should not be available
            part.surfaceArea

    def test_load_trimesh_cube(self):

        # create a cube from geometry
        mesh = trimesh.creation.box(extents=[1, 1, 1])

        part = pyslm.Part('cube')

        part.setGeometry(mesh)

        assert part is not None

        bboxDiff = np.diff(part.boundingBox.reshape(2,3) - mesh.bounds, axis=0)
        bboxVolDiff = np.prod(bboxDiff)

        assert bboxVolDiff < 1e-5

        bboxExtentCalculated = np.diff(part.boundingBox.reshape(2,3), axis=0)
        bboxExtentDiff = np.prod(bboxExtentCalculated - np.array([1, 1, 1]))

        assert bboxExtentDiff < 1e-5

        bboxExtentDiff = np.prod(bboxExtentCalculated - mesh.extents)

        assert bboxExtentDiff < 1e-5