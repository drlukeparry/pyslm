
# Change Log
All notable changes to this project will be documented in this file.

## [Unreleased]

## [0.5.0] - 2022-03-04

- Added the Support Module: `pyslm.support` [9ee1f6668b56e76137c3dade178dc837460fc056](https://github.com/drlukeparry/pyslm/commit/9ee1f6668b56e76137c3dade178dc837460fc056) [0f6e29815a5d6a65789d27c49131d49486415d31](https://github.com/drlukeparry/pyslm/commit/0f6e29815a5d6a65789d27c49131d49486415d31)
    - Added generic abstract class `support.SupportStructure` for providing the generic infrastructure for defining support structures (e.g. block, grid, tree) and common methods
    - Introduced Various Classes - `BaseSupportGenerator` and `BlockSupportGenerator` for Generating BlockSupports
    - Introduced `support.geometry.extrudeFace` method for extruding block faces with variable height
    - Introduced method `support.geometry.triangualtePolygonFromPaths` to triangulate polygons generated in the Support Module
    - Introduced a method `support.render.projectHeightMap` for projecting height/depth maps using OpenGL to calculate the depth map for projecting supports
    - Introduced classes `GridBlockSupport`, `GridBlockSupportGenerator` and associating mapping function for generating polygon truss grid supports
    - Introduced classes for identifying and generating conforming support volume
    - Self-intersecting support volumes are generated via the [pycork](https://github.com/drlukeparry/pycork) library [e438bef493553b18dd1ee5031a226d04b63a5c67](https://github.com/drlukeparry/pyslm/commit/e438bef493553b18dd1ee5031a226d04b63a5c67)
- Added point delay and jump speed in `pyslm.geometry` and these are used in calculations performed in `pyslm.analysis` [55cc7e79c9e6f1f5f8b25d38dd6a48639b422fd2](https://github.com/drlukeparry/pyslm/commit/55cc7e79c9e6f1f5f8b25d38dd6a48639b422fd2)

### Fixed
- Fix `pyslm.analysis` tools to use jump speed and jump delay parameters [3bce788b5362062755e0fb7f17757b4174d1e877](https://github.com/drlukeparry/pyslm/commit/3bce788b5362062755e0fb7f17757b4174d1e877)
- Fixed `plotSequential` showing an invalid scan vector at the end of the contour scanning [c46ae70980ac652d30ce7e3a2aa6667246752744](https://github.com/drlukeparry/pyslm/commit/c46ae70980ac652d30ce7e3a2aa6667246752744)
- Fixed the offsetting applied to contour/border scans. An overall offset is not applied if a contour scan is not used [40fb789164aa76e25222e1c822e3462eb1c6fd82](https://github.com/drlukeparry/pyslm/commit/40fb789164aa76e25222e1c822e3462eb1c6fd82)
- Updated compatability for Python 3.9 [01f79a6dcfb9625d05e3d12a5b1901d32324c7ca](https://github.com/drlukeparry/pyslm/commit/01f79a6dcfb9625d05e3d12a5b1901d32324c7ca)

#### Changed

- Added a requirement for PyQt5 in the dependencies needed for support generation [2ced797da62f9b8099c8c9f30c4ff7c6677b2b6d](https://github.com/drlukeparry/pyslm/commit/2ced797da62f9b8099c8c9f30c4ff7c6677b2b6d)

## [0.4.0] - 2021-07-23

### Added
- Added [example_laser_iterator.py](https://github.com/drlukeparry/pyslm/blob/master/examples/example_laser_iterator.py) for demonstrating the basic use of the iterator class  [0f26f4a4aa33d80769d9713157e8b675cb48a862](https://github.com/drlukeparry/pyslm/commit/0f26f4a4aa33d80769d9713157e8b675cb48a862)
- Added [example_parametric_study.py](https://github.com/drlukeparry/pyslm/blob/master/examples/example_parametric_study.py) for showing how create a design of experiment study [54dfca913b23ad71b025f5eec646f5f896b605b8](https://github.com/drlukeparry/pyslm/commit/54dfca913b23ad71b025f5eec646f5f896b605b8)
- Added `fixGeometry` option added to repair polygons generated following slicing
- Added various modes for polygon line simplification - including absolute and 'line' based on the mean edge length in the polygon. [4e6bf23cccb4ac091d44381649a11288ce495033](https://github.com/drlukeparry/pyslm/commit/4e6bf23cccb4ac091d44381649a11288ce495033)
- Added a simplification tolerance based on relative or absolute tolerance for contours [078cc554bba4c22ce1a36f682b401fb72d59f335](https://github.com/drlukeparry/pyslm/commit/078cc554bba4c22ce1a36f682b401fb72d59f335)
- Added a method `visualise.plotSequential` to plot `LayerGeometry` in sequential order with jump vectors  [4a0cd56c21e35ee09ec8afdc408471e8211d057d](https://github.com/drlukeparry/pyslm/commit/4a0cd56c21e35ee09ec8afdc408471e8211d057d)
- Added an option to modify order of scanning for hatch and contour scan vectors using `Hatching.scanContourFirst` property. [1120ee5b041dd0cdc9eb35f507d4ba13aa9ff02e](https://github.com/drlukeparry/pyslm/commit/1120ee5b041dd0cdc9eb35f507d4ba13aa9ff02e)
- Added a sort method - `UnidirectionalSort` class, that simply provides a pass through method to not modify scan vectors [0f2304f5f58612ca619e3f94276d38b6d85fe5e8](https://github.com/drlukeparry/pyslm/commit/0f2304f5f58612ca619e3f94276d38b6d85fe5e8)
- Added a sort method - `FlipSort` class, that flips the scan vectors [6d4c77984a25d80e9c4061860416fd75804a476f](https://github.com/drlukeparry/pyslm/commit/6d4c77984a25d80e9c4061860416fd75804a476f)
- Added an Iterator Module (`analysis/iterator.py`) for processing through the LayerScan Geometry. Various supporting classes are used for efficiently parsing across the Layer and associated Layer Geometries. This includes several classes including
    - (`Iterator` - base class
    - `LayerGeometryIterator` - Iterate across `LayerGeometry`s
    - `ScanVectorIterator` - Iterate across individual Scan Vectors
    - `ScanIterator`- Incremental position at a fixed time
- Added property `Part.extents`, `Part.getProjectedHull` and `Part.getProjectedArea` [4d8747fa3083c2005df8ddf6817db2f4102b84f2](https://github.com/drlukeparry/pyslm/commit/4d8747fa3083c2005df8ddf6817db2f4102b84f2)
- Method for visualising convex polygons from `Shapely` or `ClipperLib`, using Matplotlib.patches. This cannot be visualise polygons with holes - these are treated as boundaries [1bbf60298c634ea918d354f45c979aff3f4bedcc](https://github.com/drlukeparry/pyslm/commit/1bbf60298c634ea918d354f45c979aff3f4bedcc)

### Changed
- Contour scan vectors are scanned following hatch vectors by default [1120ee5b041dd0cdc9eb35f507d4ba13aa9ff02e](https://github.com/drlukeparry/pyslm/commit/1120ee5b041dd0cdc9eb35f507d4ba13aa9ff02e)

### Fixed
- `ModelValidator` uses LayerGeometry's Model Id is used for finding the associated `Model` [76587c58b7240822ea3b6314a404137af3342509](https://github.com/drlukeparry/pyslm/commit/76587c58b7240822ea3b6314a404137af3342509)
- Contour offset is correctly generated [8b37f5a37520b5abbace9f24e629826a3326e8bb](https://github.com/drlukeparry/pyslm/commit/8b37f5a37520b5abbace9f24e629826a3326e8bb)
- (BUG FIX) Final scan vector is correctly flipped [e935217f13dceda55f1f514f7c8cbdac852024df](https://github.com/drlukeparry/pyslm/commit/e935217f13dceda55f1f514f7c8cbdac852024df)
- (BUG FIX) Remove debugging messages during Layer Generation [fe9c31dacc8ce95d09999acd7ef83db4ff70669f](https://github.com/drlukeparry/pyslm/commit/fe9c31dacc8ce95d09999acd7ef83db4ff70669f)

## [0.3.0] - 2021-02-20

### Added
- Added BaseHatcher.boundaryBoundingBox() to obtain the bounding box of a collection of polygons - returned internally from PyClipper
- Added a `simplifyBoundaries()` in [hatching/utils.py](https://github.com/drlukeparry/pyslm/blob/master/pyslm/hatching/utils.py) to simplify polygons (shapely and raw coordinate boundaries using scikit image)
-` hatching.generateExposurePoints()` now generates for `ContourGeometry`
- Added `ModelValidator` class in `pyslm.geometry.utils` to verify the input of build files generated prior to exporting in libSLM - [e75b486c090b4ead712d2ddb950577e058c419e6](https://github.com/drlukeparry/pyslm/commit/e75b486c090b4ead712d2ddb950577e058c419e6)
- Added [example_exporting_multilayer.py](https://github.com/drlukeparry/pyslm/blob/master/examples/example_exporting_multilayer.py) showing how to export a multi-layer build using libSLM - [52090085fd52336e2cc2181ff886a8aebbdca1ef](https://github.com/drlukeparry/pyslm/commit/52090085fd52336e2cc2181ff886a8aebbdca1ef)
- Added [example_custom_island_hatcher.py](https://github.com/drlukeparry/pyslm/blob/master/examples/example_custom_island_hatcher.py) showing a method to create customised island scan
- Added a `HexagonIsland` Class to demonstrate custom implementation of island regions
- Added [example_build_time_analysis.py](https://github.com/drlukeparry/pyslm/blob/master/examples/example_build_time_analysis.py) to show the processes of estimating build-time
- Added [example_custom_sinusoidal_hatching.py](https://github.com/drlukeparry/pyslm/blob/master/examples/example_custom_sinusoidal_hatching.py) for showing custom hatch-infills - [c7c1a4304dd4f2a4cdf0286385ccb68d3968ba5e](https://github.com/drlukeparry/pyslm/commit/c7c1a4304dd4f2a4cdf0286385ccb68d3968ba5e)
- Added a method `BaseHatcher.clipContourLines` for clipping open scan paths to fill a region
- Added an analysis method utility `getBuildStyleById` to find the `BuildStyle` given a model id and build style id
- Added a method in plotLayer to visualise the scan vector properties (e.g. length)
- Added properties for geometry class to be compatible with [libSLM](https://github.com/drlukeparry/libSLM) 0.2.2 - providing multi-laser compatibility
- Added the method `Part.getTrimeshSlice` to get a `trimesh.Path2D` slice from the geometry - [bb2ebb9c4514a05cc1728c810deef7fc6c3239e4](https://github.com/drlukeparry/pyslm/commit/bb2ebb9c4514a05cc1728c810deef7fc6c3239e4)
- Added a method to find the 'inverse' projection of support faces
- Added `visualise.visualiseOverhang` for showing overhang regions
- Added `geometry.utils.ModelValidator` for validating the build inputs (layers, models) when exporting to a machine build file
- Added a `.gitignore` file - [498d9116dd9d91698695669d5d1309a7941e0dd9](https://github.com/drlukeparry/pyslm/commit/498d9116dd9d91698695669d5d1309a7941e0dd9)

### Changed
- Internally generateHatching() and hatch() in subclasses of `BaseHatcher` to generate the internal hatch geometry to use multiple boundaries
to ensure that the subregion generation sorting covers the global region.
- Internally `BaseHatcher.boundaryBoundingBox()` is called instead of `BaseHatcher.polygonBoundingBox()`
- Removed the for loop which previously iterate across boundaries.
- Updated IslandHatcher to use this behaviour
- Updated BaseHatcher to use static members where possible
- Analysis method `analysis.getLayerTime` requires a `Model` list
- Analysis methods use point exposure time and distance using `analysis.getLayerTime`
- Removed debug messages when visualising layers
- Fixed import of submodules in PySLM - [66f48fd9929d244b836583f24c087602cdc31a96](https://github.com/drlukeparry/pyslm/commit/66f48fd9929d244b836583f24c087602cdc31a96)

### Fixed
- Fixed `LaserMode`, `LaserType` Enums to be compatible with libSLM - [52090085fd52336e2cc2181ff886a8aebbdca1ef](https://github.com/drlukeparry/pyslm/commit/52090085fd52336e2cc2181ff886a8aebbdca1ef)
- Jump distance between `LayerGeometry` is accounted for in the [Analysis Submodule](https://github.com/drlukeparry/pyslm/blob/master/pyslm/analysis)
- Fixed visualisation of PointGeometry exposures - [135bed81bb57b867f2499311bacd7ad2d7aa67d9](https://github.com/drlukeparry/pyslm/commit/135bed81bb57b867f2499311bacd7ad2d7aa67d9)
- Updated Documentation across the project

## [0.2.1] - 2020-06-19

### Fixed
- Fixed the setup.py source tarball to include the PyClipper extensions.

## [0.2.0] - 2020-06-17

Development branch of PySLM with new features.

### Added
- Added transformations to the Part class so that these can be translated, rotated, scaled efficiently on demand
    - Updates on the geometry are cached when geometry method is called.
    - Method included to drop the part to the platform
- Added an enhanced version and higher performance of generated clippable island regions.
    - Introduced `InnerHatchRegion` in the hatching submodule
    - Introduce `IslandHatcher` which re-implements the Island/Checkerboard Scan Strategy
- Added a method to alternate adjacent scan vectors in the hatching module
- Added a method to `generateExposurePoint` method in the hatching submodule to create exposure points from scan vectors
- Added a Geometry Submodule in [pyslm.geometry](https://github.com/drlukeparry/pyslm/blob/master/pyslm/geometry)
    - libSLM Python Extension Library is used if available
    - A set of native compatible Python classes are available if libSLM is not available
    - Classes include:
        - `Layer`
        - `LayerGeometry`
        - `Model`
        - `BuildStyle`
        - `Header`
        - `HatchGeometry`, `ContourGeometry`, `PointsGeometry`
- Added an Analysis Module in [pyslm.analysis](https://github.com/drlukeparry/pyslm/blob/master/pyslm/analysis):
    - Total length of all the scan vectors across a layer
    - Total jump length between scan vectors across all vectors
    - Total scan time across a layer
- Added a method to generate bitmap slices from a single image is included in Part
- Added a Visualise Submodule in [pyslm.visualise](https://github.com/drlukeparry/pyslm/blob/master/pyslm/visualise.py)
    - Introduced a method to visualise a collection of shapely polygons
    - Introduced a method to generate a heatmap based on a set of exposure points in a layer
    - Introduced an updated method to plot layerGeometry
        - Plot scan order
        - Plot arrow direction
        - Plot the layers correctly in 3D
- Introduced several new examples:
    - 3D slicing example - [example_3d.py](https://github.com/drlukeparry/pyslm/blob/master/examples/example_3d.py)
    - Multi-threading example - [example_3d_multithread.py](https://github.com/drlukeparry/pyslm/blob/master/examples/example_3d_multithread.py)
    - Island hatching example using new implementation - [example_island_hatcher.py](https://github.com/drlukeparry/pyslm/blob/master/examples/example_island_hatcher.py)
    - Bitmap slicing of parts - [example_bitmap_slice.py](https://github.com/drlukeparry/pyslm/blob/master/examples/example_bitmap_slice.py)
    - A heat-map/exposure map visualisation - [example_heatmap.py](https://github.com/drlukeparry/pyslm/blob/master/examples/example_heatmap.py)


### Changed
- `Part.getVectorSlice` method by default returns a list of coord paths
- `Part.getVectorSlice` now returns a list of `Shapely.geometry.Polygon` if optional argument is passed
- `hatching.IslandHatcher` in the previous release is changed to `BasicIslandHatcher`

### Fixed
- Further changes and improvements to the overall documentation.
- Updated requirements.txt to ensure documentation can correctly build on readthedocs

## [0.1.0] - 2020-05-08

  The first release of PySLM in its distributed packaged form via PyPi. This release includes basic slicing and
  hatching using a custom version of PyClipper built internally using setuptools.

