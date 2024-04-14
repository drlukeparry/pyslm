
# Change Log
All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

### Fixed

### Changed

## [0.6.0] - 2024-01-14

### Added
- Added documentation covering basic usage for the operation of PySLM
- In `pyslm.support` added a method for sweeping polygons along a path `sweepPolygon` based on Trimesh  [7ac9e4fd771fa6abc65753c21735c1592dbc9aa1](https://github.com/drlukeparry/pyslm/commit/7ac9e4fd771fa6abc65753c21735c1592dbc9aa1)
- In `pyslm.support` for `BlockSupportBase` that are connected directly to the baseplate are now smoothly created  [df2dd41e0b04160a7ed4c96f8f7c0aed71003430](https://github.com/drlukeparry/pyslm/commit/df2dd41e0b04160a7ed4c96f8f7c0aed71003430)
- In `pyslm.support` added perforated teeth to the upper and lower surfaces of `GridBlockSupport`  [70c510cce31b0cb297873252fa72c2f67b386423](https://github.com/drlukeparry/pyslm/commit/70c510cce31b0cb297873252fa72c2f67b386423)
- In `pyslm.support` added a method for checking the approximate intersection of a cylindrical strut `checkStrutCylinderIntersection`  [f2baa3383b1c01512d3a74d74af65931b14f7986](https://github.com/drlukeparry/pyslm/commit/f2baa3383b1c01512d3a74d74af65931b14f7986)
- In `pyslm.support` added regions for strengthening the support on upper and lower regions of both the skin and slices in
  `GridBlockSupport`  [0121971813e50296e3f6d9bab0beb431067443d2](https://github.com/drlukeparry/pyslm/commit/0121971813e50296e3f6d9bab0beb431067443d2)
- In `pyslm.support.GridBlockSupport` added methods for labelling generated geometry based on the interior X,Y grid and across the skin [71fb3efd1a8376809850f83760639fa331181436](https://github.com/drlukeparry/pyslm/commit/71fb3efd1a8376809850f83760639fa331181436)
- In `pyslm.support.GridBlockSupport` added methods for slicing the geometry into layers and performing sorting to provide correct orientation [4c7000b604b3f231d0160312aa2cd5a170de69f1](https://github.com/drlukeparry/pyslm/commit/4c7000b604b3f231d0160312aa2cd5a170de69f1)
- Added methods `pyslm.hatching.poly2paths` , `pyslm.hatching.paths2clipper` and `pyslm.hatching.clipper2paths` for conversion 
  between shapely polygons and pyclipr paths  [adcb371e62e45a5dbadfa46e4e5589a0f5cb28ae](https://github.com/drlukeparry/pyslm/commit/adcb371e62e45a5dbadfa46e4e5589a0f5cb28ae)
- In `pyslm.support.GridBlockSupport` the grid truss slices include additional face-attributes `order` and `type` describing 
  their order of generation across X,Y planes. This is used for the scan order when slicing and hatching these regions

### Fixed
- Fixed a bug in 'pyslm.hatching.BaseHatcher.hatch' - internal contour is offset when needed [344941fdd951152b69d81e97a957fd2709251151](https://github.com/drlukeparry/pyslm/commit/344941fdd951152b69d81e97a957fd2709251151) 
- Fixed a bug in `pyslm.support.GridBlockSupport`  - fixes for identifying top and bottom paths of the skin and fixed ordering during slicing - [7d4c9e1294f17334f97016c22062eccd3111b2a6](https://github.com/drlukeparry/pyslm/commit/7d4c9e1294f17334f97016c22062eccd3111b2a6) 
- Fixed a bug in `BlockSupportGenerator.identifySupportRegions` where supports connected to the build-plate were not self-intersected with the original mesh - [f1d9c95a5921bf6070799f05fd940cc056d852b1](https://github.com/drlukeparry/pyslm/commit/f1d9c95a5921bf6070799f05fd940cc056d852b1) 
- Fixed a bug in `pyslm.analysis.getLayerGeometryTime` where the jump distance was not correctly calculated resulting 
  in an `NaN` by dividing by zero by checking the build-style jump speed is greater than zero  [d169b30302e79c73ea37f9759feff72784dda4e6](https://github.com/drlukeparry/pyslm/commit/d169b30302e79c73ea37f9759feff72784dda4e6) 
- Fixed a bug in `pyslm.analysis.getLayerGeometryTime` where the jump delay was not added and default argument is used 
  alternatively in the calculation [3d996ef44284959d40b81918ecfbfba4ef240d0c](https://github.com/drlukeparry/pyslm/commit/3d996ef44284959d40b81918ecfbfba4ef240d0c) 
- Fixed bug in `pyslm.support.render.Canvas` not capturing the depth correctly - [114ffa9259f844549c4aa509c5e3d4b2db4ab081](https://github.com/drlukeparry/pyslm/commit/114ffa9259f844549c4aa509c5e3d4b2db4ab081)
- Invalid projection of extruded prismatic mesh in `support.GridBlockSupportGenerator.identifySupportMethods`. Generated mesh
  is not automatically processed by Trimesh prior to extrusion to prevent mismatch between vertices. [df2dd41e0b04160a7ed4c96f8f7c0aed71003430](https://github.com/drlukeparry/pyslm/commit/df2dd41e0b04160a7ed4c96f8f7c0aed71003430)
- Fixed the `pyslm.support` module to use manifold3D library instead of pycork  [df2dd41e0b04160a7ed4c96f8f7c0aed71003430](https://github.com/drlukeparry/pyslm/commit/df2dd41e0b04160a7ed4c96f8f7c0aed71003430)
- Updated to Trimesh version 4.0 [081dede9a14357fe9fb706470a5d42e698b763a6](https://github.com/drlukeparry/pyslm/commit/081dede9a14357fe9fb706470a5d42e698b763a6)
- Update to Shapely Library version 2.0 [081dede9a14357fe9fb706470a5d42e698b763a6](https://github.com/drlukeparry/pyslm/commit/081dede9a14357fe9fb706470a5d42e698b763a6)
- Fixes for IslandHatcher when processing empty regions  [f9ff55ecc2709c3514b1d14f815cb2620856be97](https://github.com/drlukeparry/pyslm/commit/f9ff55ecc2709c3514b1d14f815cb2620856be97)
- `pyslm.visualise.plot` will now plot point geometries when included [ddb7f27f6c18b6d494870e5823dbede7987e2e12](https://github.com/drlukeparry/pyslm/commit/ddb7f27f6c18b6d494870e5823dbede7987e2e12)
- In `pyslm.visaulise.plot`, the method will correctly plot `shapely.geometry.MultiPolygon` after the update to Shapely 2.0 [99c7a475e10b9c138e061f56aa28f7780f2a5eac](https://github.com/drlukeparry/pyslm/commit/99c7a475e10b9c138e061f56aa28f7780f2a5eac)
- Update imports to prevent namespace polluting
- In `pyslm.analysis` fixed bug for build-time calculation using `pyslm.geometry.PointsGeometry` [7f1e42ae6b30ee157a96303fa6e0b6e5632eabfd](https://github.com/drlukeparry/pyslm/commit/7f1e42ae6b30ee157a96303fa6e0b6e5632eabfd) 

### Changed
- Change PySLM to use manifold library instead of pycork [bad0fc0285835e998a3acdb02afa7e0ed02619ee](https://github.com/drlukeparry/pyslm/commit/bad0fc0285835e998a3acdb02afa7e0ed02619ee)
- Change PySLM to use pyclipr library [081dede9a14357fe9fb706470a5d42e698b763a6](https://github.com/drlukeparry/pyslm/commit/dda04c15b66ace3c487bc5e20acd806dda1ba89a)
- `pyslm.visualise.plot` by default only plots for a single layer [02ccc1d503580cea802d996f6d6532c2f7526c8f](https://github.com/drlukeparry/pyslm/commit/02ccc1d503580cea802d996f6d6532c2f7526c8f)
- Removal of custom pyclipper bindings and removing the requirement for compiling via cython [f434d77c8670bbb0bf5e289d9b7d2c011a9dcc92](https://github.com/drlukeparry/pyslm/commit/f434d77c8670bbb0bf5e289d9b7d2c011a9dcc92)
- Cannot currently use` `mergeMesh` in `pyslm.support.GridBlockSupport` due to change to the manifold boolean CSG library

## [0.5.0] - 2022-04-26

### Added
- Added support for Mac OS X (Monterey 12.3) [3b4103bf2e01f3e7175adcf0f1bed5a429bad361](https://github.com/drlukeparry/pyslm/commit/3b4103bf2e01f3e7175adcf0f1bed5a429bad361)
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
- Updated examples [639e042575e019d3b3d6f3a6272cb5f140f45606](https://github.com/drlukeparry/pyslm/commit/639e042575e019d3b3d6f3a6272cb5f140f45606)

### Changed
- Added soft dependency for PyQt5 in the dependencies needed for support generation [2ced797da62f9b8099c8c9f30c4ff7c6677b2b6d](https://github.com/drlukeparry/pyslm/commit/2ced797da62f9b8099c8c9f30c4ff7c6677b2b6d)

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

