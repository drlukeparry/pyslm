
# Change Log
All notable changes to this project will be documented in this file.
  
## [Unreleased]

### Added
- Added BaseHatcher.boundaryBoundingBox() to obtain the bounding box of a collection of polygons - returned internally from PyClipper
- Added a simplifyBoundaries() in hatching/utils.py to simplify polygons (shapely and raw coordinate boundaries using scikit image)
- hatching.generateExposurePoints() now generates for ContourGeometry

### Changed
- Internally generateHatching() and hatch() in subclasses of BaseHatcher to generate the internal hatch geometry to use multiple boundaries
to ensure that the subregion generation sorting covers the global region. 
    - Internally BaseHatcher.boundaryBoundingBox() is called instead of BaseHatcher.polygonBoundingBox()
    - Removed the for loop which previously iterate across boundaries.
    - Updated IslandHatcher to use this behaviour 

### Fixed
- Jump distance between LayerGeometry is accounted for in analysis submodule

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
    - Introduced InnerHatchRegion in the hatching submodule
    - Introduce IslandHatcher which re-implements the Island/Checkerboard Scan Strategy
- Added a method to alternate adjacent scan vectors in the hatching module
- Added a method to generateExposurePoint method in the hatching submodule to create exposure points from scan vectors
- Added a geometry submodule
    - libSLM Python Extension Library is used if available
    - A set of native compatible Python classes are available if libSLM is not available
    - Classes include: 
        - Layer
        - LayerGeometry
        - Model
        - BuildStyle
        - Header
        - HatchGeometry, ContourGeometry, PointsGeometry
- Added an analysis module:
    - Total length of all the scan vectors across a layer
    - Total scan time across a layer
- Added a method to generate bitmap slices from a single image is included in Part
- Added a visualise submodule
    - Introduced a method to visualise a collection of shapely polygons
    - Introduced a method to generate a heatmap based on a set of exposure points in a layer
    - Introduced an updated method to plot layerGeometry
        - Plot scan order
        - Plot arrow direction
        - Plot the layers correctly in 3D
- Introduced several new examples:
    - 3D slicing example - example_3d.py
    - Multi-threading example - example_3d_multithread.py
    - Island hatching example using new implementation - example_island_hatcher.py
    - Bitmap slicing of parts - example_bitmap_slice.py
    - A heat-map/exposure map visualisation - example_island_hatcher.py
    
 
### Changed
- getVectorSlice method by default returns a list of coord paths
- getVectorSlice now returns a list of Shapely.geometry.Polygon if optional argument is passed
- IslandHatcher in the previous release is changed to BasicIslandHatcher
 
### Fixed
- Further changes and improvements to the overall documentation. 
- Updated requirements.txt to ensure documentation can correctly build on readthedocs
## [0.1.0] - 2020-05-08
  
  The first release of PySLM in its distributed packaged form via PyPi. This release includes basic slicing and 
  hatching using a custom version of PyClipper built internally using setuptools.
 