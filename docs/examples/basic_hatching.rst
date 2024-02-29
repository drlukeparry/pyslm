PySLM Basic Hatching Guide
==============================

Introduction
---------------------------
Hatching is a vital operation of scanner based additive manufacturing processes, typically powder-bed fusion processes
such as selective laser sintering, selective laser melting, and electron beam melting. The hatching process generates the
required rasterisation pattern for infilling a geometry, typically for the position of the exposure source on a medium
such as powder. PySLM is not currently focused on generating paths for extrusion based processes such as FDM, as there
many other open source packages that already provide this functionality.

This tutorial will show you how to use the basic hatching functionality available in the
`hatch` package.  The `hatching` package is a pyslm provides the basic underlying framework for
offsetting and hatching polygonal boundaries passed that are typically obtained from a set of slicing operations on a
volumetric object (typically a triangular mesh contained in a `trimesh.Trimesh`. However, alternative means for
obtaining geometry can be used such as implicit fields for lattice volumes as shown in `examples/implicit_surface.py`.
Fundamentally, the only requirement is the generation of polygonal boundaries which form the basis
of offsetting and hatching operations.

Structure of PySLM
----------------------
In Hatching, there are a variety of classes available that demonstrate the flexibility and capability offered for generating
alternative infills produce a variety of scan strategies (meander/serpentine, island/checkerboard and stripe), although
in practice these can be extended to any infill pattern.

The core hatching functionality is included in `pyslm.hatching.BaseHatcher`, which contains the functions for manipulating
the geometry and for performing basic offsetting and clipping of scan vectors. This is built-upon external functionaltiy
offered by the shapely and pyclipr (ClipperLib2) libraries that handle polygon clipping offsetting operations.

.. note::

    The :attr:`~pyslm.hatching.BuildStyle.pointDistance` parameter must be set or this method will fail.

Basics of Slicing and Hatching
--------------------------------

The :class:`~pyslm.hatching.BaseHatcher` operates in 2D. It is provided with boundaries typically generated from a slicing operation from the
mesh geometry. These boundaries are offset to create a series of internal coordinates and the interior is infilled using
a series of hatches or potentially other infill patterns.

.. code-block:: python

    from pyslm import hatching
    from pyslm.hatching import BaseHatcher
    from pyslm.hatching import HatchStyle

    # Create a BasicIslandHatcher object for performing any hatching operations
    myHatcher = hatching.BasicHatcher()

The :class:`~pyslm.hatching.BaseHatcher` requires some parameters to be set for both the number and distance between
offsets of the boundary and those used during hatching phase, primarily the hatch distance :math:`h_d`  and the hatch angle :math:`\theta_d` provided
in degrees.

.. code-block:: python

    # Set the base hatching parameters which are generated within Hatcher
    myHatcher.hatchAngle = 10 # [°] The angle used for the islands
    myHatcher.volumeOffsetHatch = 0.08 # [mm] Offset between internal and external boundary
    myHatcher.spotCompensation = 0.06 # [mm] Additional offset to account for laser spot size
    myHatcher.numInnerContours = 2
    myHatcher.numOuterContours = 1
    myHatcher.hatchSpacing = 0.1 # [mm] The spacing between hatch lines

These parameters can be changed at any point during the hatching process across a set of boundaries (closed polygons).

The boundaries of any geometry are passed to the :class:`~pyslm.hatching.BaseHatcher` object using the
:meth:`~pyslm.hatching.BaseHatcher.hatch` method. The boundaries must be
closed connected paths that typically originate from  watertight (manifold) geometry in a mesh. These can be obtained
from a variety of means but typically are obtained from a slicing operation on a mesh that can be performed directly
in PySLM. Alternative methods doe exist for other geometries (e.g. implicit models - see `examples/implicit_surface.py`).
or those generated procedurally. Irrespective of the source, the boundaries must be closed and connected.

Using the build-in option, :class:`~pyslm.core.Part` geometry can created from an input mesh geometry supported by the trimesh library.
Basic transformation operations (e.g. rotation, translation and scaling), which can be applied to the geometry,
especially those which include dropping the part.

.. code-block:: python

    # Create a part from a mesh
    solidPart = Part()
    solidPart.setGeometry('../models/frameGuide.stl')

    # Drop the part to the build platform
    solidPart.dropToBuildPlatform()

    # Rotate the part
    solidPart.rotate([0, 0, 1], np.pi/4)

    # Scale the part
    solidPart.scale(0.5)

The slicing can be obtained from the part using the :meth:`~pyslm.core.Part.getVectorSlice` method. This method returns a set of
2D polygons:

.. code-block:: python

    # Get the slice of the part at Z height
    z = 0.1
    geomSlice = solidPart.getVectorSlice(z, simplificationFactor=0.1)


These can be provided to the hatching method accordingly. This will create suitable structures used for being processed
by most PBF systems, which consist of fundamental types defined in `pyslm.hatching`. In principle,
these will create a set of contours (:class:`pyslm.hatching.ContourGeometry`), and for the interior a series of
hatch vectors for each region (:class:`pyslm.hatching.HatchGeometry`). Separately parameters must be defined for these, but
these are specific to different L-PBF platforms.


.. code-block:: python

    # Hatch the geometry
    hatchLayer = myHatcher.hatch(geomSlice)

The hatch layer can then be visualised with functions provided in `pyslm.visualise` module using a variety
of plotting options, currently built on top of the matplotlib library. Using the :meth:`~pyslm.visualise.plot` function, the order of scanning
for the hatch region can be displayed by setting the parameter (`plotOrderLine=True`) and arrows can be enabled by
setting the parameter `plotArrows` to `True`. Alternatively, plots can be showed sequentially in time using the
:meth:`~pyslm.visualise.plotSequential` function, which can include the jumps between scan vectors by setting the parameter
`plotJumps` to `True`.

.. code-block:: python

    pyslm.visualise.plot(layer, plot3D=False, plotOrderLine=True, plotArrows=False)
    pyslm.visualise.plotSequential(layer, plotJumps=True)
