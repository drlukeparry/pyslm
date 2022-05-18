PySLM Python Library for Selective Laser Melting and Additive Manufacturing
=============================================================================

.. https://github.com/drlukeparry/pyslm/raw/dev/docs/images/pyslm.png

.. image:: https://github.com/drlukeparry/pyslm/raw/dev/docs/images/pyslm.png
    :alt:  PySLM - Library for  Additive Manufacturing and 3D Printing including Selective Laser Melting

.. image:: https://github.com/drlukeparry/pyslm/actions/workflows/pythonpublish.yml/badge.svg
    :target: https://github.com/drlukeparry/pyslm/actions
.. image:: https://readthedocs.org/projects/pyslm/badge/?version=latest
    :target: https://pyslm.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
.. image:: https://badge.fury.io/py/PythonSLM.svg
    :target: https://badge.fury.io/py/PythonSLM
.. image:: https://badges.gitter.im/pyslm/community.svg
    :target: https://gitter.im/pyslm/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge
    :alt: Chat on Gitter
.. image:: https://static.pepy.tech/personalized-badge/pythonslm?period=total&units=international_system&left_color=black&right_color=orange&left_text=Downloads
 :target: https://pepy.tech/project/pythonslm


PySLM is a Python library for supporting development of input files used in Additive Manufacturing or 3D Printing,
in particular Selective Laser Melting (SLM), Direct Metal Laser Sintering (DMLS) platforms typically used in both
academia and industry. The core capabilities aim to include slicing, hatching and support generation and providing
an interface to the binary build file formats available for platforms. The library is built of core classes which
may provide the basic functionality to generate the scan vectors used on systems and also be used as building blocks
to prototype and develop new algorithms.

This library provides design tools for use in Additive Manufacturing including the slicing, hatching, support generation
and related analysis tools (e.g. overhang analysis, build-time estimation).

PySLM is built-upon python libraries `Trimesh <https://github.com/mikedh/trimesh>`_ and based on some custom modifications
to the `PyClipper <https://pypi.org/project/pyclipper/>`_ libraries, which are leveraged to provide the slicing and
manipulation of polygons, such as offsetting and clipping of lines.

The aims of this library is to provide a useful set of tools for prototyping novel pre-processing approaches to aid
research and development of Additive Manufacturing processes, amongst an academic environment. The tools aim to compliment
experimental and analytical studies that can enrich scientific understanding of the process. This includes data-fusion
from expeirments and sensors within the process but also enahcning the capability of the process by providing greater control
over the process. Furthermore, the open nature of the library intends to inform and educate those interested in the
underlying algorithms of preparing toolpaths in Additive Manufacturing.

Current Features
******************

PySLM is building up a core feature set aiming to provide the basic blocks for primarily generating the scan paths and
additional design features used for AM and 3D printing systems typically (SLM/SLS/SLA) systems which consolidate material
using a single/multi point exposure by generating a series of scan vectors in a region.

**Slicing:**

* Slicing of triangular meshes supported via the `Trimesh <https://github.com/mikedh/trimesh>`_ library.
* Simplification of 2D layer boundaries
* Bitmap slicing for SLA, DLP, Inkjet Systems

**Hatching:**
The following operations are provided as a convenience to aid the development of novel scan strategies in Selective
Laser Melting:

* Offsetting of contours and boundaries
* Trimming of lines and hatch vectors (sequentially ordered and sorted)

The following scan strategies have been implemented as reference for AM platforms:

* Standard 'alternating' hatching
* Stripe scan strategy
* Island or checkerboard scan strategy

**Support Structure Generation**

PySLM provides underlying tools and a framework for identifying and generating support structures suitable for SLM.
Tools are provided identifying overhang areas based on their mesh and connectivity information, but also
usingprojection based method. The projection method takes advantage of GPU GLSL shaders for providing an efficient
raytracing approach. Using the `PyCork <https://github.com/drlukeparry/pycork>`_ boolean CSG library, an algorithm for
extracting precise defintion of volumetric support regions. These regions are segmented based on self-intersections with
the mesh. From these volumes, porous grid-truss structure suitable for SLM based process can be generated.

.. image:: https://github.com/drlukeparry/pyslm/raw/dev/docs/images/pyslmSupportStructures.png
    :alt: The tools available in PySLM for locating overhang regions and support regions for 3D Printing and
          generating volumetric block supports alongside grid-truss based support structures suitable for SLM.
    :width: 80%
    :align: center

* Extracting overhang surfaces from meshes with optional connectivity information
* Projection based block and truss support structure generation
    * 3D intersected support volumes are generated from overhang regions using OpenGL ray-tracing approach
    * Generate a truss grid using support volumes suitable for Metal AM processes
    * Exact support volume generation using the `pycork <https://github.com/drlukeparry/pycork>`_ library

**Visualisation:**

The laser scan vectors can be visualised using ``Matplotlib``. The order of the scan vectors can be shown to aid
development of the scan strategies, but additional information such length, laser parameter information associated
with each scan vector can be shown.

.. image:: https://github.com/drlukeparry/pyslm/raw/dev/docs/images/pyslmVisualisationTools.png
    :alt: The tools available in PySLM for visualising analyisng collections of scan vectors used in SLM.
    :width: 80%
    :align: center

* Scan vector plots (including underlying BuildStyle information and properties)
* Exposure point visualisation
* Exposure (effective heat) map generation
* Overhang visualisation

**Analysis:**

* Build time estimation tools
    * Based on scan strategy and geometry
    * Time estimation based on LayerGeometry
* Iterators (Scan Vector and Exposure Points) useful for simulation studies

**Export to Machine Files:**

Currently the capability to enable translation to commercial machine build platforms is being providing through a
supporting library called `libSLM <https://github.com/drlukeparry/libSLM>`_ . This is a c++ library to enable efficient
import and export across various commercial machine build files. With support from individuals the following machine
build file formats have been developed.

* Renishaw MTT (**.mtt**),
* DMG Mori Realizer (**.rea**),
* EOS SLI formats (**.sli**)
* SLM Solutions (**.slm**).

If you would like to support implementing a custom format, please raise a `request <https://github.com/drlukeparry/pyslm/issues>`_.
For further information, see the latest `release notes <https://github.com/drlukeparry/pyslm/blob/dev/CHANGELOG.md>`_.

Installation
*************
Installation is currently supported on Windows, Mac OS X and Linux environments. The pre-requisites for using PySLM can be installed
via PyPi and/or Anaconda distribution.

.. code:: bash

    conda install -c conda-forge shapely, Rtree, networkx, scikit-image, cython
    conda install trimesh

Installation of PySLM can then be performed using pre-built python packages using the PyPi repository. Additionally to
interface with commercial systems, the user can choose to install libSLM. Note, the user should contact the author to
request machine build file translators, as this cannot be installed currently without having the machine build file
translators available.

.. code:: bash

    pip install PythonSLM

Alternatively, PySLM may be compiled directly from source. Currently the prerequisites are the cython package and a compliant c++
build environment.

.. code:: bash

    git clone https://github.com/drlukeparry/pyslm.git && cd ./pyslm
    python setup.py install

Usage
******
A basic example below, shows how relatively straightforward it is to generate a single layer from a STL mesh which generates
a the hatch infill using a Stripe Scan Strategy typically employed on some commercial systems to limit the maximum scan vector
length generated in a region.

.. code:: python

    import pyslm
    import pyslm.visualise
    from pyslm import hatching as hatching

    # Imports the part and sets the geometry to an STL file (frameGuide.stl)
    solidPart = pyslm.Part('myFrameGuide')
    solidPart.setGeometry('../models/frameGuide.stl')

    # Set te slice layer position
    z = 23.

    # Create a StripeHatcher object for performing any hatching operations
    myHatcher = hatching.StripeHatcher()
    myHatcher.stripeWidth = 5.0

    # Set the base hatching parameters which are generated within Hatcher
    myHatcher.hatchAngle = 10 # [°]
    myHatcher.volumeOffsetHatch = 0.08 # [mm]
    myHatcher.spotCompensation = 0.06 # [mm]
    myHatcher.numInnerContours = 2
    myHatcher.numOuterContours = 1

    # Slice the object at Z and get the boundaries
    geomSlice = solidPart.getVectorSlice(z)

    # Perform the hatching operations
    layer = myHatcher.hatch(geomSlice)

    # Plot the layer geometries generated
    pyslm.visualise.plot(layer, plot3D=False, plotOrderLine=True) # plotArrows=True)

The result of the script output is shown here

.. image:: https://github.com/drlukeparry/pyslm/raw/dev/docs/images/stripe_scan_strategy_example.png
   :width: 50%
   :align: center
   :alt:  PySLM - Illustration of a Stripe Scan Strategy employed in 3D printing

For further guidance please look at documented examples are provided in `examples <https://github.com/drlukeparry/pyslm/tree/master/examples>`_ .
