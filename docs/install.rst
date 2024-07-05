.. _installing:

Installing the library
######################

There are several ways to get the PySLM source, which lives on
`github <https://github.com/drlukeparry/pyslm>`_. There are two approaches available for the installation.


Installation directly from source
==================================

When you are working on a project in Git, you can use the pyslm github repository. It is useful for having the cloned
repository as it allows the user to see and execute the examples within the source tree. From your git repository, use:

.. code:: bash

    git clone https://github.com/drlukeparry/pyslm.git

PySLM (version >0.6) and onwards is now source only, and the ClipperLib library originally bundled with PySLM is not
required therefore internal compilation via cython is not required. Thus, this simplifies the distribution and
installation of PySLM.

Legacy versions of PySLM (version <= 0.5) require compiling directly from source. This requires compiling the code
rather than pure vanilla installation of the source. Currently the prerequisites are the cython package and a compliant
c++ build environment. In order to compile, it is necessary to have a valid c++ compiler environment depending on the
user's platform and additionally cython in order to compile the ClipperLib module. For Windows this is likely to be MSVC
compiler, Linux - GCC/Clang, Mac OS X - Clang.

The compilation of PySLM can be done using the following command

.. code:: bash

    git clone https://github.com/drlukeparry/pyslm.git
    cd ./pyslm
    python setup.py install


Installation with PyPI
========================

Installation is currently supported across all platforms on Windows, Mac OS X and Linux environments.
The pre-requisites for using PySLM can be installed via PyPi and/or Anaconda distribution. PySLM is solely a source
distribution, with all additional dependencies precompiled where possible across all main platforms. The main
dependencies are contained within the
`requirements.txt <https://github.com/drlukeparry/pyslm/blob/master/requirements.txt>`_
file, however, the predominant dependencies are those required by `Trimesh <https://github.com/mikedh/trimesh>`_ -
the library for loading and manipulating and working with popular mesh formats.

.. code:: bash

    conda install -c conda-forge shapely, Rtree, networkx, scikit-image, cython, trimesh
    conda install trimesh
    
Alternatively, the more common approach is using PyPi to install the dependencies.

Packages are now pre-compiled (inc. build test) using github actions CI for use on a variety of Platforms
(Windows 10, Ubuntu, Mac OS X). The pre-compiled packages are securely uploaded directly from github to PyPi repository
and should only be installed using *PythonSLM* package only.


.. code:: bash

    pip install shapely, Rtree, networkx, scikit-image, pyclipr
    pip install trimesh

    
You can download the precompiled binaries as a Python package from PyPI using pip using:

.. code-block:: bash

    pip install PythonSLM

.. note::
    Historically PySLM package was unavailable for use by the project. The name is secured and for user's security and
    to prevent confusion will be retained by the author.

Installation Support Module Dependencies
###################################################

The support module requires a system with a working implementation of OpenGL 2.1 (available across the majority of
platforms and hardware, with adequate driver support). There are subtleties with the implementation but this has been
successfully tested across all platforms (Windows 10, Ubuntu, Mac OS X).

Due to the technical complexity of the support module, a number of additional soft dependencies are currently
required amongst a working Python OpenGL environment. These are not required for the core functionality of
PySLM such as slicing and hatching to provide a maximise accessibility of the library.

The Python OpenGL environment can be installed with the following dependencies:

.. code-block:: bash

    pip install PyQt5, vispy

The remaining dependencies are required for the support module to function correctly:

.. code-block:: bash

    pip install triangle, manifold3d, mapbox-earcut

The `manifold <https://github.com/elalish/manifold>`_ library provides the boolean CSG operations used for intersecting
meshes between the part model and the support structures.

Installing libSLM
###################

libSLM is a c++ support library for the translation (reading and writing) of machine build files commonly used with
commercial SLM systems. Potentially the library could be extended to SLA platforms.

The library does not generate the scan vectors used by the machine, rather, merely provides an interface for
importing and exporting a collection of layers containing a number of layer geometries containing points, contours and
scan vectors. These follow the same predefined structure in 'pyslm.geometry' submodule with a few specific
exceptions depending on the translator used.

.. note::
    The library does not provide an implementation for generating low-level, specific G-codes used by systems, however,
    could potentially be implemented as a feature in the future.

Access to these specific translators for exporting to different machine platforms are currently available on request
as pre-compiled modules due to sensitivity of working with proprietary formats. The source code of these specific
translators used for commercial systems will be made available for research (non-commercial) purposes via requests
at the discretion of the author until prior notice.

Installation
===============

libSLM is a c++ library for directly interfacing with machine build files used on commercial L-PBF fusion systems.

No strict dependencies are required for compiling libSLM, originally based on the Qt library. This design decision was
taken to improve the cross-platform behaviour of the project. Python bindings are generated via
`pybind <https://pybind11.readthedocs.io/en/stable/>`_, which is automatically pulled in by as sub-module by calling
`git clone` with `--recursive`.


.. code:: bash

    git clone --recursive https://github.com/libSLM
    cmake .


Compiler Requirements
=========================

libSLM was designed to minimise the number of dependencies to improve the compatibility to integrate into existing software
- in particular linking to subroutines used in commercial FEA simulation codes. The underlying library is developed
to be compatible on both Windows and Unix systems.

**On Unix (Linux)**

* A compiler (GCC, Clang) with C++11 support
* CMake >= 3.0

**On Mac OS X (Intel, Arm64)**

* Install XCode tools to provide the LLVM compiler-chain if this is not already available.
* Ensure Cmake is installed and available via brew

.. code:: bash

    brew install cmake

**On Windows**

* Visual Studio 2015 (required for all Python versions)
* CMake >= 3.0

During the build process both dynamic and static libraries are generated and these can be statically or
dynamically linked respectively within other c++ programs.


Installation: Python Bindings - Compiling from Source
=========================================================

The Python module in libSLM can be generated using python by simply cloning this repository and then running pip install
in your python environment. Note the `--recursive` option which is needed for the `pybind11`, `eigen`, and `filesystem`
submodules:

.. code:: bash

    git clone --recursive https://github.com/libSLM

After requesting access to the libSLM translators from the author, copy the contents of the Translator directory from
the private repository and into the 'Translators' folder. Complete the compilation by calling:

.. code:: bash
    pip install ./libSLM

With the `setup.py` file included in this example, the `pip install` command will invoke CMake to build the pybind11
module as specified in `CMakeLists.txt` and generate a package. A specific version of python is not required provided
it is compatible with pybind. During the process The CMake Option flag `BUILD_PYTHON` will be automatically toggled on
during the build phase.
