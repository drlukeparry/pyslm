.. _installing:

Installing the library
######################

There are several ways to get the PySLM source, which lives at
`github <https://github.com/drlukeparry/pyslm>`_. There are two approaches available for the installation.


Installation directly from source
==================================

When you are working on a project in Git, you can use the pyslm github repository. It is useful for having the cloned
repository as it allows the user to see and execute the examples within the source tree. From your git repository, use:

.. code:: bash

    git clone https://github.com/drlukeparry/pyslm.git

PySLM may be compiled directly from source. This requires compiling the code rather than pure vanilla nstallation of the
source. Currently the prerequisites are the cython package and a compliant c++
build environment.  In order to compile, it is necessary to have a valid c++ compiler environment depending on the user's
platforn and additionally cython in order to compile the ClipperLib module. For Windows this is likely to be MSVC
compiler, Linux - GCC/Clang.

.. code:: bash

    git clone https://github.com/drlukeparry/pyslm.git
    cd ./pyslm
    python setup.py install
    

    
Installation with PyPI
========================

Installation is currently supported on Windows, Mac OS X and Linux environments. The pre-requisites for using PySLM
can be installed via PyPi and/or Anaconda distribution. The main dependencies are contained within the
`requirements.txt <https://github.com/drlukeparry/pyslm/blob/master/requirements.txt>`_
file, however, the predominant dependencies are those required by `Trimesh <https://github.com/mikedh/trimesh>`_ -
the library for loading,and manipulating and working with popular mesh formats.

.. code:: bash

    conda install -c conda-forge shapely, Rtree, networkx, scikit-image, cython
    conda install trimesh
    
or using PyPi

Packages are now pre-compiled (inc. build test) using github actions CI for use on a variety of Platforms
(Win10, Ubuntu, Mac OSX). The pre-compiled packages are securely uploaded directly from github to PyPi repository and
should only be installed using *PythonSLM* package only.


.. code:: bash

    pip install shapely, Rtree, networkx, scikit-image, cython
    pip install trimesh
    
    
You can download the binaries as a Python package from PyPI
using Pip. Just use:

.. code-block:: bash

    pip install PythonSLM

.. note::
    Historically PySLM package was unavailable for use by the project. The name is secured and for user's security and
    to prevent confusion will be retained by the author.
