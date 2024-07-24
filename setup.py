# -*- coding: utf-8 -*-

import os

from setuptools import setup, find_packages

# load __version__ without importing anything
version_file = os.path.join(
    os.path.dirname(__file__),
    'pyslm/version.py')

with open(version_file, 'r') as f:
    # use eval to get a clean string of version from file
    __version__ = eval(f.read().strip().split('=')[-1])

# minimal requirements for installing PySLM
# note that `pip` requires setuptools itself
requirements_default = set([
    'numpy',  # all data structures
    'scipy',
    'scikit-image',
    'setuptools',  # used for packaging
    'shapely',
    'cython',
    'pyclipr',
    'manifold3d',
    'Rtree',
    'networkx',
    'matplotlib',
    'trimesh'
])

# "easy" requirements should install without compiling
# anything on Windows, Linux, and Mac, for Python 2.7-3.4+
requirements_easy = set([
    'setuptools',  # do setuptools stuff
    'shapely',
    'Rtree',
    'scikit-image',
    'networkx',
    'trimesh',  # Required for meshing geometry
    'triangle',
    'colorlog'])  # log in pretty colors


requirements_supports = set([
    'setuptools',  # do setuptools stuff
    'shapely',
    'Rtree',
    'scikit-image',
    'networkx',
    'trimesh',  # Required for meshing geometry
    'triangle',
    'vispy',
    'PyQt5',
    'mapbox-earcut'
    'colorlog'])  # log in pretty colors

# requirements for building documentation
# Note API is only read from pyclipper in external project
requirements_docs = set([
    'sphinx',
    'jupyter',
    'sphinx_rtd_theme',
    'sphinx-paramlinks'
    'pypandoc',
    'pyclipr'
    'autodocsumm',
    'numpy',
    'shapely',
    'scipy',
    'scikit-image'
    'pyclipper',
    'networkx',
    'trimesh',
    'cython'])

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='PythonSLM',
    version=__version__,
    description='Python Package for Additive Manufacturing and 3D Printing Development',
    long_description_content_type='text/x-rst',
    long_description=readme,
    author='Luke Parry',
    author_email='dev@lukeparry.uk',
    url='https://github.com/drlukeparry/pyslm',
    keywords=['3D Printing', 'AM', 'Additive Manufacturing', 'Geometry', 'SLM', 'Selective Laser Melting', 'L-PBF'],
    python_requires='>=3.7',
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Natural Language :: English',
        'Topic :: Scientific/Engineering'],
    license="",
    packages=find_packages(exclude=('tests', 'docs', 'examples')),
    package_data = {'pyslm': ['../LICENSE',  '../CHANGELOG.md', '../README.rst']},
    include_package_data=False,
    install_requires=list(requirements_default),
    extras_require={'easy': list(requirements_easy),
                                'support': list(requirements_supports),
                                'docs': list(requirements_docs)},

    project_urls = {
        'Documentation': 'https://pyslm.readthedocs.io/en/latest/',
        'Source': 'https://github.com/drylukeparry/pyslm/pyslm/',
        'Tracker': 'https://github.com/drlukeparry/pyslm/issues'
    }

)