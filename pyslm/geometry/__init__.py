
try:
    from libSLM import Header, BuildStyle, Model, Layer, LayerGeometry, ContourGeometry, HatchGeometry, PointsGeometry, LaserMode

except BaseException as E:
    """
    The libSLM library is not available so instead use the fallback python equivalent in order to store the layer and 
    geometry information for use later. This removes the capability to export to machine build file format
    """
    from .geometry import Header, BuildStyle, Model, Layer, LayerGeometry, ContourGeometry, HatchGeometry, PointsGeometry, LaserMode

from .utils import *

