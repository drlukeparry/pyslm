from typing import Any, List, Optional, Tuple, Union
import numpy as np
import trimesh.path.polygons
import shapely.geometry

from shapely.geometry import Polygon
from skimage.measure import approximate_polygon


def simplifyBoundaries(paths: List[Any], tolerance: Optional[float] = 0.5, method: Optional[str] = '') -> Any:
    """
    Simplify the boundaries of paths using Douglas-Peucker algorithm provided by the
    scikit-image library or an internal method within shapely.

    :param paths: The paths to simplify
    :param tolerance: The simplification tolerance used
    :param method: Not used
    :return: Simplified paths
    """
    if paths is None:
        return

    boundaries = []

    if isinstance(paths[0], shapely.geometry.Polygon):
        boundaries = [path.simplify(tolerance, preserve_topology=True) for path in paths]
    else:
        boundaries = [approximate_polygon(path, tolerance) for path in paths]

    return boundaries


def pathsToClosedPolygons(paths) -> List[shapely.geometry.Polygon]:
    """
    Converts closed paths to Shapely polygons with both exterior and interior boundaries. This method leverages the same
    functionality in Trimesh but in a more convenient form.

    :param paths: A list of closed paths consisting of (n x 2) coordinates
    :return: List of non-overlapping Shapely Polygons
    """

    closedPolygons = trimesh.path.polygons.paths_to_polygons(paths)

    (roots, tree) = trimesh.path.polygons.enclosure_tree(closedPolygons)

    complete = []
    for root in roots:
        interior = list(tree[root].keys())
        shell = closedPolygons[root].exterior.coords
        holes = [closedPolygons[i].exterior.coords for i in interior]
        complete.append(Polygon(shell=shell,
                                holes=holes))

    return complete


def isValidHatchArray(hatchVectors: np.ndarray) -> bool:
    """ Utility method  to check if the numpy array is a valid hatch array"""
    return hatchVectors.ndim == 2 and (hatchVectors.shape[0] % 2) == 0


def to3DHatchArray(hatchVectors: np.ndarray) -> np.ndarray:
    """
    Utility to reshape a  flat 2D hatch vector array into a 3D array to allow manipulation of individual vectors

    :param hatchVectors: Numpy Array of Hatch Coordinates of shape (2n, 2) where n is the number of of individual hatch vectors
    :return: A view of the hatch vector formatted as 3D array of shape (n,2,2)
    """
    if hatchVectors.ndim != 2:
        raise ValueError('Hatch Vector Shape should be 2D array')

    return hatchVectors.reshape(-1, 2, 2)


def from3DHatchArray(hatchVectors: np.ndarray) -> np.ndarray:
    """
    Utility to reshape a 3D hatch vector array of shape (n,2,2)  into a flat 2D array to
    allow manipulation of individual vectors.

    :param hatchVectors: Numpy Array of Hatch Coordinates of shape (n, 2, 2) where n is the number of individual hatch vectors
    :return: A view of the hatch vector formatted as 2D array of shape (2n, 2)
    """

    if hatchVectors.ndim != 3:
        raise ValueError('Hatch Vector Shape should be a 3D array')

    return hatchVectors.reshape(-1, 2)


def poly2Paths(polygons: Union[shapely.geometry.Polygon, shapely.geometry.MultiPolygon]) -> List[np.array]:
    """
    Converts a Shapely Polygon or MultiPolygon to a list of paths

    :param polygons: A polygon to convert to individual paths
    :return: A list of paths (interior and exterior) for each polygon
    """
    paths = []

    if isinstance(polygons, shapely.geometry.MultiPolygon):
        paths = [np.array(path.exterior.coords.xy).T for path in list(polygons.geoms)]
    elif isinstance(polygons, shapely.geometry.Polygon):
        pathsExterior = [np.array(polygons.exterior.coords.xy).T]
        pathsInterior = [np.array(interior.coords.xy).T for interior in polygons.interiors]

        paths = pathsExterior + pathsInterior

    else:
        raise ValueError('Type of polygons is not supported')

    return paths


def paths2clipper(paths: Any) -> List[np.array]:
    """
    Converts a list of paths to clipper format input

    :param paths: Paths for conversion
    :return: The list of coordinates for each path
    """
    return [np.hstack([path, np.arange(len(path)).reshape(-1, 1)]) for path in paths]


def clipper2Paths(paths, scaleFactor: float, close: bool = False) -> List[np.array]:
    """
    Returns scaled closed paths from clipper paths

    :param paths: The input clipper paths
    :param scaleFactor: Scale factor to apply
    :param close: If True, the paths are closed by appending the first point to the end of the path
    :return:
    """
    out = [np.array(path)[:,:2]/scaleFactor for path in paths]

    if close:
        outPaths = []
        for path in out:
            outPaths.append(np.vstack([path, path[0,:]]))
        out = outPaths

    return out

