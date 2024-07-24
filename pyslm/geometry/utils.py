from typing import Dict,Iterable, List, Optional, Tuple, Union
from warnings import warn

import trimesh.transformations
import numpy as np

from . import Layer, LayerGeometry, HatchGeometry, ContourGeometry, PointsGeometry, BuildStyle, Model

def createLayerDict(layerList: List[Layer]) -> Dict[int, Layer]:
    """
    Create a dict from a list of Layers with a key (LayerId) and the corresponding :class:`Layer` as a value.

    :param layerList: A list of layers with a unique LayerId
    :return: A Dict Structure with Layers
    """
    return {layer.layerId: layer for layer in layerList }


def mergeLayers(layerLists: List[Dict[int, Layer]]) -> Dict[int, Layer]:
    """
    Merges a list of Layers - typically from separate parts into a unified layer. The layer geometries are merged
    together into a new Dict structure, with the order :class:`LayerGeometry`

    :param layerLists: A list of Dictionary consisting of layers.
    :return:
    """
    mergedLayers = dict()

    # Iterate across each list of layers [typically a slice part]
    for layerSet in layerLists:
        for layer in layerSet.values():
            fndLayer = mergedLayers.get(layer.layerId)
            if fndLayer:
                fndLayer.geometry.extend(layer.geometry)
            else:
                mergedLayers[layer.layerId] = layer

    return mergedLayers

def translateLayerGeoms(layerGeoms: Union[LayerGeometry, List[LayerGeometry]],
                        translation: np.ndarray) -> None:
    """
    Apply a translation in-situ to :class:`LayerGeometry` objects

    :param layerGeoms: The Layer Geometries to transform
    :param translation: The translation vector to apply
    """

    M = trimesh.transformations.translation_matrix(translation)
    transformLayerGeoms(layerGeoms, M)


def transformLayerGeoms(layerGeoms: Union[LayerGeometry, List[LayerGeometry]],
                        transform: np.ndarray) -> None:
    """
    Apply a transformation in-situ to :class:`LayerGeometry` objects

    .. note::

        The function applies affine transformation matrix first and subsequently followed by translation

    :param layerGeoms: The Layer Geometries to transform
    :param transform: A (2x2 or 3x3) transformation matrix applied to the coordinates in each :class:`LayerGeometry`
    """

    if not isinstance(layerGeoms, Iterable):
        layerGeoms = [layerGeoms]

    if not(transform.shape == (3,3) or transform.shape == (2,2)):
        raise ValueError('Transformation matrix should be 2x2 or 3x3')

    # Extract the affine transformation (excluding the translation)
    M = transform[0:2, 0:2]

    # Extract the translation vector
    if transform.shape == (3,3):
        T = transform[0:2, 2]
    else:
        T = np.array([0.,0.0])

    for geom in layerGeoms:
        geom.coords = M.dot(geom.coords.T).T + T.reshape(1,2)


def getBuildStyleById(models: List[Model], mid: int, bid: int) -> Union[BuildStyle, None]:
    """
    Returns the :class:`BuildStyle` found from a list of :class:`Model` given a model id and buildstyle
    :attr:`BuildStyle.bid`.

    :param models: A list of models
    :param mid: The selected model id
    :param bid: The selected `BuildStyle` id

    :return: The :class:`BuildStyle` if found or `None`
    """
    model = next(x for x in models if x.mid == mid)

    if model:
        bstyle = next(x for x in model.buildStyles if x.bid == bid)

        return bstyle

    return None

def getLayerById(layers: List[Layer], layerId: int) -> Layer:
    """
    Convenience function that locates the :class:`Layer` within a list given a provided layer id by searching across
    :attr:`Layer.layerId`

    :param layers: The list of layers to search
    :param layerId: The layer id to find

    :return: If found the Layer or ``None``
    """

    layer = next(x for x in layers if x.layerId == layerId)

    return layer


def getModel(models: List[Model], mid: int) -> Union[BuildStyle, None]:
    """
    Returns the Model found from a list of :class:`Model` given a model id and build id.

    :param models: A list of models
    :param mid: The selected model id

    :return: The BuildStyle if found or ``None``
    """
    model = next(x for x in models if x.mid == mid)

    return  model if model else None


class ModelValidator:
    """
    ModelValidator takes the  data structures in `pyslm.geometry` such as a list of :class:`Layer` and :class:`Model`
    and validates their input for consistency when utilised together to form a machine build file prior to exporting
    using `libSLM <https://github.com/drlukeparry/libSLM>`_. Basic consistency checks include:

    * Validating each :class:`BuildStyle` used in each :class:`Model`, including individual laser parameters
    * References to a correct :class:`BuildStyle` via its (:attr:`~BuildStyle.bid`) for each :class:`LayerGeometry` included
    * References to a correct :class:`Model` via its (:attr:`~BuildStyle.mid`) for each :class:`LayerGeometry` included
    * Ensure there are unique :class:`BuildStyle` entries for each :class:`Model` included

    The key function that can be called is :meth:`validateBuild`, which is recommened to be called before attempting to
    export the layer and model information to a libSLM machine build file translator. Additional sub-functions are also
    available for checking specific objects used to construct the build-file.
    """

    @staticmethod
    def _buildStyleIndex(models: List[Model]):

        index = dict()
        for model in models:
            for bstyle in model.buildStyles:
                index[model.mid, bstyle.bid] = bstyle

                print(bstyle.bid, model.mid)
        return index

    @staticmethod
    def _modelIndex(models: List[Model]):

        index = {}
        for model in models:
            index[model.mid] = model

        return index

    @staticmethod
    def validateBuildStyle(bstyle: BuildStyle):
        """
        Validates a single :class:`BuildStyle` ensuring that its individual parameters are not malformed.

        :param bstyle: The BuildStyle to validate
        :raise Exception: When an invalid BuildStyle is provided
        """
        if bstyle.bid < 1 or not isinstance(bstyle.bid, int):
            raise Exception("BuildStyle ({:d}) should have a positive integer id".format(bstyle.bid))

        if bstyle.laserPower < 0:
            raise Exception("BuildStyle({:d}).laserPower must be a positive integer".format(bstyle.bid))

        if bstyle.laserSpeed < 0:
            raise Exception("BuildStyle({:d}).laserSpeed must be a positive integer".format(bstyle.bid))

        if bstyle.pointDistance < 1 or not isinstance(bstyle.pointDistance, int):
            raise Exception("BuildStyle({:d}).pointDistance must be a positive integer (>0)".format(bstyle.bid))

        if bstyle.pointExposureTime < 1 or not isinstance(bstyle.pointExposureTime, int):
            raise Exception("BuildStyle({:d}).pointExposureTime must be a positive integer (>0)".format(bstyle.bid))

        if bstyle.jumpDelay < 0 or not isinstance(bstyle.jumpDelay, int):
            raise Exception("BuildStyle({:d}).jumpDelay must be a positive integer ".format(bstyle.bid))

        if bstyle.jumpSpeed < 0 or not isinstance(bstyle.jumpSpeed, int):
            raise Exception("BuildStyle({:d}).jumpSpeed must be a positive integer (>0)".format(bstyle.bid))

        if bstyle.laserId < 1 or not isinstance(bstyle.laserId, int):
            raise Exception("BuildStyle({:d}).laserId must be a positive integer (>0)".format(bstyle.bid))

        if bstyle.laserMode < 0 or not isinstance(bstyle.laserMode, int):
            raise Exception("BuildStyle({:d}).laserMode must be a positive integer (0)".format(bstyle.bid))

        if bstyle.laserId < 1 or not isinstance(bstyle.laserId, int):
            raise Exception("BuildStyle({:d}).laserId must be a positive integer (>0)".format(bstyle.bid))

    @staticmethod
    def validateModel(model: Model):
        """
        Validates a single :class:`Model` ensuring that its individual BuildStyles are not malformed.

        :param model: The `Model` to validate
        :raise Exception: When an invalid `BuildStyle` is provided
        """

        bstyleList = []

        if model.topLayerId == 0:
            raise Exception('The top layer id of Model ({:s}) has not been set'.format(model.name))

        if len(model.buildStyles) == 0:
            raise Exception('Model ({:s} does not contain any build styles'.format(model.name))

        for bstyle in model.buildStyles:

            if bstyle in bstyleList:
                raise Exception('Model ({:s} does not contain build styles with unique id'.format(model.name))
            else:
                bstyleList.append(bstyle.bid)

            ModelValidator.validateBuildStyle(bstyle)

    @staticmethod
    def validateBuild(models: List[Model], layers: List[Layer]):
        """
        Validates an AM Build which compromises of a list of models and layers

        :param models: A list of `Models` used in the build
        :param layers: A list of `Layers` used in the build
        :raise Exception: When an invalid `BuildStyle` is provided
        """
        # Build the indices for the models and the build styles
        modelIdx = ModelValidator._modelIndex(models)
        bstyleIdx = ModelValidator._buildStyleIndex(models)

        modelTopLayerIdx = dict()
        layerDelta = np.array([layer.z for layer in layers])

        for model in models:
            ModelValidator.validateModel(model)

        """ Iterate across each layer and validate the input """
        for layer in layers:

            if len(layer.geometry) == 0:
                warn("Warning: Layer ({:d}) does not contain any layer geometry. It is advised to check this is valid".format(layer.layerId))

            for layerGeom in layer.geometry:
                model = modelIdx.get(layerGeom.mid, None)

                if not model:
                    raise Exception("Layer Geometry in layer ({:d} @ {:.3f}) has not been assigned a model".format(layer.layerId, layer.z))

                bstyle = bstyleIdx.get((layerGeom.mid, layerGeom.bid), None)

                if not bstyle:
                    raise Exception("Layer Geometry in layer ({:d} @ {:.3f}) has not been assigned a buildstyle".format(layer.layerId, layer.z))

                modelTopLayerIdx[model.mid] = layer.layerId

        """ Check to see if all models were assigned to a layer geometry"""
        for model in models:
            if not modelTopLayerIdx.get(model.mid, False):
                warn("Warning: Model({:s}) was not used in any layer)".format(model.name))

            if model.topLayerId != modelTopLayerIdx[model.mid]:
                raise Exception("Top Layer Id {:d} of Model ({:d}) differs in the layers used ({:d})".format(model.topLayerId,
                                                                                                             model.mid,
                                                                                                             modelTopLayerIdx[model.mid]))

        return True
