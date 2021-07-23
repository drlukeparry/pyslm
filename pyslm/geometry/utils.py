from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from warnings import warn

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


def getBuildStyleById(models: List[Model], mid: int, bid: int) -> Union[BuildStyle, None]:
    """
    Returns the :class:`Buildstyle` found from a list of :class:`Model`  given a model id and buildstyle  id.

    :param models: A list of models
    :param mid: The selected model id
    :param bid: The selected :class:`Buildstyle` id

    :return: The :class:`BuildStyle` if found or `None`
    """
    model = next(x for x in models if x.mid == mid)

    if model:
        bstyle = next(x for x in model.buildStyles if x.bid == bid)

        return bstyle

    return None

def getLayerById(layers: List[Layer], layerId: int) -> Layer:
    """
    Finds the `Layer` within a list given an id

    :param layers: The list of layers to search
    :param layerId: The layer id to find

    :return: If found the layer or `None`
    """
    layer = next(x for x in layers if x.layerId == layerId)

    return layer


def getModel(models: List[Model], mid: int) -> Union[BuildStyle, None]:
    """
    Returns the Model found from a list of :class:`Model`  given a model id and build id.

    :param models: A list of models
    :param mid: The selected model id

    :return: The BuildStyle if found or `None`
    """
    model = next(x for x in models if x.mid == mid)

    return  model if model else None


class ModelValidator:
    """
    ModelValidator takes the `pyslm.geometry` data structures such as a list of  :class:`Layer` and :class:`Model`
    and validates their input for consistency when utilised together to form a build file prior to exporting
    using libSLM. Basic checks include:

    * Validating each :class:`BuildStyle` used in each :class:`Model`, including individual laser parameters
    * References to a correct :class:`BuildStyle` via its id (`bid`) for each :class:`LayerGeometry` section
    * References to a correct :class:`Model` via its id (`mid`) for each :class:`LayerGeometry` section
    * Ensure there are unique :class:`BuildStyle` entries for each :class:`Model`


    The key function that can be called is :meth:`validateBuild` which ideally should be called before attempting to
    export the layer and model information to a libSLM Machine Build file translator. Additional sub-functions are also
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

        :param bstyle: The :class:`BuildStyle` to validate
        :raise Exception: When an invalid :class:`BuildStyle` is provided
        """
        if bstyle.bid < 1 or not isinstance(bstyle.bid, int):
            raise Exception("BuildStyle ({:d}) should have a positive integer id".format(bstyle.bid))

        if bstyle.laserPower < 0 :
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

        :param model: The :class:`Model` to validate
        :raise Exception: When an invalid :class:`BuildStyle` is provided
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

        :param models: A list of :class:`Model` used in the build
        :param layers: A list of :class:`Layer` used in the build
        :raise Exception: When an invalid :class:`BuildStyle` is provided
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
