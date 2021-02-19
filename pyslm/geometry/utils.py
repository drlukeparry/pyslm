from typing import List, Optional, Tuple, Union
import numpy as np
from warnings import warn

from . import Layer, LayerGeometry, HatchGeometry, ContourGeometry, PointsGeometry, BuildStyle, Model

def getBuildStyleById(models: List[Model], mid: int, bid: int) -> Union[BuildStyle, None]:
    """
    Returns the Buildstyle found from a list of :class:`Model`  given a model id and build id.

    :param models: A list of models
    :param mid: The selected model id
    :param bid:  The selected buildstyle id

    :return: The BuildStyle if found or `None`
    """
    model = next(x for x in models if x.mid == mid)

    if model:
        bstyle = next(x for x in model.buildStyles if x.bid == bid)

        return bstyle

    return None


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

    @staticmethod
    def _buildStyleIndex(models):

        index = {}
        for model in models:
            for bstyle in model.buildStyles:
                index[model.mid, bstyle.bid] = bstyle

        return index

    @staticmethod
    def _modelIndex(models):

        index = {}
        for model in models:
            index[model.mid] = model

        return index

    @staticmethod
    def validateBuildStyle(bstyle: BuildStyle):
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
                model = modelIdx.get(model.mid, None)

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
                raise Exception("Top Layer Id of Model ({:d}) differs in the layers used ({:d})".format(model.topLayerId,
                                                                                                        modelTopLayerIdx[model.mid]))

        return True
