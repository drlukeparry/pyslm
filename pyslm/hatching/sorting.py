import numpy as np
import networkx as nx

import abc

from .utils import *


class BaseSort(abc.ABC):
    def __init__(self):
        pass

    def __str__(self):
        return 'BaseSorter Feature'

    @abc.abstractmethod
    def sort(self, vectors: np.ndarray) -> np.ndarray:
        """
        Sorts the scan vectors in a particular order

        :param vectors: The un-sorted array of scan vectors
        :return: The sorted array of scan vectors
        """
        raise NotImplementedError('Sort method must be implemented')


class UnidirectionalSort(BaseSort):
    """
    Method simply passes the hatch vectors in their current form.
    """
    def __init__(self):
        super().__init__()

    def __str__(self):
        return 'Unidrectional Hatch Sort'

    def sort(self, scanVectors: np.ndarray) -> np.ndarray:
        """ This approach simply flips the odd pair of hatches"""


class FlipSort(BaseSort):
    """
    Sort method flips all pairs of scan vectors so that their direction alternates across the input
    """
    def __init__(self):
        super().__init__()

    def __str__(self):
        return 'Alternating Hatch Sort'

    def sort(self, scanVectors: np.ndarray) -> np.ndarray:
        """ This approach simply flips the odd pair of hatches"""
        sv = to3DHatchArray(scanVectors)
        sv = np.flip(sv, 1)
        return from3DHatchArray(sv)


class AlternateSort(BaseSort):
    """
    Sort method flips pairs of scan vectors so that their direction alternates across adjacent vectors.
    """
    def __init__(self):
        super().__init__()

    def __str__(self):
        return 'Alternating Hatch Sort'

    def sort(self, scanVectors: np.ndarray) -> np.ndarray:
        """ This approach simply flips the odd pair of hatches"""
        sv = to3DHatchArray(scanVectors)
        sv[1::2] = np.flip(sv[1::2], 1)
        return from3DHatchArray(sv)


class LinearSort(BaseSort):
    """
    A linear sort approaches to sorting the scan vectors based on the current hatch angle specified in
    :attr:`pyslm.hatching.sorting.LinearSort.hatchAngle`. The approach takes the dot product of the hatch mid-point
    and the projection along the X-axis is sorted in ascending order (+ve X direction).
    """

    def __init__(self):
        super().__init__()
        self._hatchAngle = 0.0

    @property
    def hatchAngle(self) -> float:
        """
        The hatch angle reference across the scan vectors to be sorted
        """
        return self._hatchAngle

    @hatchAngle.setter
    def hatchAngle(self, angle: float):
        self._hatchAngle = angle

    def sort(self, scanVectors: np.ndarray) -> np.ndarray:
        # requires an n x 2 x 2 array

        # Sort along the x-axis and obtain the indices of the sorted array

        theta_h = np.deg2rad(self._hatchAngle)

        # Find the unit vector normal based on the hatch angle
        norm = np.array([np.cos(theta_h), np.sin(theta_h)])

        midPoints = np.mean(scanVectors, axis=1)
        idx2 = norm.dot(midPoints.T)
        idx3 = np.argsort(idx2)

        sortIdx = np.arange(len(midPoints))[idx3]

        return scanVectors[sortIdx]


class GreedySort(BaseSort):
    """
    The greedy sort approach is a heuristic approach to sorting the scan vectors based on the current hatch angle
    specified in :attr:`pyslm.hatching.sorting.LinearSort.hatchAngle` and clustering vectors together based on the
    hatch group distance - :attr:`pyslm.hatching.sorting.LinearSort.hatchTol`.

    The approach finds clusters of scan vectors based on their connectivity based on a threshold
    """
    def __init__(self, hatchAngle = 0.0, hatchTol = None):

        super().__init__()

        self._hatchAngle = hatchAngle
        self._sortY = False
        self._clusterDistance = 5  # mm

        if hatchTol:
            self._hatchTol = hatchTol
        else:
            self._hatchTol = 0.1 * 5 # hatchDistance * 5

    def __str__(self):
        return 'GreedySort Feature'

    @property
    def hatchAngle(self) -> float:
        """
        The hatch angle reference across the scan vectors to be sorted
        """
        return self._hatchAngle

    @hatchAngle.setter
    def hatchAngle(self, angle: float):
        self._hatchAngle = angle


    @property
    def hatchTol(self):
        """  The hatch group tolerance specifies the arbitrary distance used for grouping the scan vectors into
        'scanning clusters'"""
        return self._hatchTol

    @hatchTol.setter
    def hatchTol(self, tolerance):
        self._hatchTol = tolerance

    @property
    def sortY(self) -> bool:
        """ Used to set the sorting mode (default sort along x)"""
        return self._sortY

    @sortY.setter
    def sortY(self, state: bool):
        self._sortY = state

    def sort(self, scanVectors):
        """
        Sorts the scan vectors
        """

        from scipy.spatial import distance_matrix

        theta_h = np.deg2rad(self._hatchAngle)

        # vectors is actually the list of midpoints
        midPoints = np.mean(scanVectors, axis=1)

        #print('{:=^60} \n'.format(' Finding hatch distance '))

        # TODO find a more efficient way of producing distance matrix usign KD-tree

        # scipy spatial distancematrix
        distMap = distance_matrix(midPoints, midPoints)
        distMap += np.eye(distMap.shape[0]) * 1e7

        G = nx.from_numpy_matrix(distMap <  self._hatchTol)
    #        print(' Time taken to find hatch distance', time.time() - start)
    #        print('{:=^60} \n'.format(' Finding hatch distance (finished) '))
    #        #G = nx.from_scipy_sparse_matrix(sA)
    #        print('num con components', nx.number_connected_components(G))

        graphs = [G.subgraph(c) for c in nx.algorithms.connected_components(G)] #graphs = list(nx.connected_component_subgraphs(G))

        clusterPaths = []

        for i in range(len(graphs)):

            # Locate the mid points
            gNodes = np.array([n for n in graphs[i]] )
    #        graphPnts = midPoints[gNodes]
    #        sortXidx  =  np.argsort(graphPnts[:,0], axis=0)
    #        sortPnts = graphPnts[sortXidx]
    #
    #        distTravelled = np.cumsum(np.diff(sortPnts[:,0]))
    #
    #        bounds = np.argwhere(np.diff(np.divmod(distTravelled,20)[0]))
    #        branchPnts.append(gNodes[bounds])
    #


            # Find the unit vector normal

            # Sort the cluster of hatches by the hatch direction
            norm = np.array([np.cos(theta_h),np.sin(theta_h)])
            idx2 = norm.dot(midPoints[gNodes].T)
            idx3 = np.argsort(idx2)

            # Add the sorted cluster of vectors
            shortPath = gNodes[idx3]
            clusterPaths.append(shortPath)


        # Next part of algorithm greedily collects the scan vectors then moves onto
        # the next group cluster after a set distance [clusterDistance]
        scanVectorList = []
        lastScanIdx = [0] * len(clusterPaths)

        dPos = 0
        maxMove = dPos

        complete = False

        grpId = 0
        grp = []

        # The cluster groups should be sorted by the first index
        #sort(clusterPaths)
        firstPnts = midPoints[[path[0] for path in clusterPaths]]

        if self._sortY:
            clusterPaths = [clusterPaths[i] for i in np.argsort(firstPnts[:,1])]
        else:
            clusterPaths = [clusterPaths[i] for i in np.argsort(firstPnts[:,0])]

        advancePos = True
        while not complete:

            if advancePos:
                maxMove += self._clusterDistance

            advancePos = True # Reset the flag
            for i in range(len(clusterPaths)):
                innerDist = 0
                clusterNodes = np.array(clusterPaths[i])

                if lastScanIdx[i] == (len(clusterNodes)):
                    continue # Finished scanning the cluster

                # Get the next point in the cluster
                pnt = midPoints[clusterNodes[lastScanIdx[i]]]

                if self._sortY:
                    if pnt[1] > maxMove:
                        continue # Don't jump ahead
                else:
                    if pnt[0] > maxMove:
                        continue # Don't jump ahead

                while innerDist < self._clusterDistance:


                    if lastScanIdx[i] < (len(clusterNodes)):

                        #print('adding', i, lastScanIdx[i])
                        scanVectorList.append(clusterNodes[lastScanIdx[i]])
                        grp.append(grpId)
                        lastScanIdx[i] += 1

                    if lastScanIdx[i] == (len(clusterNodes)):
                        break

                    pntCur = midPoints[clusterNodes[lastScanIdx[i]]]
                    pntPrev = midPoints[clusterNodes[lastScanIdx[i]-1]]
                    delta = pntCur -  pntPrev

                    if self._sortY:
                        innerDist += delta[1]
                        dPos = np.max([dPos, pntCur[1]])
                    else:
                        innerDist += delta[0]
                        dPos = np.max([dPos, pntCur[0]])

                # because a valid group was found move back to start


                grpId += 1

                advancePos = False
                break


            clusterLen = [len(path) for path in clusterPaths]
            complete = np.sum(np.array(lastScanIdx) - clusterLen) == 0

        idx6 = np.arange(len(midPoints))[scanVectorList]

        return scanVectors[idx6]

