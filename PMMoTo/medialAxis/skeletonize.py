import numpy as np
from mpi4py import MPI
from .. import communication
from ._skeletonize_3d_cy import _compute_thin_image
from .. import nodes
from .. import sets
import math
comm = MPI.COMM_WORLD


class medialAxis(object):
    """
    Calculate Medial Axis and PostProcess
    Nodes -> Sets -> Paths
    Sets are broken into Reaches -> Medial Nodes -> Medial Clusters
    """
    def __init__(self,Domain,subDomain):
        self.Domain = Domain
        self.subDomain = subDomain
        self.Orientation = subDomain.Orientation
        self.padding = np.zeros([3],dtype=np.int64)
        self.haloGrid = None
        self.MA = None

    def skeletonize_3d(self):
        """Compute the skeleton of a binary image.

        Thinning is used to reduce each connected component in a binary image
        to a single-pixel wide skeleton.

        Parameters
        ----------
        image : ndarray, 2D or 3D
            A binary image containing the objects to be skeletonized. Zeros
            represent background, nonzero values are foreground.

        Returns
        -------
        skeleton : ndarray
            The thinned image.

        See Also
        --------
        skeletonize, medial_axis

        Notes
        -----
        The method of [Lee94]_ uses an octree data structure to examine a 3x3x3
        neighborhood of a pixel. The algorithm proceeds by iteratively sweeping
        over the image, and removing pixels at each iteration until the image
        stops changing. Each iteration consists of two steps: first, a list of
        candidates for removal is assembled; then pixels from this list are
        rechecked sequentially, to better preserve connectivity of the image.

        The algorithm this function implements is different from the algorithms
        used by either `skeletonize` or `medial_axis`, thus for 2D images the
        results produced by this function are generally different.

        References
        ----------
        .. [Lee94] T.-C. Lee, R.L. Kashyap and C.-N. Chu, Building skeleton models
               via 3-D medial surface/axis thinning algorithms.
               Computer Vision, Graphics, and Image Processing, 56(6):462-478, 1994.

        """
        self.haloGrid = np.ascontiguousarray(self.haloGrid)
        image_o = np.copy(self.haloGrid)

        # normalize to binary
        image_o[image_o != 0] = 1

        # do the computation
        image_o = np.asarray(_compute_thin_image(image_o))
        dim = image_o.shape
        self.MA = image_o[self.halo[1]:dim[0]-self.halo[0],
                          self.halo[3]:dim[1]-self.halo[2],
                          self.halo[5]:dim[2]-self.halo[4]]
        self.MA = np.ascontiguousarray(self.MA)

    def genPadding(self):
        """
        Current Parallel MA implementation simply pads subDomains to match. Very work ineffcieint and needs to be changes
        """
        gridShape = self.Domain.subNodes
        factor = 0.95
        self.padding[0] = math.ceil(gridShape[0]*factor)
        self.padding[1] = math.ceil(gridShape[1]*factor)
        self.padding[2] = math.ceil(gridShape[2]*factor)

        for n in [0,1,2]:
            if self.padding[n] == gridShape[n]:
                self.padding[n] = self.padding[n] - 1


    def collectPaths(self):
        """
        Collect Sets into Paths
        """
        self.paths = {}
        for nS in range(0,self.setCount):
            pathID = self.Sets[nS].pathID
            if pathID not in self.paths.keys():
                self.paths[pathID] = {'Sets':[],
                                      'boundarySets':[],
                                      'inlet':False,
                                      'outlet':False}

            self.paths[pathID]['Sets'].append(nS)

            if self.Sets[nS].boundary:
                self.paths[pathID]['boundarySets'].append(self.Sets[nS].localID)

            if self.Sets[nS].inlet:
                self.paths[pathID]['inlet'] = True

            if self.Sets[nS].outlet:
                self.paths[pathID]['outlet'] = True

        self.boundPathCount = 0
        for p in self.paths.keys():
            if self.paths[p]['boundarySets']:
                self.boundPathCount = self.boundPathCount + 1


    def updateConnectedSetsID(self,connectedSetData):
        self.connectedSetIDs = {}
        for s in connectedSetData:
            for ss in s:
                if ss[0] == self.subDomain.ID:
                    for c,n in enumerate(ss[4]):
                        self.connectedSetIDs[n]=self.Sets[n].globalID
                if ss[2] == self.subDomain.ID:
                    for c,n in enumerate(ss[5]):
                        self.connectedSetIDs[n]=self.Sets[n].globalID

    def updatePaths(self,globalPathIndexStart,globalPathBoundarySetID):
        self.paths = {}
        c = 0
        for nS in range(0,self.setCount):
            pathID = self.Sets[nS].pathID

            ind = np.where(globalPathBoundarySetID[:,1]==pathID)[0]
            if ind:
                ind = ind[0]
                pathID = globalPathBoundarySetID[ind,2]
                setInlet = globalPathBoundarySetID[ind,3]
                setOutlet = globalPathBoundarySetID[ind,4]

                if pathID not in self.paths.keys():
                    self.paths[pathID] = {'Sets':-1,'boundarySets':[],'inlet':setInlet,'outlet':setOutlet}
                self.paths[pathID]['Sets'] = nS

                if self.Sets[nS].boundary:
                    self.paths[pathID]['boundarySets'].append(self.Sets[nS].globalID)
            else:
                pathID = globalPathIndexStart + c
                c = c + 1
                if pathID not in self.paths.keys():
                    self.paths[pathID] = {'Sets':[],'boundarySets':[],'inlet':False,'outlet':False}
                self.paths[pathID]['Sets'] = nS

                if self.Sets[nS].boundary:
                    self.paths[pathID]['boundarySets'].append(self.Sets[nS].globalID)

                if self.Sets[nS].inlet:
                    self.paths[pathID]['inlet'] = True

                if self.Sets[nS].outlet:
                    self.paths[pathID]['outlet'] = True

    def trimSets(self):
        """
        Trim Set if Not Connected to Inlet/Outlet or Two Medial Nodes or Clusters
        """
        for p in self.paths.keys():
            setID = self.paths[p]['Sets']
            if self.Sets[setID].inlet or self.Sets[setID].outlet:
                pass
            else:
                if self.Sets[setID].type == 0  and len(self.Sets[setID].globalConnectedSets) < 2:
                    self.Sets[setID].trim = True






def medialAxisEval(rank,size,Domain,subDomain,grid,distance):

    ### Initialize Classes
    sDMA = medialAxis(Domain = Domain,subDomain = subDomain)
    sDComm = communication.Comm(Domain = Domain,subDomain = subDomain,grid = grid)

    ### Adding Padding so Identical MA at Processer Interfaces
    sDMA.genPadding()

    ### Send Padding Data to Neighbors
    sDMA.haloGrid,sDMA.halo = sDComm.haloCommunication(sDMA.padding)

    ### Determine MA
    sDMA.skeletonize_3d()

    ### Get Info for Medial Axis Nodes and Get Connected Sets and Boundary Sets
    sDMA.nodeInfo,sDMA.nodeInfoIndex,sDMA.nodeDirections,sDMA.nodeDirectionsIndex,sDMA.nodeTable = nodes.getNodeInfo(sDMA.MA,Domain,subDomain,subDomain.Orientation)
    sDMA.Sets,sDMA.setCount,sDMA.pathCount = nodes.getConnectedMedialAxis(rank,sDMA.MA,sDMA.nodeInfo,sDMA.nodeInfoIndex,sDMA.nodeDirections,sDMA.nodeDirectionsIndex)

    sDMA.boundaryData,sDMA.boundarySets,sDMA.boundSetCount = sets.getBoundarySets(sDMA.Sets,sDMA.setCount,subDomain)

    ### Connect the Sets into Paths
    sDMA.collectPaths()

    ### Send Boundary Set Data to Neighbors and Match Boundary Sets. Gather Matched Sets
    sDMA.boundaryData = sets.setCOMM(subDomain.Orientation,subDomain,sDMA.boundaryData)
    sDMA.matchedSets,sDMA.matchedSetsConnections,error = sets.matchProcessorBoundarySets(subDomain,sDMA.boundaryData,True)
    if error:
        communication.raiseError()
    setData = [sDMA.matchedSets,sDMA.setCount,sDMA.boundSetCount,sDMA.pathCount,sDMA.boundPathCount]
    setData = comm.gather(setData, root=0)

    ### Gather Connected Sets and Update Path and Set Infomation (ID,Inlet/Outlet)
    connectedSetData =  comm.allgather(sDMA.matchedSetsConnections)
    globalIndexStart,globalBoundarySetID,globalPathIndexStart,globalPathBoundarySetID = sets.organizePathAndSets(subDomain,size,setData,True)
    if size > 1:
        sets.updateSetPathID(rank,sDMA.Sets,globalIndexStart,globalBoundarySetID,globalPathIndexStart,globalPathBoundarySetID)
        sDMA.updatePaths(globalPathIndexStart,globalPathBoundarySetID)
        sDMA.updateConnectedSetsID(connectedSetData)
        connectedSetIDs =  comm.allgather(sDMA.connectedSetIDs)
        sets.getGlobalConnectedSets(sDMA.Sets,connectedSetData[rank],connectedSetIDs)

    ### Trim Sets on Paths that are Dead Ends
        ## TODO: Currently nonfunctional for single processor solves, not high priority
        sDMA.trimSets()

    ### Get Min and MAx Distance for Every Set
    for s in sDMA.Sets:
        s.getDistMinMax(distance)
        # if rank == 0:
        #     print(s.localID,s.globalID,s.localConnectedSets,s.globalConnectedSets)

    return sDMA
