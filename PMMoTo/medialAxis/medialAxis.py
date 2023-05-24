import numpy as np
from mpi4py import MPI
from .. import communication
from .medialExtraction import _compute_thin_image
from .. import nodes
from .. import sets
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
        self.halo = np.zeros(6)
        self.haloPadNeigh = np.zeros(6)
        self.haloPadNeighNot = np.zeros(6)
        self.MA = None

    def skeletonizeAxis(self,connect = False):
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

        ### Grab Medial Axis with Single and Two Buffer to 
        self.haloPadNeigh = np.zeros_like(self.halo)
        self.haloPadNeighNot = np.ones_like(self.halo)
        for n in range(0,6):
            if self.halo[n] > 0:
                self.haloPadNeigh[n] = 1
                self.haloPadNeighNot[n] = 0

        if connect:
            self.MA = image_o[self.halo[0] - self.haloPadNeigh[0] : dim[0] - self.halo[1] + self.haloPadNeigh[1],
                              self.halo[2] - self.haloPadNeigh[2] : dim[1] - self.halo[3] + self.haloPadNeigh[3],
                              self.halo[4] - self.haloPadNeigh[4] : dim[2] - self.halo[5] + self.haloPadNeigh[5]]

        else:
            self.MA = image_o[self.halo[0]:dim[0] - self.halo[1],
                              self.halo[2]:dim[1] - self.halo[3],
                              self.halo[4]:dim[2] - self.halo[5]]
                
        self.MA = np.ascontiguousarray(self.MA)


    def genPadding(self):
        """
        Current Parallel MA implementation simply pads subDomains to match. Very work ineffcieint and needs to be changed
        """
        gridShape = self.Domain.subNodes
        factor = 0.95
        self.padding[0] = 1#math.ceil(gridShape[0]*factor)
        self.padding[1] = 1#math.ceil(gridShape[1]*factor)
        self.padding[2] = 1#math.ceil(gridShape[2]*factor)

        # for n in [0,1,2]:
        #     if self.padding[n] == gridShape[n]:
        #         self.padding[n] = self.padding[n] - 1

        

    def genMAArrays(self):
        """
        Generate Trimmed MA arrays to get nodeInfo and Correct Neighbor Counts for Boundary Nodes
        """
        dim = self.MA.shape
        tempMA = self.MA[self.haloPadNeigh[0] : dim[0] - self.haloPadNeigh[1],
                         self.haloPadNeigh[2] : dim[1] - self.haloPadNeigh[3],
                         self.haloPadNeigh[4] : dim[2] - self.haloPadNeigh[5]]
                
        neighMA = np.pad(self.MA, ( (self.haloPadNeighNot[0], self.haloPadNeighNot[1]), 
                                    (self.haloPadNeighNot[2], self.haloPadNeighNot[3]), 
                                    (self.haloPadNeighNot[4], self.haloPadNeighNot[5]) ), 
                                    'constant', constant_values=0)
        return tempMA,neighMA

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


    def genLocalGlobalConnectedSetsID(self,connectedSetData):
        """
        Generate dictionaries for 
            localGlobalConnectedSetID: ['localID'] = globalID
        """
        self.localGlobalConnectedSetID = {}
        for s in connectedSetData:
            for ss in s:
                if ss[0] == self.subDomain.ID:
                    self.localGlobalConnectedSetID[ss[1]]=self.Sets[ss[1]].globalID
                    for n in ss[4]:
                        self.localGlobalConnectedSetID[n]=self.Sets[n].globalID
                if ss[2] == self.subDomain.ID:
                    self.localGlobalConnectedSetID[ss[3]]=self.Sets[ss[3]].globalID
                    for n in ss[5]:
                        self.localGlobalConnectedSetID[n]=self.Sets[n].globalID

    def genGlobalLocalConnectedSetsID(self,localGlobalSets):
        """
        Input [numProcs][localID] = globalID
        Generate dictionaries for 
            globalLocalConnectedSetID: ['globalID']= {procID:localID}
        """
        self.globalLocalConnectedSetID = {}
        for nP,nSets in enumerate(localGlobalSets):
            for lID in nSets:
                if nSets[lID] not in self.globalLocalConnectedSetID.keys():
                    self.globalLocalConnectedSetID[nSets[lID]] = {}
                if nP not in self.globalLocalConnectedSetID[nSets[lID]]:
                    self.globalLocalConnectedSetID[nSets[lID]][nP] = []
                self.globalLocalConnectedSetID[nSets[lID]][nP].append(lID)


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

    def globalCleanSets(self,setData):
        
        setData = [i for j in setData for i in j]

        ## Sort by id
        setData.sort()
        ## Remove repeated entries, based on 
        ## sets added to local lists to check
        ## connectivity
        cleanSetData = []
        seenID = []
        for set in setData:
            if not set[0] in seenID:
                seenID.append(set[0])
                cleanSetData.append(set)
            else:
                # if sorted(cleanSetData[set[0]][1]) != sorted(set[1]):
                #     print('WARNING: Global connectivity information does not match for sets shared by subprocesses.')
                cleanSetData[set[0]][1] = list(frozenset(cleanSetData[set[0]][1] + set[1]))
                if cleanSetData[set[0]][5] != set[5]:
                    # print('WARNING: Min distance values are not equal.')
                    cleanSetData[set[0]][5] = min(cleanSetData[set[0]][5],set[5])
        
        ### There exist cases where a set has itself listed as a neighbor, remove these. SEE ISSUE 21
        for i,set in enumerate(cleanSetData):
            if i in set[1]:
                set[1] = [j for j in set[1] if j != i]
        
        return setData,cleanSetData

    def globalTrimSets(self,cleanSetData):
        ### Perform extra iterations of fork trimming
        ### will apply to Sets not connected to inlet
        trimsAdded = 1
        while trimsAdded:
            trimsAdded = 0
            for set in cleanSetData:
                if (set[2] == 1) or (set[3] == 1) or (set[4] == 1):
                    continue
                nConnectedTrim = 0
                
                for connectedSetID in set[1]:
                    if cleanSetData[connectedSetID][4]:
                        nConnectedTrim += 1

                isoCheck = (nConnectedTrim == len(set[1]))
                forkCheck = (nConnectedTrim == (len(set[1])-1))
                endCheck  = (len(set[1]) < 2)

                if isoCheck or forkCheck or endCheck:
                    set[4] = True
                    trimsAdded = 1
        
        return cleanSetData
    
    def globalInaccessibleTrimSets(self,cleanSetData):
        trimsAdded = 1
        while trimsAdded:
            trimsAdded = 0
            for set in cleanSetData:
                if set[5] == 1:
                    set[6] = 1
                if (set[2] == 1)  or (set[3] == 1) or (set[6] == 1):
                    continue

                nConnectedTrim = 0
                for connectedSetID in set[1]:
                    if cleanSetData[connectedSetID][6]:
                        nConnectedTrim += 1

                isoCheck = (nConnectedTrim == len(set[1]))
                forkCheck = (nConnectedTrim == (len(set[1])-1))
                endCheck  = (len(set[1]) < 2)

                if isoCheck or forkCheck or endCheck:
                    set[6] = 1
                    trimsAdded = 1
        
        return cleanSetData
    
    def globalInaccessibleSets(self,cleanSetData,cutoff):

        ### Perform extra iterations of fork trimming
        ### will apply to Sets not connected to inlet
        for set in cleanSetData:
            if set[7] < cutoff:
                set[5] = 1
                    
        return cleanSetData

    def globalCreateListSetData(self,size,setData,cleanSetData):
        
        for set in setData:
            setID = set[0]
            set[4] = cleanSetData[setID][4]
            set[5] = cleanSetData[setID][5]
            set[6] = cleanSetData[setID][6]
            set[7] = cleanSetData[setID][7]

        listSetData = []
        for i in range(size):
            listSetData.append([set for set in setData if set[-1] == i])
        return listSetData
    
    def localTrimSets(self):
        """
        Iteratively flip trim flag at rank level until no new changes are made 
        Trim flag is flipped from 0 to 1 if:
        (1) set is a dead end (only one neighbor)
        (2) set is upstream from a dead end (number of neighbors - 1 = number of neighbors flagged for trim)
        (3) set is surrounded by trimmed sets (number of neighbors = number of neighbors flagged for trim)

        At this stage, sets that touch > 1 boundary are skipped to prevent preemptive trims of sets that span
        multiple subdomains
        """
        self.Sets.sort()
        idHashTable = []
        for set in self.Sets:
            idHashTable.append(set.globalID)
        
        trimsAdded = 1
        while trimsAdded:
            trimsAdded = 0
            for set in self.Sets:
                if set.trim or (set.numBoundaries > 1):
                    continue
                
                nConnectedTrim = 0
                
                for connectedSet in set.globalConnectedSets:
                    if (connectedSet in idHashTable):
                        setsIndex = idHashTable.index(connectedSet)
                        if self.Sets[setsIndex].trim:
                            nConnectedTrim += 1

                isoCheck = (nConnectedTrim == len(set.globalConnectedSets))
                forkCheck = (nConnectedTrim == (len(set.globalConnectedSets)-1))
                endCheck  = (len(set.globalConnectedSets) < 2)

                if isoCheck or forkCheck or endCheck:
                    set.trim = True
                    trimsAdded = 1

    def gatherSetInfo(self,rank):
        """
        Convert list of Set objects into a list of lists
        containing only the information needed to trim and
        set pathID
        """

        setData = []
        for s in self.Sets:
            ### globalID,globalID of connected sets, on inlet, on outlet, is trimmed, globalPathID init to -1, visited flag, rank of contributing for reconstruction via scatter
            setData.append([s.globalID, s.globalConnectedSets, s.inlet, s.outlet, s.trim, s.inaccessible, s.inaccessibleTrim, s.minDistance, rank])

        return setData
    
    def updateSetInfo(self,setData,rank):
        """
        Use list form of set information scattered from root to update
        Set objects contained in a list on all ranks
        """
        self.Sets.sort()
        setData.sort()
        for i, set in enumerate(self.Sets):
            set.trim = setData[i][4]
            set.inaccessible = setData[i][5]
            set.inaccessibleTrim = setData[i][6]
            set.minDistance = setData[i][7]
        


def medialAxisEval(rank,size,Domain,subDomain,grid,distance,connect,cutoff):

    ### Initialize Classes
    sDMA = medialAxis(Domain = Domain,subDomain = subDomain)
    sDComm = communication.Comm(Domain = Domain,subDomain = subDomain,grid = grid)

    ### Adding Padding so Identical MA at Processer Interfaces
    sDMA.genPadding()

    ### Send Padding Data to Neighbors
    sDMA.haloGrid,sDMA.halo = sDComm.haloCommunication(sDMA.padding)

    ### Determine MA
    sDMA.skeletonizeAxis(connect)
    
    if connect:

      ### Get Info for Medial Axis Nodes and Get Connected Sets and Boundary Sets
      tempMA,neighMA = sDMA.genMAArrays()
      sDMA.nodeInfo,sDMA.nodeInfoIndex,sDMA.nodeDirections,sDMA.nodeDirectionsIndex,sDMA.nodeTable = nodes.getNodeInfo(rank,tempMA,1,subDomain.inlet,subDomain.outlet,Domain,subDomain,subDomain.Orientation)
      sDMA.MANodeType = nodes.updateMANeighborCount(neighMA,subDomain,subDomain.Orientation,sDMA.nodeInfo)
      sDMA.MA = np.ascontiguousarray(tempMA)
      
      sDMA.Sets,sDMA.setCount,sDMA.pathCount = sets.getConnectedMedialAxis(rank,sDMA.MA,sDMA.nodeInfo,sDMA.nodeInfoIndex,sDMA.nodeDirections,sDMA.nodeDirectionsIndex,sDMA.MANodeType)
      sDMA.boundaryData,sDMA.boundarySets,sDMA.boundSetCount = sets.getBoundarySets(sDMA.Sets,sDMA.setCount,subDomain)

    #   ### Connect the Sets into Paths
    #   sDMA.collectPaths()

    #   ### Send Boundary Set Data to Neighbors and Match Boundary Sets. Gather Matched Sets
    #   ### matchedSets = [subDomain.ID,ownSet,nbProc,otherSetKeys[testSetKey],inlet,outlet,ownPath,otherPath]
    #   ### matchedSetsConnections = [subDomain.ID,ownSet,nbProc,otherSetKeys[testSetKey],ownConnections,otherConnections]
    #   sDMA.boundaryData = sets.setCOMM(subDomain.Orientation,subDomain,sDMA.boundaryData)
    #   sDMA.matchedSets,sDMA.matchedSetsConnections,error = sets.matchProcessorBoundarySets(subDomain,sDMA.boundaryData,True)
    #   if error:
    #       communication.raiseError()
    #   setData = [sDMA.matchedSets,sDMA.setCount,sDMA.boundSetCount,sDMA.pathCount,sDMA.boundPathCount]
    #   setData = comm.gather(setData, root=0)

    #   ### Gather Connected Sets and Update Path and Set Infomation (ID,Inlet/Outlet)
    #   connectedSetData =  comm.allgather(sDMA.matchedSetsConnections)
    #   globalIndexStart,globalBoundarySetID,globalPathIndexStart,globalPathBoundarySetID = sets.organizePathAndSets(subDomain,size,setData,True)
    #   if size > 1:
    #       sets.updateSetPathID(rank,sDMA.Sets,globalIndexStart,globalBoundarySetID,globalPathIndexStart,globalPathBoundarySetID)
    #       sDMA.updatePaths(globalPathIndexStart,globalPathBoundarySetID)

    #       ### Generate Local <-> Global Connected Set IDs
    #       sDMA.genLocalGlobalConnectedSetsID(connectedSetData)
    #       localGlobalConnectedSetIDs = comm.allgather(sDMA.localGlobalConnectedSetID) 
    #       sDMA.genGlobalLocalConnectedSetsID(localGlobalConnectedSetIDs)
    #       sets.getGlobalConnectedSets(rank,size,subDomain,sDMA.Sets,connectedSetData,localGlobalConnectedSetIDs,sDMA.globalLocalConnectedSetID)


    #   for s in sDMA.Sets:
    #       s.getDistMinMax(distance)

    #   ### Trim Sets on Paths that are Dead Ends
    #   ### Trim sets that are not viable inlet-outlet pathways via subdomain level observation
    #   sDMA.localTrimSets()


    #   ### Collect only necessary Set object data for transfer to root
    #   setData = sDMA.gatherSetInfo(rank)
    #   ### Gather all lists into one on root
    #   setData = comm.gather(setData,root=0)

    #   ### Initialize object for later scattering
    #   if rank != 0:
    #       listSetData = None
    #   if rank == 0:
    #       setData,cleanSetData = sDMA.globalCleanSets(setData)
    #       cleanSetData = sDMA.globalTrimSets(cleanSetData)
    #       cleanSetData = sDMA.globalInaccessibleSets(cleanSetData,cutoff)
    #       cleanSetData = sDMA.globalInaccessibleTrimSets(cleanSetData)
    #       listSetData = sDMA.globalCreateListSetData(size,setData,cleanSetData)
    #   setData = comm.scatter(listSetData,root=0)

    #   sDMA.updateSetInfo(setData,rank)


    return sDMA
