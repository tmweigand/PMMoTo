import math
import numpy as np
cimport numpy as cnp
from libc.stdio cimport printf
cnp.import_array()
from mpi4py import MPI
#from pykdtree.kdtree import KDTree
### if using WSL, uncomment line below. 
from scipy.spatial import KDTree

import edt
from . import communication
cimport cython

comm = MPI.COMM_WORLD

""" Solid = 0, Pore = 1 """

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def _fixInterfaceCalc(self,
                     tree,
                     int faceID,
                     int lShape,
                     int dir,
                     cnp.ndarray[cnp.int32_t, ndim=2] _faceSolids,
                     cnp.ndarray[cnp.float32_t, ndim=3] _EDT,
                     cnp.ndarray[cnp.uint8_t, ndim=3] _visited,
                     double minD,
                     list coords,
                     cnp.ndarray[cnp.uint8_t, ndim=1] argOrder):
    """
    Uses the solids from neighboring subProcessors to determine if distance is less than determined
    """
    cdef int i,l,m,n,endL,iShape
    cdef float maxD,d

    _orderG = np.ones((1,3), dtype=np.double) #Global Order
    _orderL = np.ones((3), dtype=np.uint32)   #Local Order
    cdef cnp.uint32_t [:] orderL
    orderL = _orderL

    cdef int a0 = argOrder[0]
    cdef int a1 = argOrder[1]
    cdef int a2 = argOrder[2]

    cdef cnp.double_t [:] c0 = coords[a0]
    cdef cnp.double_t [:] c1 = coords[a1]
    cdef cnp.double_t [:] c2 = coords[a2]

    iShape = _faceSolids.shape[0]

    if (dir == 1):
        for i in range(0,iShape):

            if _faceSolids[i,argOrder[0]] < 0:
                endL = lShape
            else:
                endL = _faceSolids[i,argOrder[0]]

            distChanged = True
            l = 0
            while distChanged and l < endL:
                m = _faceSolids[i,argOrder[1]]
                n = _faceSolids[i,argOrder[2]]
                _orderG[0,a0] = c0[l]
                _orderG[0,a1] = c1[m]
                _orderG[0,a2] = c2[n]
                orderL[a0] = l
                orderL[a1] = m
                orderL[a2] = n

                maxD = _EDT[orderL[0],orderL[1],orderL[2]]
                if (maxD > minD):
                    d,ind = tree.query(_orderG,distance_upper_bound=maxD)
                    if d < maxD:
                        _EDT[orderL[0],orderL[1],orderL[2]] = d
                        distChanged = True
                        _visited[orderL[0],orderL[1],orderL[2]] = 1
                    elif _visited[orderL[0],orderL[1],orderL[2]] == 0:
                        distChanged = False
                l = l + 1

    if (dir == -1):
        for i in range(0,iShape):

            if _faceSolids[i,argOrder[0]] < 0:
                endL = 0
            else:
                endL = _faceSolids[i,argOrder[0]]

            distChanged = True
            l = lShape - 1

            while distChanged and l > endL:

                m = _faceSolids[i,argOrder[1]]
                n = _faceSolids[i,argOrder[2]]
                _orderG[0,a0] = c0[l]
                _orderG[0,a1] = c1[m]
                _orderG[0,a2] = c2[n]
                orderL[a0] = l
                orderL[a1] = m
                orderL[a2] = n

                maxD = _EDT[orderL[0],orderL[1],orderL[2]]
                if (maxD > minD):
                    d,ind = tree.query(_orderG,distance_upper_bound=maxD)
                    if d < maxD:
                        _EDT[orderL[0],orderL[1],orderL[2]] = d
                        distChanged = True
                        _visited[orderL[0],orderL[1],orderL[2]] = 1
                    elif _visited[orderL[0],orderL[1],orderL[2]] == 0:
                        distChanged = False
                l = l - 1
    return _EDT,_visited


def _getBoundarySolids(self,
                       int faceID,
                       int dir,
                       cnp.ndarray[cnp.uint8_t, ndim=1] argOrder,
                       int nS,
                       cnp.ndarray[cnp.uint8_t, ndim=3] _grid,
                       cnp.ndarray[cnp.int32_t, ndim=2] _solids):
    """
    Determine the nearest solid voxel to the boundary using optimized loops for the faces
    """
    cdef int c,m,n

    _order = np.ones((3), dtype=np.uint32)
    cdef cnp.uint32_t [:] order
    order = _order

    cdef int a0 = argOrder[0]
    cdef int a1 = argOrder[1]
    cdef int a2 = argOrder[2]

    cdef int s0 = _grid.shape[a0]
    cdef int s1 = _grid.shape[a1]
    cdef int s2 = _grid.shape[a2]

    if (dir == 1):
        for m in range(0,s1):
            for n in range(0,s2):
                solid = False
                c = 0
                while not solid and c < s0:
                    order[a0] = c
                    order[a1] = m
                    order[a2] = n
                    if _grid[order[0],order[1],order[2]] == 0:
                        solid = True
                        _solids[nS,0:3] = order
                        _solids[nS,3] = faceID
                        nS = nS + 1
                    else:
                        c = c + 1
                if (not solid and c == s0):
                    order[a0] = -1
                    _solids[nS,0:3] = order
                    _solids[nS,3] = faceID
                    nS = nS + 1

    elif (dir == -1):
        for m in range(0,s1):
            for n in range(0,s2):
                solid = False
                c = s0 - 1
                while not solid and c > 0:
                    order[a0] = c
                    order[a1] = m
                    order[a2] = n
                    if _grid[order[0],order[1],order[2]] == 0:
                        solid = True
                        _solids[nS,0:3] = order
                        _solids[nS,3] = faceID
                        nS = nS + 1
                    else:
                        c = c - 1
                if (not solid and c == 0):
                    order[a0] = -1
                    _solids[nS,0:3] = order
                    _solids[nS,3] = faceID
                    nS = nS + 1
    return nS


class EDT(object):

    def __init__(self,subDomain,Domain,Orientation,grid):
        bufferSize         = 1
        self.extendFactor  = 0.7
        self.useIndex      = False
        self.subDomain     = subDomain
        self.Domain        = Domain
        self.Orientation   = Orientation
        self.EDT = np.zeros_like(grid)
        self.visited = np.zeros_like(grid,dtype=np.uint8)
        self.solids = None
        self.nS = 0
        self.faceSolids = []
        self.edgeSolids = []
        self.cornerSolids = []
        self.solidsAll = {self.subDomain.ID: {'orientID':{}}}
        self.grid = grid
        self.x = self.subDomain.x
        self.y = self.subDomain.y
        self.z = self.subDomain.z
        self.subDomainSize = self.subDomain.subDomainSize
        self.buffer = self.subDomain.buffer
        self.distVals = None
        self.distCounts  = None
        self.minD = 0
        self.maxD = 0

    def genLocalEDT(self,):
        """
        Determine the Euclidian distance on each subProcessor knowing the values may be too high
        """
        self.EDT = edt.edt3d(self.grid, anisotropy=(self.Domain.dX, self.Domain.dY, self.Domain.dZ))

    def getBoundarySolids(self):
        """
        Loop through all faces to determine the nearest pore voxels to the boundnary
        """
        area = 2*self.grid.shape[0]*self.grid.shape[1] + 2*self.grid.shape[0]*self.grid.shape[2] + 2*self.grid.shape[1]*self.grid.shape[2]
        self.solids = -np.ones([area,4],dtype='int32')
        for fIndex in self.Orientation.faces:
            self.nS = _getBoundarySolids(self,
                               fIndex,
                               self.Orientation.faces[fIndex]['dir'],
                               self.Orientation.faces[fIndex]['argOrder'],
                               self.nS,
                               self.grid,
                               self.solids)

    def getFaceSolids(self):
        """
        Trim to minimize communication and reduce KD Tree. Identify on Surfaces, Edges, and Corners
        """
        self.faceSolids = [[] for _ in range(len(self.Orientation.faces))]

        extend = [self.extendFactor*x for x in self.subDomainSize]
        for fIndex in self.Orientation.faces:
            pointsXYZ = []
            points = self.solids[np.where( (self.solids[:,0]>-1)
                                 & (self.solids[:,1]>-1)
                                 & (self.solids[:,2]>-1)
                                 & ((self.solids[:,3]==fIndex)) )][:,0:3]

            for x,y,z in points:
                pointsXYZ.append([self.x[x],self.y[y],self.z[z]] )
            self.faceSolids[fIndex] = np.asarray(pointsXYZ)
            name = self.Orientation.faces[fIndex]['ID']
            self.solidsAll[self.subDomain.ID]['orientID'][name] = np.copy(self.faceSolids[fIndex])

    def getEdgeSolids(self):
        """
        Trim to minimize communication and reduce KD Tree. Identify on Surfaces, Edges, and Corners
        """
        self.edgeSolids = [[] for _ in range(len(self.Orientation.edges))]
        for eIndex in self.Orientation.edges:
            edgeID = self.Orientation.edges[eIndex]['ID']
            face1 = self.faceSolids[self.Orientation.edges[eIndex]['faceIndex'][0]]
            face2 = self.faceSolids[self.Orientation.edges[eIndex]['faceIndex'][1]]
            arg1  = self.Orientation.edges[eIndex]['dir'][0]
            arg2  = self.Orientation.edges[eIndex]['dir'][1]
            coords = [self.x,self.y,self.z]
            coord1 = coords[arg1]
            coord2 = coords[arg2]

            extend = [self.extendFactor*x for x in self.subDomainSize]
            plusArg = 0
            if edgeID[arg1] == 1:
                dom11 = coord1[-1] - extend[arg1]
                dom12 = coord1[-1]
            elif (edgeID[arg1] == -1):
                dom11 = coord1[0]
                dom12 = coord1[0] + extend[arg1]
            if edgeID[arg2] == 1:
                dom21 = coord2[-1] - extend[arg2]
                dom22 = coord2[-1]
            elif (edgeID[arg2] == -1):
                dom21 = coord2[0]
                dom22 = coord2[0] + extend[arg2]

            truth1 = dom21 <= face1[:,arg2+plusArg]
            truth2 = face1[:,arg2+plusArg] <= dom22
            self.edgeSolids[eIndex] = face1[np.where( truth1 & truth2 )]
            truth1 = dom11 <= face2[:,arg1+plusArg]
            truth2 = face2[:,arg1+plusArg] <= dom12
            self.edgeSolids[eIndex] = np.append(self.edgeSolids[eIndex],face2[np.where( truth1 & truth2 )],axis=0)
            self.edgeSolids[eIndex] = np.unique(np.array(self.edgeSolids[eIndex]),axis=0)

    def getCornerSolids(self):
        """
        Trim to minimize communication and reduce KD Tree. Identify on Surfaces, Edges, and Corners
        """
        self.cornerSolids = [[] for _ in range(len(self.Orientation.corners))]

        for cIndex in self.Orientation.corners:
            cID = self.Orientation.corners[cIndex]['ID']
            face1 = self.faceSolids[self.Orientation.corners[cIndex]['faceIndex'][0]]
            face2 = self.faceSolids[self.Orientation.corners[cIndex]['faceIndex'][1]]
            face3 = self.faceSolids[self.Orientation.corners[cIndex]['faceIndex'][2]]

            plusArg = 0
            extend = [self.extendFactor*x for x in self.subDomainSize]
            if (cID[0] == 1):
                dom11 = self.x[-1] - extend[0]
                dom12 = self.x[-1]
            elif (cID[0] == -1):
                dom11 = self.x[0]
                dom12 = self.x[0] + extend[0]

            if (cID[1] == 1):
                dom21 = self.y[-1] - extend[1]
                dom22 = self.y[-1]
            elif (cID[1] == -1):
                dom21 = self.y[0]
                dom22 = self.y[0] + extend[1]
            if (cID[2] == 1):
                dom31 = self.z[-1] - extend[2]
                dom32 = self.z[-1]
            elif (cID[2] == -1):
                dom31 = self.z[0]
                dom32 = self.z[0] + extend[2]

            truth1 = (dom21 <= face1[:,1+plusArg]) & (face1[:,1+plusArg] <= dom22)
            truth2 = (dom31 <= face1[:,2+plusArg]) & (face1[:,2+plusArg] <= dom32)
            self.cornerSolids[cIndex] = face1[np.where(truth1 & truth2)]
            truth1 = (dom11 <= face2[:,0+plusArg]) & (face2[:,0+plusArg] <= dom12)
            truth2 = (dom31 <= face2[:,2+plusArg]) & (face2[:,2+plusArg] <= dom32)
            self.cornerSolids[cIndex] = np.append(self.cornerSolids[cIndex],face2[np.where(truth1 & truth2)],axis=0)
            truth1 = (dom11 <= face3[:,0+plusArg]) & (face3[:,0+plusArg] <= dom12)
            truth2 = (dom21 <= face3[:,1+plusArg]) & (face3[:,1+plusArg] <= dom22)
            self.cornerSolids[cIndex] = np.append(self.cornerSolids[cIndex], face3[np.where(truth1 & truth2)],axis=0)

    def initRecieve(self):

        for neigh in self.subDomain.neighborF:
            if neigh > -1 and neigh != self.subDomain.ID:
                self.solidsAll[neigh] = {'orientID':{}}

        for neigh in self.subDomain.neighborE:
            if neigh > -1 and neigh != self.subDomain.ID:
                self.solidsAll[neigh] = {'orientID':{}}

        for neigh in self.subDomain.neighborC:
            if neigh > -1 and neigh != self.subDomain.ID:
                self.solidsAll[neigh] = {'orientID':{}}

    def fixInterface(self):

        minD = min(self.Domain.dX,self.Domain.dY,self.Domain.dZ)
        coords = [self.x,self.y,self.z]

        for fIndex in self.Orientation.faces:
            orientID = self.Orientation.faces[fIndex]['ID']
            data = np.empty((0,3))
            for procs in self.solidsAll.keys():
                for fID in self.solidsAll[procs]['orientID'].keys():
                    if fID == orientID:
                        data = np.append(data,self.solidsAll[procs]['orientID'][fID],axis=0)

            tree = KDTree(data)

            faceSolids = self.solids[np.where(self.solids[:,3]==fIndex)][:,0:3]

            self.EDT,self.visited = _fixInterfaceCalc(self,
                                                      tree,
                                                      fIndex,
                                                      self.grid.shape[self.Orientation.faces[fIndex]['argOrder'][0]],
                                                      self.Orientation.faces[fIndex]['dir'],
                                                      faceSolids,
                                                      self.EDT,
                                                      self.visited,
                                                      minD,
                                                      coords,
                                                      self.Orientation.faces[fIndex]['argOrder'])

  
    def genStats(self):
        """
        Get Inforation (non-zero min/max) of distance tranform
        """
        own = self.subDomain.ownNodesIndex
        ownEDT =  self.EDT[own[0]:own[1],
                           own[2]:own[3],
                           own[4]:own[5]]
        distVals,distCounts  = np.unique(ownEDT,return_counts=True)
        EDTData = [self.subDomain.ID,distVals,distCounts]
        EDTData = comm.gather(EDTData, root=0)
        if self.subDomain.ID == 0:
            bins = np.empty([])
            for d in EDTData:
                if d[0] == 0:
                    bins = d[1]
                else:
                    bins = np.append(bins,d[1],axis=0)
                bins = np.unique(bins)

            counts = np.zeros_like(bins,dtype=np.int64)
            for d in EDTData:
                for n in range(0,d[1].size):
                    ind = np.where(bins==d[1][n])[0][0]
                    counts[ind] = counts[ind] + d[2][n]

            stats = np.stack((bins,counts), axis = 1)
            self.minD = bins[1]
            self.maxD = bins[-1]
            distData = [self.minD,self.maxD]
            print("Minimum distance:",self.minD,"Maximum distance:",self.maxD)
        else:
            distData = None
        distData = comm.bcast(distData, root=0)
        self.minD = distData[0]
        self.maxD = distData[1]

def calcEDT(subDomain,grid,stats=False,sendClass=False):

    size = subDomain.Domain.numSubDomains
    rank = subDomain.ID

    sDEDT = EDT(Domain = subDomain.Domain, subDomain = subDomain, Orientation = subDomain.Orientation, grid = grid)
    sDComm = communication.Comm(Domain = subDomain.Domain,subDomain = subDomain,grid = grid)

    sDEDT.genLocalEDT()
    if size > 1:
        sDEDT.getBoundarySolids()
        sDEDT.getFaceSolids()
        sDEDT.getEdgeSolids()
        sDEDT.getCornerSolids()
        sDEDT.initRecieve()
        sDEDT.solidsAll = sDComm.EDTCommunication(sDEDT.solidsAll,sDEDT.faceSolids,sDEDT.edgeSolids,sDEDT.cornerSolids)
        sDEDT.fixInterface()

        ### Update EDT Buffer
        sDComm = communication.Comm(Domain = subDomain.Domain,subDomain = subDomain,grid = sDEDT.EDT)
        sDEDT.EDT = sDComm.updateBuffer()

    if stats:
        sDEDT.genStats()

    if sendClass:
        return sDEDT
    else:
        return sDEDT.EDT

