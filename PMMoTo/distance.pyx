import numpy as np
cimport numpy as cnp
cnp.import_array()
from mpi4py import MPI
from pykdtree.kdtree import KDTree
### if using WSL, uncomment line below. 
#from scipy.spatial import KDTree

import edt
from . import Orientation
from . import communication
from . import nodes
cimport cython

comm = MPI.COMM_WORLD

""" Solid = 0, Pore = 1 """

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def _fixInterfaceCalc(tree,
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


class EDT(object):
    def __init__(self,subdomain,grid):
        self.subdomain = subdomain
        self.extendFactor  = 0.7
        self.grid = grid
        self.EDT = np.zeros_like(grid)
        self.solids = None
        self.faceSolids = []
        self.edgeSolids = []
        self.cornerSolids = []
        self.external_solids = {key: None for key in Orientation.features}
        self.distVals = None
        self.distCounts  = None
        self.minD = 0
        self.maxD = 0

    def genLocalEDT(self):
        """
        Determine the Euclidian distance on each process knowing the values may be too high
        """
        self.EDT = edt.edt3d(self.grid, anisotropy=self.subdomain.domain.voxel)


    def partition_boundary_solids(self):
        """
        Trim solids to minimize communication and reduce KD Tree. Identify on Surfaces, Edges, and Corners
        Keep all face solids, and use extend factor to query which solids to include for edges and corners
        """
        face_solids = [[] for _ in range(len(Orientation.faces))]
        edge_solids = [[] for _ in range(len(Orientation.edges))]
        corner_solids = [[] for _ in range(len(Orientation.corners))]
        
        extend = [self.extendFactor*x for x in self.subdomain.size_subdomain]
        coords = self.subdomain.coords

        ### Faces ###
        for fIndex in Orientation.faces:
            pointsXYZ = []
            points = self.solids[np.where( (self.solids[:,0]>-1)
                                 & (self.solids[:,1]>-1)
                                 & (self.solids[:,2]>-1)
                                 & ((self.solids[:,3]==fIndex)) )][:,0:3]
            for x,y,z in points:
                pointsXYZ.append([coords[0][x],coords[1][y],coords[2][z]] )
            face_solids[fIndex] = np.asarray(pointsXYZ)

        ### Edges ###
        for edge in self.subdomain.edges:
            edge.get_extension(extend,self.subdomain.bounds)
            for f,d in zip(edge.info['faceIndex'],reversed(edge.info['dir'])): # Flip dir for correct nodes
                f_solids = face_solids[f]
                values = (edge.extend[d][0] <= f_solids[:,d]) & (f_solids[:,d] <= edge.extend[d][1])
                if len(edge_solids[edge.ID]) == 0:
                    edge_solids[edge.ID] = f_solids[np.where(values)]
                else:
                    edge_solids[edge.ID] = np.append(edge_solids[edge.ID],f_solids[np.where(values)],axis=0)
            edge_solids[edge.ID] = np.unique(edge_solids[edge.ID],axis=0)

        ### Corners ###
        iter = [[1,2],[0,2],[0,1]]
        for corner in self.subdomain.corners:
            corner.get_extension(extend,self.subdomain.bounds)
            values = [None,None]
            for it,f in zip(iter,corner.info['faceIndex']):
                f_solids = face_solids[f]
                for n,i in enumerate(it):
                    values[n] = (corner.extend[i][0] <= f_solids[:,i]) & (f_solids[:,i] <= corner.extend[i][1])
                if len(corner_solids[corner.ID]) == 0:
                    corner_solids[corner.ID] = f_solids[np.where(values[0] & values[1])]
                else:
                    corner_solids[corner.ID] = np.append(corner_solids[corner.ID],f_solids[np.where(values[0] & values[1])],axis=0)
            corner_solids[corner.ID] = np.unique(corner_solids[corner.ID],axis=0)

        return face_solids,edge_solids,corner_solids

    def fixInterface(self):
        """
        Loop through faces and correct distance with external_solids
        """
        visited = np.zeros_like(self.grid,dtype=np.uint8)

        for face in self.subdomain.faces:
            if face.n_proc > -1:
                arg = face.info['argOrder']
                data = np.copy(self.external_solids[face.info['ID']])

                for edge in self.subdomain.edges:
                    for f in edge.info['faceIndex']:
                        if f == face.ID and edge.n_proc > -1: 
                            data = np.append(data,self.external_solids[edge.info['ID']],axis=0)

                for corner in self.subdomain.corners:
                    for f in corner.info['faceIndex']:
                        if f == face.ID and corner.n_proc > -1: 
                            data = np.append(data,self.external_solids[corner.info['ID']],axis=0)

                tree = KDTree(data)
                face_solids = self.solids[np.where(self.solids[:,3]==face.ID)][:,0:3]
                self.EDT,visited = _fixInterfaceCalc(tree,
                                                     face.ID,
                                                     self.grid.shape[arg[0]],
                                                     face.info['dir'],
                                                     face_solids,
                                                     self.EDT,
                                                     visited,
                                                     min(self.subdomain.domain.voxel),
                                                     self.subdomain.coords,
                                                     arg)

  
    def genStats(self):
        """
        Get Information (non-zero min/max) of distance tranform
        """
        own = self.subdomain.index_own_nodes
        ownEDT =  self.EDT[own[0]:own[1],
                           own[2]:own[3],
                           own[4]:own[5]]
        distVals,distCounts  = np.unique(ownEDT,return_counts=True)
        EDTData = [self.subdomain.ID,distVals,distCounts]
        EDTData = comm.gather(EDTData, root=0)
        if self.subdomain.ID == 0:
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

def calcEDT(subdomain,grid,stats = False,sendClass = False):
    size = subdomain.domain.num_subdomains
    sDEDT = EDT(subdomain = subdomain, grid = grid)
    sDEDT.genLocalEDT()
    if size > 1 or (size == 1 and any(subdomain.boundary_ID == 2)):
        sDEDT.solids = nodes.get_boundary_nodes(grid,0)
        face_solids,edge_solids,corner_solids = sDEDT.partition_boundary_solids()
        sDEDT.external_solids = communication.pass_external_data(subdomain,face_solids,edge_solids,corner_solids)
        sDEDT.fixInterface()
        sDEDT.EDT = communication.update_buffer(subdomain,sDEDT.EDT)
    if stats:
        sDEDT.genStats()
    if sendClass:
        return sDEDT
    else:
        return sDEDT.EDT

