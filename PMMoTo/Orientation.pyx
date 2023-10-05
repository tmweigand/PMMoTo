import numpy as np
cimport numpy as cnp
cimport cython


cdef class cOrientation(object):
    cdef public int numFaces,numEdges,numCorners,numNeighbors
    cdef public int[26][5] directions
    def __cinit__(self):
        self.numFaces = 6
        self.numEdges = 12
        self.numCorners = 8
        self.numNeighbors = 26
        self.directions = [[-1,-1,-1,  0, 13],  #0
                           [-1,-1, 1,  1, 12],  #1
                           [-1,-1, 0,  2, 14],  #2
                           [-1, 1,-1,  3, 10],  #3
                           [-1, 1, 1,  4,  9],  #4
                           [-1, 1, 0,  5, 11],  #5
                           [-1, 0,-1,  6, 16],  #6
                           [-1, 0, 1,  7, 15],  #7
                           [-1, 0, 0,  8, 17],  #8
                           [ 1,-1,-1,  9,  4],  #9
                           [ 1,-1, 1, 10,  3],  #10
                           [ 1,-1, 0, 11,  5],  #11
                           [ 1, 1,-1, 12,  1],  #12
                           [ 1, 1, 1, 13,  0],  #13
                           [ 1, 1, 0, 14,  2],  #14
                           [ 1, 0,-1, 15,  7],  #15
                           [ 1, 0, 1, 16,  6],  #16
                           [ 1, 0, 0, 17,  8],  #17
                           [ 0,-1,-1, 18, 22],  #18
                           [ 0,-1, 1, 19, 21],  #19
                           [ 0,-1, 0, 20, 23],  #20
                           [ 0, 1,-1, 21, 19],  #21
                           [ 0, 1, 1, 22, 18],  #22
                           [ 0, 1, 0, 23, 20],  #23
                           [ 0, 0,-1, 24, 25],  #24
                           [ 0, 0, 1, 25, 24]]  #25



    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    cpdef int getBoundaryIDReference(self,cnp.ndarray[cnp.int8_t, ndim=1] boundaryID):
        """
        Determining boundary ID
        Input: boundaryID[3] corresponding to [x,y,z] and values range from [-1,0,1]
        Output: boundaryID
        """
        cdef int cI,cJ,cK
        cdef int i,j,k
        i = boundaryID[0]
        j = boundaryID[1]
        k = boundaryID[2]

        if i < 0:
            cI = 0
        elif i > 0:
            cI = 9
        else:
            cI = 18

        if j < 0:
            cJ = 0
        elif j > 0:
            cJ = 3
        else:
            cJ = 6

        if k < 0:
            cK = 0
        elif k > 0:
            cK = 1
        else:
            cK = 2

        return cI+cJ+cK


class Orientation(object):
    """
    Orientation of the voxels broken into face, edge, and corner neighbors
    """
    def __init__(self):
        self.numFaces = 6
        self.numEdges = 12
        self.numCorners = 8
        self.numNeighbors = 26

        self.sendFSlices = np.empty([self.numFaces,3],dtype=object)
        self.recvFSlices = np.empty([self.numFaces,3],dtype=object)
        self.sendESlices = np.empty([self.numEdges,3],dtype=object)
        self.recvESlices = np.empty([self.numEdges,3],dtype=object)
        self.sendCSlices = np.empty([self.numCorners,3],dtype=object)
        self.recvCSlices = np.empty([self.numCorners,3],dtype=object)


        self.faces=  {0:{'ID':(-1, 0, 0),'oppIndex':1, 'argOrder':np.array([0,1,2],dtype=np.uint8), 'dir': 1},
                      1:{'ID':( 1, 0, 0),'oppIndex':0, 'argOrder':np.array([0,1,2],dtype=np.uint8), 'dir':-1},
                      2:{'ID':( 0,-1, 0),'oppIndex':3, 'argOrder':np.array([1,0,2],dtype=np.uint8), 'dir': 1},
                      3:{'ID':( 0, 1, 0),'oppIndex':2, 'argOrder':np.array([1,0,2],dtype=np.uint8), 'dir':-1},
                      4:{'ID':( 0, 0,-1),'oppIndex':5, 'argOrder':np.array([2,0,1],dtype=np.uint8), 'dir': 1},
                      5:{'ID':( 0, 0, 1),'oppIndex':4, 'argOrder':np.array([2,0,1],dtype=np.uint8), 'dir':-1}
                      }
        
        self.edges = {0 :{'ID':(-1, 0,-1), 'oppIndex':5, 'faceIndex':(0,4), 'dir':(0,2)},
                      1 :{'ID':(-1, 0, 1), 'oppIndex':4, 'faceIndex':(0,5), 'dir':(0,2)},
                      2 :{'ID':(-1,-1, 0), 'oppIndex':7, 'faceIndex':(0,2), 'dir':(0,1)},
                      3 :{'ID':(-1, 1, 0), 'oppIndex':6, 'faceIndex':(0,3), 'dir':(0,1)},
                      4 :{'ID':( 1, 0,-1), 'oppIndex':1, 'faceIndex':(1,4), 'dir':(0,2)},
                      5 :{'ID':( 1, 0, 1), 'oppIndex':0, 'faceIndex':(1,5), 'dir':(0,2)},
                      6 :{'ID':( 1,-1, 0), 'oppIndex':3, 'faceIndex':(1,2), 'dir':(0,1)},
                      7 :{'ID':( 1, 1, 0), 'oppIndex':2, 'faceIndex':(1,3), 'dir':(0,1)},
                      8 :{'ID':( 0,-1,-1), 'oppIndex':11,'faceIndex':(2,4), 'dir':(1,2)},
                      9 :{'ID':( 0,-1, 1), 'oppIndex':10,'faceIndex':(2,5), 'dir':(1,2)},
                      10:{'ID':( 0, 1,-1), 'oppIndex':9, 'faceIndex':(3,4), 'dir':(1,2)},
                      11:{'ID':( 0, 1, 1), 'oppIndex':8, 'faceIndex':(3,5), 'dir':(1,2)},
                    }

        self.corners = {0:{'ID':(-1,-1,-1),'oppIndex':7, 'faceIndex':(0,2,4), 'edgeIndex':(0,2,8)},
                        1:{'ID':(-1,-1, 1),'oppIndex':6, 'faceIndex':(0,2,5), 'edgeIndex':(1,2,9)},
                        2:{'ID':(-1, 1,-1),'oppIndex':5, 'faceIndex':(0,3,4), 'edgeIndex':(0,3,10)},
                        3:{'ID':(-1, 1, 1),'oppIndex':4, 'faceIndex':(0,3,5), 'edgeIndex':(1,3,11)},
                        4:{'ID':( 1,-1,-1),'oppIndex':3, 'faceIndex':(1,2,4), 'edgeIndex':(4,6,8)}, 
                        5:{'ID':( 1,-1, 1),'oppIndex':2, 'faceIndex':(1,2,5), 'edgeIndex':(5,6,9)},
                        6:{'ID':( 1, 1,-1),'oppIndex':1, 'faceIndex':(1,3,4), 'edgeIndex':(4,7,10)}, 
                        7:{'ID':( 1, 1, 1),'oppIndex':0, 'faceIndex':(1,3,5), 'edgeIndex':(5,7,11)}
                        }
        

        self.directions ={0 :{'ID':[-1,-1,-1],'index': 0 ,'oppIndex': 25},
                          1 :{'ID':[-1,-1,0], 'index': 1 ,'oppIndex': 24},
                          2 :{'ID':[-1,-1,1], 'index': 2 ,'oppIndex': 23},
                          3 :{'ID':[-1,0,-1], 'index': 3 ,'oppIndex': 22},
                          4 :{'ID':[-1,0,0],  'index': 4 ,'oppIndex': 21},
                          5 :{'ID':[-1,0,1],  'index': 5 ,'oppIndex': 20},
                          6 :{'ID':[-1,1,-1], 'index': 6 ,'oppIndex': 19},
                          7 :{'ID':[-1,1,0],  'index': 7 ,'oppIndex': 18},
                          8 :{'ID':[-1,1,1],  'index': 8 ,'oppIndex': 17},
                          9 :{'ID':[0,-1,-1], 'index': 9 ,'oppIndex': 16},
                          10:{'ID':[0,-1,0],  'index': 10 ,'oppIndex': 15},
                          11:{'ID':[0,-1,1],  'index': 11 ,'oppIndex': 14},
                          12:{'ID':[0,0,-1],  'index': 12 ,'oppIndex': 13},
                          13:{'ID':[0,0,1],   'index': 13 ,'oppIndex': 12},
                          14:{'ID':[0,1,-1],  'index': 14 ,'oppIndex': 11},
                          15:{'ID':[0,1,0],   'index': 15 ,'oppIndex': 10},
                          16:{'ID':[0,1,1],   'index': 16 ,'oppIndex': 9},
                          17:{'ID':[1,-1,-1], 'index': 17 ,'oppIndex': 8},
                          18:{'ID':[1,-1,0],  'index': 18 ,'oppIndex': 7},
                          19:{'ID':[1,-1,1],  'index': 19 ,'oppIndex': 6},
                          20:{'ID':[1,0,-1],  'index': 20 ,'oppIndex': 5},
                          21:{'ID':[1,0,0],   'index': 21 ,'oppIndex': 4},
                          22:{'ID':[1,0,1],   'index': 22 ,'oppIndex': 3},
                          23:{'ID':[1,1,-1],  'index': 23 ,'oppIndex': 2},
                          24:{'ID':[1,1,0],   'index': 24 ,'oppIndex': 1},
                          25:{'ID':[1,1,1],   'index': 25 ,'oppIndex': 0},
                         }
        ### Faces/Edges for Faces,Edges,Corners ###
        self.allFaces = [[0, 2, 6, 8, 18, 20, 24],         # 0
                         [1, 2, 7, 8, 19, 20, 25],         # 1
                         [2, 8, 20],                       # 2
                         [3, 5, 6, 8, 21, 23, 24],         # 3
                         [4, 5, 7, 8, 22, 23, 25],         # 4
                         [5, 8, 23],                       # 5
                         [6, 8, 24],                       # 6
                         [7, 8, 25],                       # 7
                         [8],                              # 8
                          [9, 11, 15, 17, 18, 20, 24],      # 9
                         [10, 11, 16, 17, 19, 20, 25],     # 10
                         [11, 17, 20],                     # 11
                         [12, 14, 15, 17, 21, 23, 24],     # 12
                         [13, 14, 16, 17, 22, 23, 25],     # 13
                         [14, 17, 23],                     # 14
                         [15, 17, 24],                     # 15
                         [16, 17, 25],                     # 16
                         [17],                             # 17
                         [18, 20, 24],                     # 18
                         [19, 20, 25],                     # 19
                         [20],                             # 20
                         [21, 23, 24],                     # 21
                         [22, 23, 25],                     # 22
                         [23],                             # 23
                         [24],                             # 24
                         [25]]                             # 25

    def get_index_ordering(self,inlet,outlet):
        """
        This function rearranges the loopInfo ordering so
        the inlet and outlet faces are first. 
        """
        order = [0,1,2]
        for n in range(0,3):
            if inlet[n*2] or outlet[n*2] or inlet[n*2+1] or outlet[n*2+1]:
                order.remove(n);
                order.insert(0,n)
        
        return order


    def getLoopInfo(self,grid,subDomain,inlet,outlet,resPad):
        """
        Grap  Loop Information to Cycle through the Boundary Faces and Internal Nodes
        Reservois are Treated as Entire Face  
        Order ensure that inlet/outlet edges and corners are included in optimized looping 
        """

        order = self.get_index_ordering(inlet,outlet)

        loopInfo = np.zeros([self.numFaces+1,3,2],dtype = np.int64)

        rangeInfo = 2*np.ones([6],dtype=np.uint8)
        for fIndex in self.faces:
            if subDomain.boundaryID[fIndex] == 0:
                rangeInfo[fIndex] = rangeInfo[fIndex] - 1
            if inlet[fIndex] > 0:
                rangeInfo[fIndex] = rangeInfo[fIndex] + resPad
            if outlet[fIndex] > 0:
                rangeInfo[fIndex] = rangeInfo[fIndex] + resPad

        for fIndex in self.faces:
            face = self.faces[fIndex]['argOrder'][0]

            if self.faces[fIndex]['dir'] == -1:
                if face == order[0]:
                    loopInfo[fIndex,order[0]] = [grid.shape[order[0]]-rangeInfo[order[0]*2+1],grid.shape[order[0]]]
                    loopInfo[fIndex,order[1]] = [0,grid.shape[order[1]]]
                    loopInfo[fIndex,order[2]] = [0,grid.shape[order[2]]]
                elif face == order[1]:
                    loopInfo[fIndex,order[0]] = [rangeInfo[order[0]*2],grid.shape[order[0]]-rangeInfo[order[0]*2+1]]
                    loopInfo[fIndex,order[1]] = [grid.shape[order[1]]-rangeInfo[order[1]*2+1],grid.shape[order[1]]]
                    loopInfo[fIndex,order[2]] = [0,grid.shape[order[2]]]
                elif face == order[2]:
                    loopInfo[fIndex,order[0]] = [rangeInfo[order[0]*2],grid.shape[order[0]]-rangeInfo[order[0]*2+1]]
                    loopInfo[fIndex,order[1]] = [rangeInfo[order[1]*2],grid.shape[order[1]]-rangeInfo[order[1]*2+1]]
                    loopInfo[fIndex,order[2]] = [grid.shape[order[2]]-rangeInfo[order[2]*2+1],grid.shape[order[2]]]

            elif self.faces[fIndex]['dir'] == 1:
                if face == order[0]:
                    loopInfo[fIndex,order[0]] = [0,rangeInfo[order[0]*2]]
                    loopInfo[fIndex,order[1]] = [0,grid.shape[order[1]]]
                    loopInfo[fIndex,order[2]] = [0,grid.shape[order[2]]]
                elif face == order[1]:
                    loopInfo[fIndex,order[0]] = [rangeInfo[order[0]*2],grid.shape[order[0]]-rangeInfo[order[0]*2+1]]
                    loopInfo[fIndex,order[1]] = [0,rangeInfo[order[1]*2]]
                    loopInfo[fIndex,order[2]] = [0,grid.shape[order[2]]]
                elif face == order[2]:
                    loopInfo[fIndex,order[0]] = [rangeInfo[order[0]*2],grid.shape[order[0]]-rangeInfo[order[0]*2+1]]
                    loopInfo[fIndex,order[1]] = [rangeInfo[order[1]*2],grid.shape[order[1]]-rangeInfo[order[1]*2+1]]
                    loopInfo[fIndex,order[2]] = [0,rangeInfo[order[2]*2]]

        loopInfo[self.numFaces][order[0]] = [rangeInfo[order[0]*2],grid.shape[order[0]]-rangeInfo[order[0]*2+1]]
        loopInfo[self.numFaces][order[1]] = [rangeInfo[order[1]*2],grid.shape[order[1]]-rangeInfo[order[1]*2+1]]
        loopInfo[self.numFaces][order[2]] = [rangeInfo[order[2]*2],grid.shape[order[2]]-rangeInfo[order[2]*2+1]]
        
        return loopInfo

    def getSendSlices(self,structRatio,buffer,dim):
        """
        Determine slices of face, edge, and corner neighbor to send data 
        structRatio is size of voxel window to send and is [nx,ny,nz]
        buffer is the subDomain.buffer
        dim is grid.shape
        Buffer is always updated on edges and corners due to geometry contraints
        """

        #############
        ### Faces ###
        #############
        for fIndex in self.faces:
            fID = self.faces[fIndex]['ID']
            for n in range(len(fID)):
                if fID[n] != 0:
                    if fID[n] > 0:
                        self.sendFSlices[fIndex,n] = slice(dim[n]-structRatio[n*2+1]-buffer[n*2+1]-1,dim[n]-buffer[n*2+1]-1)
                    else:
                        self.sendFSlices[fIndex,n] = slice(buffer[n*2]+1,buffer[n*2]+structRatio[n*2]+1)
                else:
                    self.sendFSlices[fIndex,n] = slice(None,None)
        #############

        #############
        ### Edges ###
        #############
        for eIndex in self.edges:
            eID = self.edges[eIndex]['ID']
            for n in range(len(eID)):
                if eID[n] != 0:
                    if eID[n] > 0:
                        self.sendESlices[eIndex,n] = slice(dim[n]-structRatio[n*2+1]-buffer[n*2+1]-1,dim[n]-1)
                    else:
                        self.sendESlices[eIndex,n] = slice(buffer[n*2],buffer[n*2]+structRatio[n*2]+1)
                else:
                    self.sendESlices[eIndex,n] = slice(None,None)
        #############

        ###############
        ### Corners ###
        ###############
        for cIndex in self.corners:
            cID = self.corners[cIndex]['ID']
            for n in range(len(cID)):
                if cID[n] > 0:
                    self.sendCSlices[cIndex,n] = slice(dim[n]-structRatio[n*2+1]-buffer[n*2+1]-1,dim[n]-1)
                else:
                    self.sendCSlices[cIndex,n] = slice(buffer[n*2],buffer[n*2]+structRatio[n*2]+1)
        ###############

    def getSendBufferSlices(self,buffer,dim):
        """
        Determine slices of face, edge, and corner neighbor to send data 
        structRatio is size of voxel window to send and is [nx,ny,nz]
        buffer is the subDomain.buffer
        dim is grid.shape
        Buffer is always updated on edges and corners due to geometry contraints
        """

        #############
        ### Faces ###
        #############
        for fIndex in self.faces:
            fID = self.faces[fIndex]['ID']
            for n in range(len(fID)):
                if fID[n] != 0:
                    if fID[n] > 0:
                        self.sendFSlices[fIndex,n] = slice(dim[n]-2*buffer[n*2+1],dim[n]-buffer[n*2+1])
                    else:
                        self.sendFSlices[fIndex,n] = slice(buffer[n*2],2*buffer[n*2])
                else:
                    self.sendFSlices[fIndex,n] = slice(buffer[n*2],dim[n]-buffer[n*2+1])
        #############

        #############
        ### Edges ###
        #############
        for eIndex in self.edges:
            eID = self.edges[eIndex]['ID']
            for n in range(len(eID)):
                if eID[n] != 0:
                    if eID[n] > 0:
                        self.sendESlices[eIndex,n] = slice(dim[n]-2*buffer[n*2+1],dim[n]-buffer[n*2+1])
                    else:
                        self.sendESlices[eIndex,n] = slice(buffer[n*2],2*buffer[n*2])
                else:
                    self.sendESlices[eIndex,n] = slice(buffer[n*2],dim[n]-buffer[n*2+1])
        #############

        ###############
        ### Corners ###
        ###############
        for cIndex in self.corners:
            cID = self.corners[cIndex]['ID']
            for n in range(len(cID)):
                if cID[n] > 0:
                    self.sendCSlices[cIndex,n] = slice(dim[n]-2*buffer[n*2+1],dim[n]-buffer[n*2+1])
                else:
                    self.sendCSlices[cIndex,n] = slice(buffer[n*2],2*buffer[n*2])
        ###############


    def getRecieveSlices(self,halo,buffer,dim):
        """
        Determine slices of face, edge, and corner neighbor to recieve data 
        Buffer is always updated on edges and corners due to geometry contraints
        """

        #############
        ### Faces ###
        #############
        for fIndex in self.faces:
            fID = self.faces[fIndex]['ID']
            for n in range(len(fID)):
                if fID[n] != 0:
                    if fID[n] > 0:
                        self.recvFSlices[fIndex,n] = slice(dim[n]-halo[n*2+1],dim[n])
                    else:
                        self.recvFSlices[fIndex,n] = slice(None,halo[n*2])
                else:
                    self.recvFSlices[fIndex,n] = slice(halo[n*2],dim[n]-halo[n*2+1])
        #############

        #############
        ### Edges ###
        #############
        for eIndex in self.edges:
            eID = self.edges[eIndex]['ID']
            for n in range(len(eID)):
                if eID[n] != 0:
                    if eID[n] > 0:
                        self.recvESlices[eIndex,n] = slice(dim[n]-halo[n*2+1]-buffer[n*2+1],dim[n])
                    else:
                        self.recvESlices[eIndex,n] = slice(None,halo[n*2]+buffer[n*2])
                else:
                    self.recvESlices[eIndex,n] = slice(halo[n*2],dim[n]-halo[n*2+1])
        #############

        ###############
        ### Corners ###
        ###############
        for cIndex in self.corners:
            cID = self.corners[cIndex]['ID']
            for n in range(len(cID)):
                if cID[n] > 0:
                    self.recvCSlices[cIndex,n] = slice(dim[n]-halo[n*2+1]-buffer[n*2+1],dim[n])
                else:
                    self.recvCSlices[cIndex,n] = slice(None,halo[n*2]+buffer[n*2])
        ###############

    def getRecieveBufferSlices(self,buffer,dim):
        """
        Determine slices of face, edge, and corner neighbor to recieve data 
        Buffer is always updated on edges and corners due to geometry contraints
        """

        #############
        ### Faces ###
        #############
        for fIndex in self.faces:
            fID = self.faces[fIndex]['ID']
            for n in range(len(fID)):
                if fID[n] != 0:
                    if fID[n] > 0:
                        self.recvFSlices[fIndex,n] = slice(dim[n]-buffer[n*2+1],dim[n])
                    else:
                        self.recvFSlices[fIndex,n] = slice(None,buffer[n*2])
                else:
                    self.recvFSlices[fIndex,n] = slice(buffer[n*2],dim[n]-buffer[n*2+1])
        #############

        #############
        ### Edges ###
        #############
        for eIndex in self.edges:
            eID = self.edges[eIndex]['ID']
            for n in range(len(eID)):
                if eID[n] != 0:
                    if eID[n] > 0:
                        self.recvESlices[eIndex,n] = slice(dim[n]-buffer[n*2+1],dim[n])
                    else:
                        self.recvESlices[eIndex,n] = slice(None,buffer[n*2])
                else:
                    self.recvESlices[eIndex,n] = slice(buffer[n*2],dim[n]-buffer[n*2+1])
        #############

        ###############
        ### Corners ###
        ###############
        for cIndex in self.corners:
            cID = self.corners[cIndex]['ID']
            for n in range(len(cID)):
                if cID[n] > 0:
                    self.recvCSlices[cIndex,n] = slice(dim[n]-buffer[n*2+1],dim[n])
                else:
                    self.recvCSlices[cIndex,n] = slice(None,buffer[n*2])
        ###############
