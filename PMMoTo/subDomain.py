import numpy as np
from mpi4py import MPI
from .domainGeneration import domainGenINK
from .domainGeneration import domainGen
from . import communication
import sys
import pdb
comm = MPI.COMM_WORLD

""" Solid = 0, Pore = 1 """

""" TO DO:
           Create Orientationa and Domains files. 
           Switch to pass periodic info and not generate from samples??
           Redo Domain decomposition - Maybe
"""

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

        self.faces=  {0:{'ID':(1,0,0), 'Index':1, 'oppIndex':1, 'argOrder':np.array([0,1,2],dtype=np.uint8), 'dir':-1},
                      1:{'ID':(-1,0,0),'Index':0, 'oppIndex':0, 'argOrder':np.array([0,1,2],dtype=np.uint8), 'dir':1},
                      2:{'ID':(0,1,0), 'Index':1, 'oppIndex':3, 'argOrder':np.array([1,0,2],dtype=np.uint8), 'dir':-1},
                      3:{'ID':(0,-1,0),'Index':0, 'oppIndex':2, 'argOrder':np.array([1,0,2],dtype=np.uint8), 'dir':1},
                      4:{'ID':(0,0,1), 'Index':1, 'oppIndex':5, 'argOrder':np.array([2,0,1],dtype=np.uint8), 'dir':-1},
                      5:{'ID':(0,0,-1),'Index':0, 'oppIndex':4, 'argOrder':np.array([2,0,1],dtype=np.uint8), 'dir':1},
                      }
        self.edges = {0 :{'ID':(1,1,0),  'oppIndex':5, 'faceIndex':(0,2), 'dir':(0,1)},
                      1 :{'ID':(1,-1,0), 'oppIndex':4, 'faceIndex':(0,3), 'dir':(0,1)},
                      2 :{'ID':(1,0,1),  'oppIndex':7, 'faceIndex':(0,4), 'dir':(0,2)},
                      3 :{'ID':(1,0,-1), 'oppIndex':6, 'faceIndex':(0,5), 'dir':(0,2)},
                      4 :{'ID':(-1,1,0), 'oppIndex':1, 'faceIndex':(1,2), 'dir':(0,1)},
                      5 :{'ID':(-1,-1,0),'oppIndex':0, 'faceIndex':(1,3), 'dir':(0,1)},
                      6 :{'ID':(-1,0,1), 'oppIndex':3, 'faceIndex':(1,4), 'dir':(0,2)},
                      7 :{'ID':(-1,0,-1),'oppIndex':2, 'faceIndex':(1,5), 'dir':(0,2)},
                      8 :{'ID':(0,1, 1), 'oppIndex':11,'faceIndex':(2,4), 'dir':(1,2)},
                      9 :{'ID':(0,1,-1), 'oppIndex':10,'faceIndex':(2,5), 'dir':(1,2)},
                      10:{'ID':(0,-1,1), 'oppIndex':9, 'faceIndex':(3,4), 'dir':(1,2)},
                      11:{'ID':(0,-1,-1),'oppIndex':8, 'faceIndex':(3,5), 'dir':(1,2)},
                       }
        self.corners = {0:{'ID':(1,1,1),   'oppIndex':7, 'faceIndex':(0,2,4)},
                        1:{'ID':(1,1,-1),  'oppIndex':6, 'faceIndex':(0,2,5)},
                        2:{'ID':(1,-1,1),  'oppIndex':5, 'faceIndex':(0,3,4)},
                        3:{'ID':(1,-1,-1), 'oppIndex':4, 'faceIndex':(0,3,5)},
                        4:{'ID':(-1,1,1),  'oppIndex':3, 'faceIndex':(1,2,4)},
                        5:{'ID':(-1,1,-1), 'oppIndex':2, 'faceIndex':(1,2,5)},
                        6:{'ID':(-1,-1,1), 'oppIndex':1, 'faceIndex':(1,3,4)},
                        7:{'ID':(-1,-1,-1),'oppIndex':0, 'faceIndex':(1,3,5)},
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

    def getSendSlices(self,structRatio,buffer):
        """
        Determine slices of face, edge, and corner neighbor to send data 
        structRatio is size of voxel window to send
        buffer is XXX
        """

        #############
        ### Faces ###
        #############
        for fIndex in self.faces:
            fID = self.faces[fIndex]['ID']
            for n in range(len(fID)):
                if fID[n] != 0:
                    if fID[n] > 0:
                        buf = None
                        if buffer[n][1] > 0:
                            buf = -buffer[n][1]*2
                        self.sendFSlices[fIndex,n] = slice(-structRatio[n]-buffer[n][1]*2,buf)
                    else:
                        buf = None
                        if buffer[n][0] > 0:
                            buf = buffer[n][0]*2
                        self.sendFSlices[fIndex,n] = slice(buf,structRatio[n]+buffer[n][0]*2)
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
                        buf = None
                        if buffer[n][1] > 0:
                            buf = -buffer[n][1]*2
                        self.sendESlices[eIndex,n] = slice(-structRatio[n]-buffer[n][1]*2,buf)
                    else:
                        buf = None
                        if buffer[n][0] > 0:
                            buf = buffer[n][0]*2
                        self.sendESlices[eIndex,n] = slice(buf,structRatio[n]+buffer[n][0]*2)
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
                    buf = None
                    if buffer[n][1] > 0:
                        buf = -buffer[n][1]*2
                    self.sendCSlices[cIndex,n] = slice(-structRatio[n]-buffer[n][1]*2,buf)
                else:
                    buf = None
                    if buffer[n][0] > 0:
                        buf = buffer[n][0]*2
                    self.sendCSlices[cIndex,n] = slice(buf,structRatio[n]+buffer[n][0]*2)
        ###############

    def getRecieveSlices(self,structRatio,pad,arr):
        """
        Determine slices of face, edge, and corner neighbor to recieve data 
        structRatio is 
        pad is XXX
        arr is 
        """

        dim = arr.shape
        if pad.shape != [3,2]:
            pad = pad.reshape([3,2])

        #############
        ### Faces ###
        #############
        for fIndex in self.faces:
            fID = self.faces[fIndex]['ID']
            for n in range(len(fID)):
                if fID[n] != 0:
                    if fID[n] > 0:
                        self.recvFSlices[fIndex,n] = slice(-structRatio[n],None)
                    else:
                        self.recvFSlices[fIndex,n] = slice(None,structRatio[n])
                else:
                    self.recvFSlices[fIndex,n] = slice(0+pad[n,1],dim[n]-pad[n,0])
        #############

        #############
        ### Edges ###
        #############
        for eIndex in self.edges:
            eID = self.edges[eIndex]['ID']
            for n in range(len(eID)):
                if eID[n] != 0:
                    if eID[n] > 0:
                        self.recvESlices[eIndex,n] = slice(-structRatio[n],None)
                    else:
                        self.recvESlices[eIndex,n] = slice(None,structRatio[n])
                else:
                    self.recvESlices[eIndex,n] = slice(0+pad[n,1],dim[n]-pad[n,0])
        #############

        ###############
        ### Corners ###
        ###############
        for cIndex in self.corners:
            cID = self.corners[cIndex]['ID']
            for n in range(len(cID)):
                if cID[n] > 0:
                    self.recvCSlices[cIndex,n] = slice(-structRatio[n],None)
                else:
                    self.recvCSlices[cIndex,n] = slice(None,structRatio[n])
        ###############

class Domain(object):
    """
    Determine information for entire Domain
    """
    def __init__(self,nodes,domainSize,subDomains,boundaries,inlet=[0,0,0],outlet=[0,0,0]):
        self.nodes        = nodes
        self.domainSize   = domainSize
        self.boundaries   = boundaries
        self.subDomains   = subDomains
        self.numSubDomains = np.prod(subDomains)
        self.subNodes     = np.zeros([3],dtype=np.uint64)
        self.subNodesRem  = np.zeros([3],dtype=np.uint64)
        self.domainLength = np.zeros([3])
        self.inlet = inlet
        self.outlet = outlet
        self.dX = 0
        self.dY = 0
        self.dZ = 0
        self.inputChecks()

    def getdXYZ(self):
        """
        Get domain length and voxel size
        """
        self.domainLength[0] = (self.domainSize[0,1]-self.domainSize[0,0])
        self.domainLength[1] = (self.domainSize[1,1]-self.domainSize[1,0])
        self.domainLength[2] = (self.domainSize[2,1]-self.domainSize[2,0])
        self.dX = self.domainLength[0]/self.nodes[0]
        self.dY = self.domainLength[1]/self.nodes[1]
        self.dZ = self.domainLength[2]/self.nodes[2]

    def getSubNodes(self):
        """
        Calculate number of voxels in each subDomain
        """
        self.subNodes[0],self.subNodesRem[0] = divmod(self.nodes[0],self.subDomains[0])
        self.subNodes[1],self.subNodesRem[1] = divmod(self.nodes[1],self.subDomains[1])
        self.subNodes[2],self.subNodesRem[2] = divmod(self.nodes[2],self.subDomains[2])

    def inputChecks(self):
        """
        Ensure Input Parameters are Valid
        """
        error = False

        ### Check Nodes are Positive
        for n in self.nodes:
            if n <= 0:
                error = True
                print("Error: Nodes must be positive integer!")
        
        ### Check subDomain Size
        for n in self.subDomains:
            if n <= 0:
                error = True
                print("Error: Number of Subdomains must be positive integer!")

        ### Check Boundaries
        for n in self.boundaries:
            if n < 0 or n > 2:
                error = True
                print("Error: Allowable Boundary IDs are (0) None (1) Walls (2) Periodic")

        ### Check Inlet Condition
        sN = 0
        for nI,nB in zip(self.inlet,self.boundaries):
            if nI !=0:
                sN = sN + 1
                if nB != 0:
                    error = True
                    print("Error: Boundary must be type (0) None at Inlet")
        if nI > 0:
            error = True
            print("Error: Only 1 Inlet Allowed")

        ### Check Outlet Condition
        sN = 0
        for nI,nB in zip(self.outlet,self.boundaries):
            if nI !=0:
                sN = sN + 1
                if nB != 0:
                    error = True
                    print("Error: Boundary must be type (0) None at Outlet")
        if nI > 0:
            error = True
            print("Error: Only 1 Outlet Allowed")  

        if error:
          communication.raiseError()


class subDomain(object):
    def __init__(self,ID,subDomains,Domain,Orientation):
        bufferSize        = 1
        self.extendFactor = 0.7
        self.useIndex     = False
        self.ID          = ID
        self.rank        = ID
        self.size        = np.prod(subDomains)
        self.subDomains  = subDomains
        self.Domain      = Domain
        self.Orientation = Orientation
        self.boundary    = False
        self.boundaryID  = np.zeros([3,2],dtype = np.int8)
        self.nodes       = np.zeros([3],dtype=np.int64)
        self.indexStart  = np.zeros([3],dtype=np.int64)
        self.subID       = np.zeros([3],dtype=np.int64)
        self.lookUpID    = np.zeros(subDomains,dtype=np.int64)
        self.buffer      = bufferSize*np.ones([3,2],dtype = np.int64)
        self.numSubDomains = np.prod(subDomains)
        self.neighborF    = -np.ones(self.Orientation.numFaces,dtype = np.int64)
        self.neighborE    = -np.ones(self.Orientation.numEdges,dtype = np.int64)
        self.neighborC    = -np.ones(self.Orientation.numCorners,dtype = np.int64)
        self.neighborPerF =  np.zeros([self.Orientation.numFaces,3],dtype = np.int64)
        self.neighborPerE =  np.zeros([self.Orientation.numEdges,3],dtype = np.int64)
        self.neighborPerC =  np.zeros([self.Orientation.numCorners,3],dtype = np.int64)
        self.ownNodes     = np.zeros([3,2],dtype = np.int64)
        self.ownNodesTotal= 0
        self.poreNodes    = 0
        self.totalPoreNodes = np.zeros(1,dtype=np.uint64)
        self.subDomainSize = np.zeros([3,1])
        self.grid = None
        self.globalBoundary = np.zeros([self.Orientation.numFaces],dtype = np.uint8)
        self.inlet = np.zeros([self.Orientation.numFaces],dtype = np.uint8)
        self.outlet = np.zeros([self.Orientation.numFaces],dtype = np.uint8)
        self.res  = None
        self.loopInfo = np.zeros([self.Orientation.numFaces+1,3,2],dtype = np.int64)

    def getInfo(self):
        """
        Gather information for each subDomain including ID, boundary information,number of nodes, global index start
        """
        n = 0
        for i in range(0,self.subDomains[0]):
            for j in range(0,self.subDomains[1]):
                for k in range(0,self.subDomains[2]):
                    self.lookUpID[i,j,k] = n
                    if n == self.ID:
                        if (i == 0):
                            self.boundary = True
                            self.boundaryID[0][0] = -1
                        if (i == self.subDomains[0]-1):
                            self.boundary = True
                            self.boundaryID[0][1] = 1
                        if (j == 0):
                            self.boundary = True
                            self.boundaryID[1][0] = -1
                        if (j == self.subDomains[1]-1):
                            self.boundary = True
                            self.boundaryID[1][1] = 1
                        if (k == 0):
                            self.boundary = True
                            self.boundaryID[2][0] = -1
                        if (k == self.subDomains[2]-1):
                            self.boundary = True
                            self.boundaryID[2][1] = 1

                        self.subID[0] = i
                        self.subID[1] = j
                        self.subID[2] = k
                        self.nodes[0] = self.Domain.subNodes[0]
                        self.nodes[1] = self.Domain.subNodes[1]
                        self.nodes[2] = self.Domain.subNodes[2]
                        self.indexStart[0] = self.Domain.subNodes[0]*i
                        self.indexStart[1] = self.Domain.subNodes[1]*j
                        self.indexStart[2] = self.Domain.subNodes[2]*k
                        if (i == self.subDomains[0]-1):
                            self.nodes[0] += self.Domain.subNodesRem[0]
                        if (j == self.subDomains[1]-1):
                            self.nodes[1] += self.Domain.subNodesRem[1]
                        if (k == self.subDomains[2]-1):
                            self.nodes[2] += self.Domain.subNodesRem[2]
                    n = n + 1

    def getXYZ(self):
        """
        Determine actual coordinate information (x,y,z) and buffer information.
        If globalBoundary and Domain.boundary == 0, buffer is not added
        Everywhere else a buffer is added
        """

        #####################################################
        ### Determine if subDomain should not have buffer ###
        #####################################################
        if (self.subID[0] == 0 and self.Domain.boundaries[0] == 0):
            self.buffer[0][0] = 0
        if (self.subID[0] == (self.subDomains[0] - 1) and self.Domain.boundaries[0] == 0):
            self.buffer[0][1] = 0
        if (self.subID[1] == 0 and self.Domain.boundaries[1] == 0):
            self.buffer[1][0] = 0
        if (self.subID[1] == (self.subDomains[1] - 1) and self.Domain.boundaries[1] == 0):
            self.buffer[1][1] = 0
        if (self.subID[2] == 0 and self.Domain.boundaries[2] == 0):
            self.buffer[2][0] = 0
        if (self.subID[2] == (self.subDomains[2] - 1) and self.Domain.boundaries[2] == 0):
            self.buffer[2][1] = 0
        #####################################################


        ###############################
        ### Get (x,y,z) coordinates ###
        ###############################
        self.x = np.zeros([self.nodes[0] + self.buffer[0][0] + self.buffer[0][1]],dtype=np.double)
        self.y = np.zeros([self.nodes[1] + self.buffer[1][0] + self.buffer[1][1]],dtype=np.double)
        self.z = np.zeros([self.nodes[2] + self.buffer[2][0] + self.buffer[2][1]],dtype=np.double)

        c = 0
        for i in range(-self.buffer[0][0], self.nodes[0] + self.buffer[0][1]):
            self.x[c] = self.Domain.domainSize[0,0] + (self.indexStart[0] + i)*self.Domain.dX + self.Domain.dX/2
            c = c + 1

        c = 0
        for j in range(-self.buffer[1][0], self.nodes[1] + self.buffer[1][1]):
            self.y[c] = self.Domain.domainSize[1,0] + (self.indexStart[1] + j)*self.Domain.dY + self.Domain.dY/2
            c = c + 1

        c = 0
        for k in range(-self.buffer[2][0], self.nodes[2] + self.buffer[2][1]):
            self.z[c] = self.Domain.domainSize[2,0] + (self.indexStart[2] + k)*self.Domain.dZ + self.Domain.dZ/2
            c = c + 1
        ###############################

        self.ownNodesTotal = self.nodes[0] * self.nodes[1] * self.nodes[2]

        self.ownNodes[0] = [self.buffer[0][0],self.nodes[0]+self.buffer[0][0]]
        self.ownNodes[1] = [self.buffer[1][0],self.nodes[1]+self.buffer[1][0]]
        self.ownNodes[2] = [self.buffer[2][0],self.nodes[2]+self.buffer[2][0]]

        self.nodes[0] = self.nodes[0] + self.buffer[0][0] + self.buffer[0][1]
        self.nodes[1] = self.nodes[1] + self.buffer[1][0] + self.buffer[1][1]
        self.nodes[2] = self.nodes[2] + self.buffer[2][0] + self.buffer[2][1]

        if self.buffer[0][0] != 0:
            self.indexStart[0] = self.indexStart[0] - self.buffer[0][0]

        if self.buffer[1][0] != 0:
            self.indexStart[1] = self.indexStart[1] - self.buffer[1][0]

        if self.buffer[2][0] != 0:
            self.indexStart[2] = self.indexStart[2] - self.buffer[2][0]


        self.subDomainSize = [self.x[-1] - self.x[0],
                              self.y[-1] - self.y[0],
                              self.z[-1] - self.z[0]]


    def gridCheck(self):
        if (np.sum(self.grid) == np.prod(self.nodes)):
            print("This code requires at least 1 solid voxel in each subdomain. Please reorder processors!")
            sys.exit()


    def genDomainSphereData(self,sphereData):
        self.grid = domainGen(self.x,self.y,self.z,sphereData)
        self.gridCheck()

    def genDomainInkBottle(self):
        self.grid = domainGenINK(self.x,self.y,self.z)
        self.gridCheck()

    def setBoundaryConditions(self,rank):
        """
        If wall boundary conditions are specified, force solid on external boundaries
        """
        if self.boundaryID[0][0] == -1 and self.Domain.boundaries[0] == 1:
            self.grid[0,:,:] = 0
        if self.boundaryID[0][1] == 1  and self.Domain.boundaries[0] == 1:
            self.grid[-1,:,:] = 0
        if self.boundaryID[1][0] == -1 and self.Domain.boundaries[1] == 1:
            self.grid[:,0,:] = 0
        if self.boundaryID[1][1] == 1  and self.Domain.boundaries[1] == 1:
            self.grid[:,-1,:] = 0
        if self.boundaryID[2][0] == -1 and self.Domain.boundaries[2] == 1:
            self.grid[:,:,0] = 0
        if self.boundaryID[2][1] == 1  and self.Domain.boundaries[2] == 1:
            self.grid[:,:,-1] = 0

    def getLoopInfo(self):
        """
        Grap the Loop Information to Cycle through the Boundary Faces
        """
        rangeInfo = 2*np.ones([3,2],dtype=np.uint8)
        if self.boundaryID[0][0] == -1 and self.Domain.boundaries[0] == 0:
            rangeInfo[0,0] = rangeInfo[0,0] - 1
        if self.boundaryID[0][1] == 1 and self.Domain.boundaries[0] == 0:
            rangeInfo[0,1] = rangeInfo[0,1] - 1
        if self.boundaryID[1][0] == -1 and self.Domain.boundaries[1] == 0:
            rangeInfo[1,0] = rangeInfo[1,0] - 1
        if self.boundaryID[1][1] == 1 and self.Domain.boundaries[1] == 0:
            rangeInfo[1,1] = rangeInfo[1,1] - 1
        if self.boundaryID[2][0] == -1 and self.Domain.boundaries[2] == 0:
            rangeInfo[2,0] = rangeInfo[2,0] - 1
        if self.boundaryID[2][1] == 1 and self.Domain.boundaries[2] == 0:
            rangeInfo[2,1] = rangeInfo[2,1] - 1

        for fIndex in self.Orientation.faces:
            face = self.Orientation.faces[fIndex]['argOrder'][0]

            if self.Orientation.faces[fIndex]['dir'] == -1:
                if face == 0:
                    self.loopInfo[fIndex,0] = [self.grid.shape[0]-rangeInfo[0,1],self.grid.shape[0]]
                    self.loopInfo[fIndex,1] = [0,self.grid.shape[1]]
                    self.loopInfo[fIndex,2] = [0,self.grid.shape[2]]
                elif face == 1:
                    self.loopInfo[fIndex,0] = [rangeInfo[0,0],self.grid.shape[0]-rangeInfo[0,1]]
                    self.loopInfo[fIndex,1] = [self.grid.shape[1]-rangeInfo[1,1],self.grid.shape[1]]
                    self.loopInfo[fIndex,2] = [0,self.grid.shape[2]]
                elif face == 2:
                    self.loopInfo[fIndex,0] = [rangeInfo[0,0],self.grid.shape[0]-rangeInfo[0,1]]
                    self.loopInfo[fIndex,1] = [rangeInfo[1,0],self.grid.shape[1]-rangeInfo[1,1]]
                    self.loopInfo[fIndex,2] = [self.grid.shape[2]-rangeInfo[2,1],self.grid.shape[2]]

            elif self.Orientation.faces[fIndex]['dir'] == 1:
                if face == 0:
                    self.loopInfo[fIndex,0] = [0,rangeInfo[0,0]]
                    self.loopInfo[fIndex,1] = [0,self.grid.shape[1]]
                    self.loopInfo[fIndex,2] = [0,self.grid.shape[2]]
                elif face == 1:
                    self.loopInfo[fIndex,0] = [rangeInfo[0,0],self.grid.shape[0]-rangeInfo[0,1]]
                    self.loopInfo[fIndex,1] = [0,rangeInfo[1,0]]
                    self.loopInfo[fIndex,2] = [0,self.grid.shape[2]]
                elif face == 2:
                    self.loopInfo[fIndex,0] = [rangeInfo[0,0],self.grid.shape[0]-rangeInfo[0,1]]
                    self.loopInfo[fIndex,1] = [rangeInfo[1,0],self.grid.shape[1]-rangeInfo[1,1]]
                    self.loopInfo[fIndex,2] = [0,rangeInfo[2,0]]

        self.loopInfo[self.Orientation.numFaces][0] = [rangeInfo[0,0],self.grid.shape[0]-rangeInfo[0,1]]
        self.loopInfo[self.Orientation.numFaces][1] = [rangeInfo[1,0],self.grid.shape[1]-rangeInfo[1,1]]
        self.loopInfo[self.Orientation.numFaces][2] = [rangeInfo[2,0],self.grid.shape[2]-rangeInfo[2,1]]

    def getBoundaryInfo(self):

        for fIndex in self.Orientation.faces:
            face = self.Orientation.faces[fIndex]['argOrder'][0]
            fID = self.Orientation.faces[fIndex]['ID'][face]
            fI = self.Orientation.faces[fIndex]['Index']

            if self.boundaryID[face][fI] != 0:
                self.globalBoundary[fIndex] = 1
                if self.Domain.inlet[face] == fID and self.Domain.inlet[face] == self.boundaryID[face][fI]:
                    self.inlet[fIndex] = True
                elif self.Domain.outlet[face] == fID and self.Domain.outlet[face] == self.boundaryID[face][fI]:
                    self.outlet[fIndex] = True

    def getNeighbors(self):
        """
        Get the Face, Edge, and Corner Neighbors for Each Domain
        """

        lookIDPad = np.pad(self.lookUpID, ( (1, 1), (1, 1), (1, 1)), 'constant', constant_values=-1)
        lookPerI = np.zeros_like(lookIDPad)
        lookPerJ = np.zeros_like(lookIDPad)
        lookPerK = np.zeros_like(lookIDPad)

        if (self.Domain.boundaries[0] == 2):
            lookIDPad[0,:,:]  = lookIDPad[-2,:,:]
            lookIDPad[-1,:,:] = lookIDPad[1,:,:]
            lookPerI[0,:,:] = 1
            lookPerI[-1,:,:] = -1

        if (self.Domain.boundaries[1] == 2):
            lookIDPad[:,0,:]  = lookIDPad[:,-2,:]
            lookIDPad[:,-1,:] = lookIDPad[:,1,:]
            lookPerJ[:,0,:] = 1
            lookPerJ[:,-1,:] = -1

        if (self.Domain.boundaries[2] == 2):
            lookIDPad[:,:,0]  = lookIDPad[:,:,-2]
            lookIDPad[:,:,-1] = lookIDPad[:,:,1]
            lookPerK[:,:,0] = 1
            lookPerK[:,:,-1] = -1

        cc = 0
        for f in self.Orientation.faces.values():
            cx = f['ID'][0]
            cy = f['ID'][1]
            cz = f['ID'][2]
            self.neighborF[cc]      = lookIDPad[self.subID[0]+cx+1,self.subID[1]+cy+1,self.subID[2]+cz+1]
            self.neighborPerF[cc,0] = lookPerI[self.subID[0]+cx+1,self.subID[1]+cy+1,self.subID[2]+cz+1]
            self.neighborPerF[cc,1] = lookPerJ[self.subID[0]+cx+1,self.subID[1]+cy+1,self.subID[2]+cz+1]
            self.neighborPerF[cc,2] = lookPerK[self.subID[0]+cx+1,self.subID[1]+cy+1,self.subID[2]+cz+1]
            cc = cc + 1

        cc = 0
        for e in self.Orientation.edges.values():
            cx = e['ID'][0]
            cy = e['ID'][1]
            cz = e['ID'][2]
            self.neighborE[cc]      = lookIDPad[self.subID[0]+cx+1,self.subID[1]+cy+1,self.subID[2]+cz+1]
            self.neighborPerE[cc,0] = lookPerI[self.subID[0]+cx+1,self.subID[1]+cy+1,self.subID[2]+cz+1]
            self.neighborPerE[cc,1] = lookPerJ[self.subID[0]+cx+1,self.subID[1]+cy+1,self.subID[2]+cz+1]
            self.neighborPerE[cc,2] = lookPerK[self.subID[0]+cx+1,self.subID[1]+cy+1,self.subID[2]+cz+1]
            cc = cc + 1


        cc = 0
        for c in self.Orientation.corners.values():
            cx = c['ID'][0]
            cy = c['ID'][1]
            cz = c['ID'][2]
            self.neighborC[cc]      = lookIDPad[self.subID[0]+cx+1,self.subID[1]+cy+1,self.subID[2]+cz+1]
            self.neighborPerC[cc,0] = lookPerI[self.subID[0]+cx+1,self.subID[1]+cy+1,self.subID[2]+cz+1]
            self.neighborPerC[cc,1] = lookPerJ[self.subID[0]+cx+1,self.subID[1]+cy+1,self.subID[2]+cz+1]
            self.neighborPerC[cc,2] = lookPerK[self.subID[0]+cx+1,self.subID[1]+cy+1,self.subID[2]+cz+1]
            cc = cc + 1

        self.lookUpID = lookIDPad

    def getPorosity(self):
        own = self.ownNodes
        ownGrid =  self.grid[own[0][0]:own[0][1],
                             own[1][0]:own[1][1],
                             own[2][0]:own[2][1]]
        self.poreNodes = np.sum(ownGrid)


    def getReservoir(self,resInd):
        self.res = np.zeros_like(self.grid)
        if  self.boundaryID[0][0]  > 0 and self.Domain.inlet[0] > 0:
            self.res[-resInd:,:,:] = 1
        elif self.boundaryID[0][1] < 0 and self.Domain.inlet[0] < 0:
            self.res[0:resInd,:,:] = 1
        elif self.boundaryID[1][0] > 0 and self.Domain.inlet[1] > 0:
            self.res[:,-resInd:,:] = 1
        elif self.boundaryID[1][1] < 0 and self.Domain.inlet[1] < 0:
            self.res[:,0:resInd,:] = 1
        elif self.boundaryID[2][0] > 0 and self.Domain.inlet[2] > 0:
            self.res[:,:,-resInd:] = 1
        elif self.boundaryID[2][1] < 0 and self.Domain.inlet[2] < 0:
            self.res[:,:,0:resInd] = 1



def genDomainSubDomain(rank,size,subDomains,nodes,boundaries,inlet,outlet,resInd,dataFormat,file,dataRead,dataReadkwargs=None):

    numSubDomains = np.prod(subDomains)

    if (size != numSubDomains):
        if rank==0: 
            print("Number of Subdomains Must Equal Number of Processors!...Exiting")
        communication.raiseError()
        
    ### Get Domain INFO for All Procs ###
    if file is not None:
        if dataReadkwargs is None:
            domainSize,sphereData = dataRead(file)
        else:
            domainSize,sphereData = dataRead(file,dataReadkwargs)
    if file is None:
        domainSize = np.array([[0.,14.],[-1.5,1.5],[-1.5,1.5]])
    domain = Domain(nodes = nodes, domainSize = domainSize, subDomains = subDomains, boundaries = boundaries, inlet=inlet, outlet=outlet)
    domain.getdXYZ()
    domain.getSubNodes()

    orient = Orientation()

    sD = subDomain(Domain = domain, ID = rank, subDomains = subDomains, Orientation = orient)
    sD.getInfo()
    sD.getNeighbors()
    sD.getXYZ()
    if dataFormat == "Sphere":
        sD.genDomainSphereData(sphereData)
    if dataFormat == "InkBotle":
        sD.genDomainInkBottle()
    sD.setBoundaryConditions(rank)
    sD.getBoundaryInfo()
    sD.getLoopInfo()
    if resInd > 0:
        sD.getReservoir(resInd)
    sD.getPorosity()
    comm.Allreduce( [sD.poreNodes, MPI.INT], [sD.totalPoreNodes, MPI.INT], op = MPI.SUM )

    loadBalancingCheck = False
    if loadBalancingCheck:
        loadData = [sD.ID,sD.ownNodesTotal]
        loadData = comm.gather(loadData, root=0)
        if rank == 0:
            sumTotalNodes = 0
            for ld in loadData:
                sumTotalNodes = sumTotalNodes + ld[2]
            print("Total Nodes",sumTotalNodes,"Pore Nodes",sD.totalPoreNodes)
            print("Ideal Load Balancing is %2.1f%%" %(1./numSubDomains*100.))
            for ld in loadData:
                p = ld[1]/sD.totalPoreNodes*100.
                t = ld[2]/sumTotalNodes*100.
                print("Rank: %i has %2.1f%% of the Pore Nodes and %2.1f%% of the total Nodes" %(ld[0],p,t))


    return domain,sD
