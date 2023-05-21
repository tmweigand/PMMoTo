import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD

from .domainGeneration import domainGenINK
from .domainGeneration import domainGen
from . import communication
from . import Orientation
from . import Domain

""" Solid = 0, Pore = 1 """

""" TO DO:
           Switch to pass periodic info and not generate from samples??
           Redo Domain decomposition - Maybe
"""

class subDomain(object):
    def __init__(self,ID,subDomains,Domain,Orientation):
        bufferSize        = 1
        self.ID          = ID
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
        self.buffer      = bufferSize*np.ones([3,2],dtype = np.int8)
        self.numSubDomains = np.prod(subDomains)
        self.neighborF    = -np.ones(self.Orientation.numFaces,dtype = np.int64)
        self.neighborE    = -np.ones(self.Orientation.numEdges,dtype = np.int64)
        self.neighborC    = -np.ones(self.Orientation.numCorners,dtype = np.int64)
        self.neighborPerF =  np.zeros([self.Orientation.numFaces,3],dtype = np.int64)
        self.neighborPerE =  np.zeros([self.Orientation.numEdges,3],dtype = np.int64)
        self.neighborPerC =  np.zeros([self.Orientation.numCorners,3],dtype = np.int64)
        self.ownNodes     = np.zeros([3,2],dtype = np.int64)
        self.poreNodes    = 0
        self.totalPoreNodes = np.zeros(1,dtype=np.uint64)
        self.subDomainSize = np.zeros([3,1])
        self.grid = None
        self.inlet = np.zeros([self.Orientation.numFaces],dtype = np.uint8)
        self.outlet = np.zeros([self.Orientation.numFaces],dtype = np.uint8)
        self.loopInfo = np.zeros([self.Orientation.numFaces+1,3,2],dtype = np.int64)

    def getInfo(self):
        """
        Gather information for each subDomain including:
        ID, boundary information,number of nodes, global index start
        """
        n = 0
        for i in range(0,self.subDomains[0]):
            for j in range(0,self.subDomains[1]):
                for k in range(0,self.subDomains[2]):
                    self.lookUpID[i,j,k] = n
                    if n == self.ID:
                        if (i == 0):
                            self.boundary = True
                            self.boundaryID[0][0] = 1
                        if (i == self.subDomains[0]-1):
                            self.boundary = True
                            self.boundaryID[0][1] = 1
                        if (j == 0):
                            self.boundary = True
                            self.boundaryID[1][0] = 1
                        if (j == self.subDomains[1]-1):
                            self.boundary = True
                            self.boundaryID[1][1] = 1
                        if (k == 0):
                            self.boundary = True
                            self.boundaryID[2][0] = 1
                        if (k == self.subDomains[2]-1):
                            self.boundary = True
                            self.boundaryID[2][1] = 1

                        self.subID[0] = i
                        self.subID[1] = j
                        self.subID[2] = k
                        self.nodes[0] = self.Domain.subNodes[0]
                        self.nodes[1] = self.Domain.subNodes[1]
                        self.nodes[2] = self.Domain.subNodes[2]
                        self.indexStart[0] = i * self.Domain.subNodes[0]
                        self.indexStart[1] = j * self.Domain.subNodes[1]
                        self.indexStart[2] = k * self.Domain.subNodes[2]
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
        If boundaryID and Domain.boundary == 0, buffer is not added
        Everywhere else a buffer is added
        """

        #########################################################
        ###   Determine if subDomain should not have buffer   ###
        ### Walls (1) and Periodic (2) Boundaries Have Buffer ### 
        #########################################################
        if (self.boundaryID[0][0] and self.Domain.boundaries[0][0] == 0):
            self.buffer[0][0] = 0
        if (self.boundaryID[0][1] and self.Domain.boundaries[0][1] == 0):
            self.buffer[0][1] = 0
        if (self.boundaryID[1][0] and self.Domain.boundaries[1] == 0):
            self.buffer[1][0] = 0
        if (self.boundaryID[1][1] and self.Domain.boundaries[1][0] == 0):
            self.buffer[1][1] = 0
        if (self.boundaryID[2][0] and self.Domain.boundaries[2][0] == 0):
            self.buffer[2][0] = 0
        if (self.boundaryID[2][1] and self.Domain.boundaries[2][1] == 0):
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
            communication.raiseError


    def genDomainSphereData(self,sphereData):
        self.grid = domainGen(self.x,self.y,self.z,sphereData)
        self.gridCheck()

    def genDomainInkBottle(self):
        self.grid = domainGenINK(self.x,self.y,self.z)
        self.gridCheck()

    def setBoundaryConditions(self):
        """
        If wall boundary conditions are specified, force solid on external boundaries
        """
        if self.boundaryID[0][0] == 1 and self.Domain.boundaries[0][0] == 1:
            self.grid[0,:,:] = 0
        if self.boundaryID[0][1] == 1 and self.Domain.boundaries[0][1] == 1:
            self.grid[-1,:,:] = 0
        if self.boundaryID[1][0] == 1 and self.Domain.boundaries[1][0] == 1:
            self.grid[:,0,:] = 0
        if self.boundaryID[1][1] == 1 and self.Domain.boundaries[1][1] == 1:
            self.grid[:,-1,:] = 0
        if self.boundaryID[2][0] == 1 and self.Domain.boundaries[2][0] == 1:
            self.grid[:,:,0] = 0
        if self.boundaryID[2][1] == 1 and self.Domain.boundaries[2][1] == 1:
            self.grid[:,:,-1] = 0

    def getLoopInfo(self):
        """
        Grap the Loop Information to Cycle through the Boundary Faces and Avoid IFs
        """
        rangeInfo = 2*np.ones([3,2],dtype=np.uint8)
        if self.boundaryID[0][0] == 1 and self.Domain.boundaries[0][0] == 0:
            rangeInfo[0,0] = rangeInfo[0,0] - 1
        if self.boundaryID[0][1] == 1 and self.Domain.boundaries[0][1] == 0:
            rangeInfo[0,1] = rangeInfo[0,1] - 1
        if self.boundaryID[1][0] == 1 and self.Domain.boundaries[1][0] == 0:
            rangeInfo[1,0] = rangeInfo[1,0] - 1
        if self.boundaryID[1][1] == 1 and self.Domain.boundaries[1][1] == 0:
            rangeInfo[1,1] = rangeInfo[1,1] - 1
        if self.boundaryID[2][0] == 1 and self.Domain.boundaries[2][0] == 0:
            rangeInfo[2,0] = rangeInfo[2,0] - 1
        if self.boundaryID[2][1] == 1 and self.Domain.boundaries[2][1] == 0:
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

        ###################################
        ### Determine inlet/outlet Info ###
        ###################################
        if (self.boundaryID[0][0] and  self.Domain.inlet[0][0]):
            self.inlet[0] = True
        if (self.boundaryID[0][1] and  self.Domain.inlet[0][1]):
            self.inlet[1] = True
        if (self.boundaryID[1][0] and  self.Domain.inlet[1][0]):
            self.inlet[2] = True
        if (self.boundaryID[1][1] and  self.Domain.inlet[1][1]):
            self.inlet[3] = True
        if (self.boundaryID[2][0] and  self.Domain.inlet[2][0]):
            self.inlet[4] = True
        if (self.boundaryID[2][1] and  self.Domain.inlet[2][1]):
            self.inlet[5] = True

        if (self.boundaryID[0][0] and  self.Domain.outlet[0][0]):
            self.outlet[0] = True
        if (self.boundaryID[0][1] and  self.Domain.outlet[0][1]):
            self.outlet[1] = True
        if (self.boundaryID[1][0] and  self.Domain.outlet[1][0]):
            self.outlet[2] = True
        if (self.boundaryID[1][1] and  self.Domain.outlet[1][1]):
            self.outlet[3] = True
        if (self.boundaryID[2][0] and  self.Domain.outlet[2][0]):
            self.outlet[4] = True
        if (self.boundaryID[2][1] and  self.Domain.outlet[2][1]):
            self.outlet[5] = True            
        #####################################################

    def getNeighbors(self):
        """
        Get the Face, Edge, and Corner Neighbors for Each Domain
        """
        lookIDPad = np.pad(self.lookUpID, ( (1, 1), (1, 1), (1, 1)), 'constant', constant_values=-1)
        lookPerI = np.zeros_like(lookIDPad)
        lookPerJ = np.zeros_like(lookIDPad)
        lookPerK = np.zeros_like(lookIDPad)

        if (self.Domain.boundaries[0][0] == 2):
            lookIDPad[0,:,:]  = lookIDPad[-2,:,:]
            lookIDPad[-1,:,:] = lookIDPad[1,:,:]
            lookPerI[0,:,:] = 1
            lookPerI[-1,:,:] = -1

        if (self.Domain.boundaries[1][0] == 2):
            lookIDPad[:,0,:]  = lookIDPad[:,-2,:]
            lookIDPad[:,-1,:] = lookIDPad[:,1,:]
            lookPerJ[:,0,:] = 1
            lookPerJ[:,-1,:] = -1

        if (self.Domain.boundaries[2][0] == 2):
            lookIDPad[:,:,0]  = lookIDPad[:,:,-2]
            lookIDPad[:,:,-1] = lookIDPad[:,:,1]
            lookPerK[:,:,0] = 1
            lookPerK[:,:,-1] = -1

        for cc,f in enumerate(self.Orientation.faces.values()):
            cx = f['ID'][0] + self.subID[0] + 1
            cy = f['ID'][1] + self.subID[1] + 1
            cz = f['ID'][2] + self.subID[2] + 1
            self.neighborF[cc]      = lookIDPad[cx,cy,cz]
            self.neighborPerF[cc,0] = lookPerI[cx,cy,cz]
            self.neighborPerF[cc,1] = lookPerJ[cx,cy,cz]
            self.neighborPerF[cc,2] = lookPerK[cx,cy,cz]

        for cc,e in enumerate(self.Orientation.edges.values()):
            cx = e['ID'][0] + self.subID[0] + 1
            cy = e['ID'][1] + self.subID[1] + 1
            cz = e['ID'][2] + self.subID[2] + 1
            self.neighborE[cc]      = lookIDPad[cx,cy,cz]
            self.neighborPerE[cc,0] = lookPerI[cx,cy,cz]
            self.neighborPerE[cc,1] = lookPerJ[cx,cy,cz]
            self.neighborPerE[cc,2] = lookPerK[cx,cy,cz]

        for cc,c in enumerate(self.Orientation.corners.values()):
            cx = c['ID'][0] + self.subID[0] + 1
            cy = c['ID'][1] + self.subID[1] + 1
            cz = c['ID'][2] + self.subID[2] + 1
            self.neighborC[cc]      = lookIDPad[cx,cy,cz]
            self.neighborPerC[cc,0] = lookPerI[cx,cy,cz]
            self.neighborPerC[cc,1] = lookPerJ[cx,cy,cz]
            self.neighborPerC[cc,2] = lookPerK[cx,cy,cz]

        self.lookUpID = lookIDPad

    def getPorosity(self):
        own = self.ownNodes
        ownGrid =  self.grid[own[0][0]:own[0][1],
                             own[1][0]:own[1][1],
                             own[2][0]:own[2][1]]
        self.poreNodes = np.sum(ownGrid)
        comm.Allreduce( [self.poreNodes, MPI.INT], [self.totalPoreNodes, MPI.INT], op = MPI.SUM )

    def loadBalancing(self):
        loadData = [self.ID,np.prod(self.ownNodes)]
        loadData = comm.gather(loadData, root=0)
        if self.ID == 0:
            sumTotalNodes = 0
            for ld in loadData:
                sumTotalNodes = sumTotalNodes + ld[2]
            print("Total Nodes",sumTotalNodes,"Pore Nodes",self.totalPoreNodes)
            print("Ideal Load Balancing is %2.1f%%" %(1./numSubDomains*100.))
            for ld in loadData:
                p = ld[1]/self.totalPoreNodes*100.
                t = ld[2]/sumTotalNodes*100.
                print("Rank: %i has %2.1f%% of the Pore Nodes and %2.1f%% of the total Nodes" %(ld[0],p,t))



def genDomainSubDomain(rank,size,subDomains,nodes,boundaries,inlet,outlet,dataFormat,file,dataRead,dataReadkwargs=None):

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
    domain = Domain.Domain(nodes = nodes, domainSize = domainSize, subDomains = subDomains, boundaries = boundaries, inlet=inlet, outlet=outlet)
    domain.getdXYZ()
    domain.getSubNodes()

    orient = Orientation.Orientation()

    sD = subDomain(Domain = domain, ID = rank, subDomains = subDomains, Orientation = orient)
    sD.getInfo()
    sD.getNeighbors()
    sD.getXYZ()
    if dataFormat == "Sphere":
        sD.genDomainSphereData(sphereData)
    if dataFormat == "InkBotle":
        sD.genDomainInkBottle()
    sD.setBoundaryConditions()
    sD.getBoundaryInfo()
    sD.getLoopInfo()
    sD.getPorosity()

    loadBalancingCheck = False
    if loadBalancingCheck:
        sD.loadBalancing()


    return domain,sD
