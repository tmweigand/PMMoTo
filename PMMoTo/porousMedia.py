import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD

from .domainGeneration import domainGenINK
from .domainGeneration import domainGen
from . import communication
from . import subDomain


class porousMedia(object):
    def __init__(self,subDomain,Domain,Orientation):
        self.subDomain   = subDomain
        self.Domain      = Domain
        self.Orientation = Orientation
        self.grid = None
        self.inlet = np.zeros([self.Orientation.numFaces],dtype = np.uint8)
        self.outlet = np.zeros([self.Orientation.numFaces],dtype = np.uint8)
        self.loopInfo = np.zeros([self.Orientation.numFaces+1,3,2],dtype = np.int64)
        self.ownNodes     = np.zeros([3,2],dtype = np.int64)
        self.poreNodes    = 0
        self.totalPoreNodes = np.zeros(1,dtype=np.uint64)

    def gridCheck(self):
        if (np.sum(self.grid) == np.prod(self.subDomain.nodes)):
            print("This code requires at least 1 solid voxel in each subdomain. Please reorder processors!")
            communication.raiseError

    def genDomainSphereData(self,sphereData):
        self.grid = domainGen(self.subDomain.x,self.subDomain.y,self.subDomain.z,sphereData)
        self.gridCheck()

    def genDomainInkBottle(self):
        self.grid = domainGenINK(self.subDomain.x,self.subDomain.y,self.subDomain.z)
        self.gridCheck()

    def setInletOutlet(self,resSize):
        """
        Determine inlet/outlet Info and Pad Grid
        """

        if (self.subDomain.boundaryID[0] == 0 and  self.Domain.inlet[0][0]):
            self.inlet[0] = resSize
        if (self.subDomain.boundaryID[1] == 0 and  self.Domain.inlet[0][1]):
            self.inlet[1] = resSize
        if (self.subDomain.boundaryID[2] == 0 and  self.Domain.inlet[1][0]):
            self.inlet[2] = resSize
        if (self.subDomain.boundaryID[3] == 0 and  self.Domain.inlet[1][1]):
            self.inlet[3] = resSize
        if (self.subDomain.boundaryID[4] == 0 and  self.Domain.inlet[2][0]):
            self.inlet[4] = resSize
        if (self.subDomain.boundaryID[5] == 0 and  self.Domain.inlet[2][1]):
            self.inlet[5] = resSize

        if (self.subDomain.boundaryID[0] == 0 and  self.Domain.outlet[0][0]):
            self.outlet[0] = resSize
        if (self.subDomain.boundaryID[1] == 0 and  self.Domain.outlet[0][1]):
            self.outlet[1] = resSize
        if (self.subDomain.boundaryID[2] == 0 and  self.Domain.outlet[1][0]):
            self.outlet[2] = resSize
        if (self.subDomain.boundaryID[3] == 0 and  self.Domain.outlet[1][1]):
            self.outlet[3] = resSize
        if (self.subDomain.boundaryID[4] == 0 and  self.Domain.outlet[2][0]):
            self.outlet[4] = resSize
        if (self.subDomain.boundaryID[5] == 0 and  self.Domain.outlet[2][1]):
            self.outlet[5] = resSize   

        pad = np.zeros([6],dtype = np.int8)
        for f in range(0,self.Orientation.numFaces):
            pad[f] = self.inlet[f] + self.outlet[f]      
        
        ### If Inlet/Outlet Res, Pad and Update XYZ
        if np.sum(pad) > 0:
            self.grid = np.pad(self.grid, ( (pad[0], pad[1]), (pad[2], pad[3]), (pad[4], pad[5]) ), 'constant', constant_values=1)
            self.subDomain.getXYZ(pad)


    def setWallBoundaryConditions(self):
        """
        If wall boundary conditions are specified, force solid on external boundaries
        """
        if self.subDomain.boundaryID[0] == 1:
            self.grid[0,:,:] = 0
        if self.subDomain.boundaryID[1] == 1:
            self.grid[-1,:,:] = 0
        if self.subDomain.boundaryID[2] == 1:
            self.grid[:,0,:] = 0
        if self.subDomain.boundaryID[3] == 1:
            self.grid[:,-1,:] = 0
        if self.subDomain.boundaryID[4] == 1:
            self.grid[:,:,0] = 0
        if self.subDomain.boundaryID[5] == 1:
            self.grid[:,:,-1] = 0

    def getPorosity(self):
        own = self.subDomain.ownNodes
        ownGrid =  self.grid[own[0][0]:own[0][1],
                             own[1][0]:own[1][1],
                             own[2][0]:own[2][1]]
        self.poreNodes = np.sum(ownGrid)
        comm.Allreduce( [self.poreNodes, MPI.INT], [self.totalPoreNodes, MPI.INT], op = MPI.SUM )


def genPorousMedia(subDomain,dataFormat,sphereData=None):

    pM = porousMedia(Domain = subDomain.Domain, subDomain = subDomain, Orientation = subDomain.Orientation)

    if dataFormat == "Sphere":
        pM.genDomainSphereData(sphereData)
    if dataFormat == "InkBotle":
        pM.genDomainInkBottle()
    pM.setInletOutlet(resSize=33)
    pM.setWallBoundaryConditions()
    pM.loopInfo = pM.Orientation.getLoopInfo(pM.grid,subDomain,pM.inlet,pM.outlet,33)
    pM.getPorosity()

    loadBalancingCheck = False
    if loadBalancingCheck:
        pM.loadBalancing()

    return pM