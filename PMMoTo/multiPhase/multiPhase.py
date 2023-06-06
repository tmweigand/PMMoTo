import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
from .. import dataOutput
from .. import dataRead

#### TO DO: Make phase ID generic
# 0 is always solid
# 1 is Wetting Phase
# 2 is NonWetting Phase


class multiPhase(object):
    def __init__(self,porousMedia,numFluidPhases):
        self.porousMedia    = porousMedia
        self.Domain         = porousMedia.Domain
        self.Orientation    = porousMedia.subDomain.Orientation
        self.subDomain      = porousMedia.subDomain
        self.numFluidPhases = numFluidPhases
        self.fluidIDs       = list(range(1,numFluidPhases+1))
        self.mpGrid         = None
        self.wID            = 2
        self.nwID           = 1
        self.inlet          = {}
        self.outlet         = {}
        self.loopInfo       = {}  

    def initializeMPGrid(self,constantPhase = -1,inputFile = None):
        """
        Set The initial distribution of fluids. 
        If fully saturated by a given phase, set constantPhase = phaseID
        If inputFile, read data
        Else set poreSpace to 1 and warn
        """
        if constantPhase > -1:
            self.mpGrid = np.where(self.porousMedia.grid == 1,constantPhase,0).astype(np.uint8)
        elif inputFile is not None:
            self.mpGrid = dataRead.readVTKGrid(self.subDomain.ID,self.Domain.numSubDomains,inputFile)
        else:
            self.mpGrid = np.copy(self.porousMedia.grid)
            if self.subDomain.ID:
                print("No input Parameter given. Setting phase distribution to 1")


    def saveMPGrid(self,fileName):
        dataOutput.saveMultiPhaseData(fileName,self.subDomain.ID,self.Domain,self.subDomain,self)

    def getBoundaryInfo(self,inlets,outlets,resSize):
        """
        Determine Inlet/Outlet for Each Fluid Phase
        TO DO: Optimize loopInfo so phases dont loop over other phase reservoirs
        """
        for fluid in self.fluidIDs:
            ### INLET ###
            self.inlet[fluid] = np.zeros([self.Orientation.numFaces],dtype = np.int8)
            if (self.subDomain.boundaryID[0] == 0 and inlets[fluid][0][0]):
                self.inlet[fluid][0] = resSize
            if (self.subDomain.boundaryID[1] == 0 and inlets[fluid][0][1]):
                self.inlet[fluid][1] = resSize
            if (self.subDomain.boundaryID[2] == 0 and inlets[fluid][1][0]):
                self.inlet[fluid][2] = resSize
            if (self.subDomain.boundaryID[3] == 0 and inlets[fluid][1][1]):
                self.inlet[fluid][3] = resSize
            if (self.subDomain.boundaryID[4] == 0 and inlets[fluid][2][0]):
                self.inlet[fluid][4] = resSize
            if (self.subDomain.boundaryID[5] == 0 and inlets[fluid][2][1]):
                self.inlet[fluid][5] = resSize

            ### OUTLET ###
            self.outlet[fluid] = np.zeros([self.Orientation.numFaces],dtype = np.int8)
            if (self.subDomain.boundaryID[0] == 0 and outlets[fluid][0][0]):
                self.outlet[fluid][0] = resSize
            if (self.subDomain.boundaryID[1] == 0 and outlets[fluid][0][1]):
                self.outlet[fluid][1] = resSize
            if (self.subDomain.boundaryID[2] == 0 and outlets[fluid][1][0]):
                self.outlet[fluid][2] = resSize
            if (self.subDomain.boundaryID[3] == 0 and outlets[fluid][1][1]):
                self.outlet[fluid][3] = resSize
            if (self.subDomain.boundaryID[4] == 0 and outlets[fluid][2][0]):
                self.outlet[fluid][4] = resSize
            if (self.subDomain.boundaryID[5] == 0 and outlets[fluid][2][1]):
                self.outlet[fluid][5] = resSize    
    
            pad = np.zeros([6],dtype = np.int8)
            for f in range(0,self.Orientation.numFaces):
                pad[f] = self.inlet[fluid][f] + self.outlet[fluid][f]      
            
            ### If Inlet/Outlet Res, Pad and Update XYZ
            if np.sum(pad) > 0:
                self.mpGrid = np.pad(self.mpGrid, ( (pad[0], pad[1]), (pad[2], pad[3]), (pad[4], pad[5]) ), 'constant', constant_values = fluid)
                self.porousMedia.grid = np.pad( self.porousMedia.grid , ( (pad[0], pad[1]), (pad[2], pad[3]), (pad[4], pad[5]) ), 'constant', constant_values = 1)
                self.subDomain.getXYZ(pad)

        for fluid in self.fluidIDs:
            self.loopInfo[fluid] = self.Orientation.getLoopInfo(self.mpGrid,self.subDomain,self.inlet[fluid],self.outlet[fluid],resSize)
        
