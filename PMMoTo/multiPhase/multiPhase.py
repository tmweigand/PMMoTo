import math
import numpy as np

from mpi4py import MPI
comm = MPI.COMM_WORLD
from .. import communication
from .. import distance
from .. import morphology
from .. import nodes
from .. import sets
from .. import dataOutput
from .. import dataRead
import sys

#### TO DO: Make phase ID generic
# 0 is always solid
# 1 is Wetting Phase
# 2 is NonWetting Phase


class multiPhase(object):
    def __init__(self,Domain,subDomain,numFluidPhases):
        self.Domain         = Domain
        self.Orientation    = subDomain.Orientation
        self.subDomain      = subDomain
        self.numFluidPhases = numFluidPhases
        self.fluidIDs       = list(range(1,numFluidPhases+1))
        self.mpGrid         = None
        self.wID            = 1
        self.nwID           = 2
        self.inlet          = {}
        self.outlet         = {}

    def initializeMPGrid(self,constantPhase = -1,inputFile = None):
        """
        Set The initial distribution of fluids. 
        If fully saturated by a given phase, set constantPhase = phaseID
        If inputFile, read data
        Else set poreSpace to 1 and warn
        """
        if constantPhase > -1:
            self.mpGrid = np.where(self.subDomain.grid == 1,constantPhase,0).astype(np.uint8)
        elif inputFile is not None:
            self.mpGrid = dataRead.readVTKGrid(self.Domain.numSubDomains,self.subDomain.ID,inputFile)
        else:
            self.mpGrid = np.copy(self.subDomain.grid)
            if self.subDomain.ID:
                print("No input Parameter given. Setting phase distribution to 1")


    def saveMPGrid(self,fileName):
        dataOutput.saveMultiPhaseData(fileName,self.subDomain.ID,self.Domain,self.subDomain,self)

    def getBoundaryInfo(self,inlets,outlets):
        """
        Determine Inlet/Outlet for Each Fluid Phase
        """

        for c,fluid in enumerate(self.fluidIDs):
            ### INLET ###
            self.inlet[fluid] = np.zeros([self.Orientation.numFaces],dtype = bool)
            if (self.subDomain.boundaryID[0][0] and  inlets[c][0][0]):
                self.inlet[fluid][0] = True
            if (self.subDomain.boundaryID[0][1] and  inlets[c][0][1]):
                self.inlet[fluid][1] = True
            if (self.subDomain.boundaryID[1][0] and  inlets[c][1][0]):
                self.inlet[fluid][2] = True
            if (self.subDomain.boundaryID[1][1] and  inlets[c][1][1]):
                self.inlet[fluid][3] = True
            if (self.subDomain.boundaryID[2][0] and  inlets[c][2][0]):
                self.inlet[fluid][4] = True
            if (self.subDomain.boundaryID[2][1] and  inlets[c][2][1]):
                self.inlet[fluid][5] = True

            ### OUTLET ###
            self.outlet[fluid] = np.zeros([self.Orientation.numFaces],dtype = bool)
            if (self.subDomain.boundaryID[0][0] and  outlets[c][0][0]):
                self.outlet[fluid][0] = True
            if (self.subDomain.boundaryID[0][1] and  outlets[c][0][1]):
                self.outlet[fluid][1] = True
            if (self.subDomain.boundaryID[1][0] and  outlets[c][1][0]):
                self.outlet[fluid][2] = True
            if (self.subDomain.boundaryID[1][1] and  outlets[c][1][1]):
                self.outlet[fluid][3] = True
            if (self.subDomain.boundaryID[2][0] and  outlets[c][2][0]):
                self.outlet[fluid][4] = True
            if (self.subDomain.boundaryID[2][1] and  outlets[c][2][1]):
                self.outlet[fluid][5] = True    
    
