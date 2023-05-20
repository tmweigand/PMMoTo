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
            self.inlet[fluid] = np.zeros([self.Orientation.numFaces],dtype = np.uint8)
            self.outlet[fluid] = np.zeros([self.Orientation.numFaces],dtype = np.uint8)

            for fIndex in self.Orientation.faces:
                face = self.Orientation.faces[fIndex]['argOrder'][0]
                fI = self.Orientation.faces[fIndex]['Index']

                if self.subDomain.boundaryID[face][fI] != 0:
                    self.subDomain.globalBoundary[fIndex] = 1
                    if inlets[c][face][fI] == True and self.subDomain.boundaryID[face][fI] == True:
                        self.inlet[fluid][fIndex] = True
                    if outlets[c][face][fI] == True and self.subDomain.boundaryID[face][fI] == True:
                        self.outlet[fluid][fIndex] = True        

        

