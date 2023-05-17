import math
import numpy as np
cimport numpy as cnp
cimport cython
from libc.stdlib cimport malloc, free

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


class Drainage(object):
    def __init__(self,Domain,Orientation,subDomain,gamma,inlet,edt):
        self.Domain      = Domain
        self.Orientation = Orientation
        self.subDomain   = subDomain
        self.edt         = edt
        self.gamma       = gamma
        self.inletDirection = 0
        self.probeD = 0
        self.probeR = 0
        self.pC = 0
        self.numNWP = 0
        self.ind = None
        self.nwp = None
        self.globalIndexStart = 0
        self.globalBoundarySetID = None
        self.inlet = inlet
        self.matchedSets = []
        self.nwpNodes = 0
        self.totalnwpNodes = np.zeros(1,dtype=np.uint64)
        self.nwpRes = np.zeros([3,2])
        self.Sets = []

    def getDiameter(self,pc):
        if pc == 0:
            self.probeD = 0
            self.probeR = 0
        else:
            self.probeR = 2.*self.gamma/pc
            self.probeD = 2.*self.probeR

    def getpC(self,radius):
        self.pC = 2.*self.gamma/radius

    def probeDistance(self):
        self.ind = np.where( (self.edt.EDT >= self.probeR) & (self.subDomain.grid == 1),1,0).astype(np.uint8)
        self.numNWP = np.sum(self.ind)

    def getNWP(self):
        NWNodes = []
        self.nwp = np.zeros_like(self.ind)

        for s in self.Sets:
            if s.inlet:
                for node in s.nodes:
                    NWNodes.append(node)

        for n in NWNodes:
            self.nwp[n[0],n[1],n[2]] = 1

    def finalizeNWP(self,nwpDist):

        if self.nwpRes[0,0]:
            nwpDist = nwpDist[-1:,:,:]
        elif self.nwpRes[0,1]:
            nwpDist = nwpDist[1:,:,:]
        self.nwpFinal = np.copy(self.subDomain.grid)
        self.nwpFinal = np.where( (nwpDist ==  1) & (self.subDomain.grid == 1),2,self.nwpFinal)
        self.nwpFinal = np.where( (self.subDomain.res == 1),2,self.nwpFinal)

        own = self.subDomain.ownNodes
        ownGrid =  self.nwpFinal[own[0][0]:own[0][1],
                             own[1][0]:own[1][1],
                             own[2][0]:own[2][1]]
        self.nwpNodes = np.sum(np.where(ownGrid==2,1,0))




def calcDrainage(rank,size,pc,Domain,subDomain,inlet,EDT,info = False, save=False):

    ### Loop through all capilarry pressures. ###
    for p in pc:
        if p == 0:
            sW = 1
        else:
            drain = Drainage(Domain = Domain, Orientation = subDomain.Orientation, subDomain = subDomain, edt = EDT, gamma = 1., inlet = inlet)
            if info:
                drain.getpC(EDT.maxD)
                print("Minimum pc",drain.pC)
                pCMax = drain.getpC(EDT.minD)
                print("Maximum pc",drain.pC)

            drain.getDiameter(p)
            drain.probeDistance()
            numNWPSum = np.zeros(1,dtype=np.uint64)
            comm.Allreduce( [drain.numNWP, MPI.INT], [numNWPSum, MPI.INT], op = MPI.SUM )
            if numNWPSum < 1:
                drain.nwp = np.copy(subDomain.grid)
                drain.nwpFinal = drain.nwp
            else:
                drain.Sets,drain.setCount = sets.collectSets(rank,size,drain.ind,1,Domain,subDomain)
                drain.getNWP()
                morphL = morphology.morph(rank,size,Domain,subDomain,drain.nwp,drain.probeR)
                drain.finalizeNWP(morphL.gridOut)

                numNWPSum = np.zeros(1,dtype=np.uint64)
                comm.Allreduce( [drain.nwpNodes, MPI.INT], [drain.totalnwpNodes, MPI.INT], op = MPI.SUM )

                if save:
                    name = "dataOut/twoPhase/phaseDist_pc_" + str(p)
                    dataOutput.saveMultiPhaseData(name,rank,Domain,subDomain,drain)
        
        if rank == 0:
            sW = 1.-drain.totalnwpNodes[0]/subDomain.totalPoreNodes[0]
            print("Wetting phase saturation is: %e at pC of %e" %(sW,p))
        

    return drain,morphL
