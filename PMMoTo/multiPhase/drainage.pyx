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
    def __init__(self,multiPhase):
        self.Domain      = multiPhase.Domain
        self.Orientation = multiPhase.Orientation
        self.subDomain   = multiPhase.subDomain
        self.gamma       = 1
        self.probeD = 0
        self.probeR = 0
        self.pC = 0

    def getDiameter(self,pc):
        if pc == 0:
            self.probeD = 0
            self.probeR = 0
        else:
            self.probeR = 2.*self.gamma/pc
            self.probeD = 2.*self.probeR

    def getpC(self,radius):
        self.pC = 2.*self.gamma/radius

    def getWResConnectedNodes(self,sets):
        """
        Grab from Sets that are on the Inlet and create binary grid
        """
        nodes = []
        gridOut = np.zeros_like(self.subDomain.grid)

        for s in sets:
            if s.inlet:
                for node in s.nodes:
                    nodes.append(node)

        for n in nodes:
            gridOut[n[0],n[1],n[2]] = 1

        return gridOut

    def getNWResConnectedNodes(self,sets):
        """
        Grab from Sets that are on the Outlet and create binary grid
        """
        nodes = []
        gridOut = np.zeros_like(self.subDomain.grid)

        for s in sets:
            if s.outlet:
                for node in s.nodes:
                    nodes.append(node)

        for n in nodes:
            gridOut[n[0],n[1],n[2]] = 1

        return gridOut
    
    def drainInfo(self,maxEDT,minEDT):
        self.getpC(maxEDT)
        print("Minimum pc",self.pC)
        self.getpC(maxEDT)
        print("Maximum pc",self.pC)

    def calcSaturation(self,grid,nwID):
        nwNodes = np.count_nonzero(grid==nwID)
        allnwNodes = np.zeros(1,dtype=np.uint64)
        comm.Allreduce( [np.int64(nwNodes), MPI.INT], [allnwNodes, MPI.INT], op = MPI.SUM )
        sw = 1. - allnwNodes[0]/self.subDomain.totalPoreNodes[0]
        return sw


    def checkPoints(self,grid,ID):
        """
        Check to make sure nodes of type ID exist in domain
        """
        noPoints = False
        if ID == 0:
            count = np.size(grid) - np.count_nonzero(grid > 0)
        else:
            count = np.count_nonzero(grid==ID)
        allCount = np.zeros(1,dtype=np.uint64)
        comm.Allreduce( [np.int64(count), MPI.INT], [allCount, MPI.INT], op = MPI.SUM )

        if allCount > 0:
            noPoints =  True

        return noPoints


def calcDrainage(pc,multiPhase):

    ### Get Distance from Solid to Pore Space (Ignore Fluid Phases)
    poreSpaceDist = distance.calcEDT(multiPhase.subDomain,multiPhase.subDomain.grid)
    drain = Drainage(multiPhase)
    save = True

    ### Loop through all Pressures
    for p in pc:
        if p == 0:
            sW = 1
        else:
            ### Get Sphere Radius from Pressure
            drain.getDiameter(p)

            # Step 1 - Reservoirs are not contained in mdGrid or grid but rather added when needed so this step is unnecessary
            
            # Step 2 - Dilate Solid Phase and Flag Allowable Fluid Voxes as 1 
            ind = np.where( (poreSpaceDist >= drain.probeR) & (multiPhase.subDomain.grid == 1),1,0).astype(np.uint8)

            # Step 3 - Check if Points were Marked
            continueFlag = drain.checkPoints(ind,1)
            if continueFlag:

                # Step 3a and 3d - Check if NW Phases Exists then Collect NW Sets
                nwCheck = drain.checkPoints(multiPhase.mpGrid,multiPhase.nwID)
                if nwCheck:
                    nwSets,nwSetCount = sets.collectSets(multiPhase.mpGrid,multiPhase.nwID,multiPhase.subDomain)
                    nwGrid = drain.getNWResConnectedNodes(nwSets)


                # Step 3b and 3d- Check if W Phases Exists then Collect W Sets
                wCheck = drain.checkPoints(multiPhase.mpGrid,multiPhase.wID)
                if wCheck:
                    wSets,wSetCount = sets.collectSets(multiPhase.mpGrid,multiPhase.wID,multiPhase.subDomain)
                    wGrid = drain.getWResConnectedNodes(wSets)
                    
                    
                    setSaveDict = {'inlet': 'inlet',
                                   'outlet':'outlet',
                                   'boundary': 'boundary',
                                   'localID': 'localID'}

                    drain.Sets = wSets
                    drain.setCount = wSetCount
                    
                    dataOutput.saveSetData("dataOut/Wset",multiPhase.subDomain,drain,**setSaveDict)

                    fileName = "dataOut/test/wGrid"
                    dataOutput.saveGrid(fileName,multiPhase.subDomain,wGrid)

                # Steb 3c and 3d - Already checked at Step 3 so Collect Sets with ID = 1
                indSets,indSetCounts = sets.collectSets(ind,1,multiPhase.subDomain)
                ind = drain.getWResConnectedNodes(indSets)

                # Step 3e - no Step 3e ha. 

                # Step 3f 
                if wCheck and nwCheck:
                    ind = np.where( (ind == 1) & (nwGrid == 0) & (wGrid == 1),1,0).astype(np.uint8)
                elif nwCheck:
                    ind = np.where( (ind == 1) & (nwGrid == 0),1,0).astype(np.uint8)
                elif wCheck:
                    ind = np.where( (ind == 1) & (wGrid == 1),1,0).astype(np.uint8)

                
                # Step 3g
                morph = morphology.morph(ind,multiPhase.subDomain,drain.probeR)
                multiPhase.mpGrid = np.where( (morph == 1) & (wGrid == 1),2,multiPhase.mpGrid)

                # Step 4
                sw = drain.calcSaturation(multiPhase.mpGrid,2)
                print(p,sw)

            if save:
                fileName = "dataOut/twoPhase/twoPhase_pc_"+str(p)
                dataOutput.saveGrid(fileName,multiPhase.subDomain,multiPhase.mpGrid)





        #     drain.probeDistance()
        #     numNWPSum = np.zeros(1,dtype=np.uint64)
        #     comm.Allreduce( [drain.numNWP, MPI.INT], [numNWPSum, MPI.INT], op = MPI.SUM )
        #     if numNWPSum < 1:
        #         drain.nwp = np.copy(subDomain.grid)
        #         drain.nwpFinal = drain.nwp
        #     else:
        #         drain.Sets,drain.setCount = sets.collectSets(rank,size,drain.ind,1,Domain,subDomain)
        #         drain.getNWP()
        #         morphL = morphology.morph(rank,size,Domain,subDomain,drain.nwp,drain.probeR)
        #         drain.finalizeNWP(morphL.gridOut)

        #         numNWPSum = np.zeros(1,dtype=np.uint64)
        #         comm.Allreduce( [drain.nwpNodes, MPI.INT], [drain.totalnwpNodes, MPI.INT], op = MPI.SUM )

        
        # if rank == 0:
        #     sW = 1.-drain.totalnwpNodes[0]/subDomain.totalPoreNodes[0]
        #     print("Wetting phase saturation is: %e at pC of %e" %(sW,p))
        

    return drain
