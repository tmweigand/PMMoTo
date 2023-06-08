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


class equilibriumDistribution(object):
    def __init__(self,multiPhase):
        self.multiPhase  = multiPhase
        self.Domain      = multiPhase.Domain
        self.Orientation = multiPhase.Orientation
        self.subDomain   = multiPhase.subDomain
        self.porousMedia = multiPhase.porousMedia
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

    def getInletConnectedNodes(self,sets,flag):
        """
        Grab from Sets that are on the Inlet Reservoir and create binary grid
        """
        nodes = []
        gridOut = np.zeros_like(self.multiPhase.mpGrid)

        for s in sets:
            if s.inlet:
                for node in s.nodes:
                    nodes.append(node)

        for n in nodes:
            gridOut[n[0],n[1],n[2]] = flag

        return gridOut

    def getDisconnectedNodes(self,sets,flag):
        """
        Grab from Sets that are on the Inlet Reservoir and create binary grid
        """
        nodes = []
        gridOut = np.zeros_like(self.subDomain.grid)

        for s in sets:
            if not s.inlet:
                for node in s.nodes:
                    nodes.append(node)

        for n in nodes:
            gridOut[n[0],n[1],n[2]] = flag

        return gridOut
    
    def drainInfo(self,maxEDT,minEDT):
        self.getpC(maxEDT)
        print("Minimum pc",self.pC)
        self.getpC(maxEDT)
        print("Maximum pc",self.pC)

    def calcSaturation(self,grid,nwID):

        own = self.subDomain.ownNodesIndex
        ownGrid =  grid[own[0]:own[1],
                        own[2]:own[3],
                        own[4]:own[5]]
        nwNodes = np.count_nonzero(ownGrid==nwID)
        allnwNodes = np.zeros(1,dtype=np.uint64)
        comm.Allreduce( [np.int64(nwNodes), MPI.INT], [allnwNodes, MPI.INT], op = MPI.SUM )
        sw = 1. - allnwNodes[0]/self.porousMedia.totalPoreNodes[0]
        return sw


    def checkPoints(self,grid,ID):
        """
        Check to make sure nodes of type ID exist in domain
        """
        noPoints = False
        if ID == 0:
            count = np.size(grid) - np.count_nonzero(grid > 0)
        else:
            own = self.subDomain.ownNodesIndex
            ownGrid =  grid[own[0]:own[1],
                            own[2]:own[3],
                            own[4]:own[5]]
            count = np.count_nonzero(ownGrid==ID)
        allCount = np.zeros(1,dtype=np.uint64)
        comm.Allreduce( [np.int64(count), MPI.INT], [allCount, MPI.INT], op = MPI.SUM )

        if allCount > 0:
            noPoints =  True

        return noPoints


def calcDrainage(pc,mP):

    ### Get Distance from Solid to Pore Space (Ignore Fluid Phases)
    poreSpaceDist = distance.calcEDT(mP.subDomain,mP.porousMedia.grid)
    eqDist = equilibriumDistribution(mP)
    sW = eqDist.calcSaturation(mP.mpGrid,2)
    save = True

    fileName = "dataOut/test/distCSV"
    dataOutput.saveGridcsv(fileName,mP.subDomain,mP.subDomain.x,mP.subDomain.y,mP.subDomain.z,poreSpaceDist,removeHalo = True)

    # fileName = "dataOut/test/dist"
    # dataOutput.saveGrid(fileName,mP.subDomain,poreSpaceDist)

    setSaveDict = {'inlet': 'inlet',
                   'outlet':'outlet',
                    'boundary': 'boundary',
                    'localID': 'localID'}

    ### Loop through all Pressures
    for p in pc:
        if p == 0:
            sW = 1
        else:
            ### Get Sphere Radius from Pressure
            eqDist.getDiameter(p)

            # Step 1 - Reservoirs are not contained in mdGrid or grid but rather added when needed so this step is unnecessary
            
            # Step 2 - Dilate Solid Phase and Flag Allowable Fluid Voxes as 1 
            ind = np.where( (poreSpaceDist >= eqDist.probeR) & (mP.porousMedia.grid == 1),1,0).astype(np.uint8)

            # fileName = "dataOut/test/indGrid"
            # dataOutput.saveGrid(fileName,mP.subDomain,ind)

            # Step 3 - Check if Points were Marked
            continueFlag = eqDist.checkPoints(ind,1)
            if continueFlag:

                # Step 3a and 3d - Check if NW Phases Exists then Collect NW Sets
                nwCheck = eqDist.checkPoints(mP.mpGrid,mP.nwID)
                if nwCheck:
                    nwSets,nwSetCount = sets.collectSets(mP.mpGrid,mP.nwID,mP.inlet[mP.nwID],mP.outlet[mP.nwID],mP.loopInfo[mP.nwID],mP.subDomain)
                    nwGrid = eqDist.getInletConnectedNodes(nwSets,1)

                    # setSaveDict = {'inlet': 'inlet',
                    #                'outlet':'outlet',
                    #                'boundary': 'boundary',
                    #                'localID': 'localID'}

                    # eqDist.Sets = nwSets
                    # eqDist.setCount = nwSetCount

                    # dataOutput.saveSetData("dataOut/NWset",mP.subDomain,eqDist,**setSaveDict)

                    # fileName = "dataOut/test/nwGrid"+str(p)
                    # dataOutput.saveGrid(fileName,mP.subDomain,nwGrid)

                # Step 3b and 3d- Check if W Phases Exists then Collect W Sets
                wCheck = eqDist.checkPoints(mP.mpGrid,mP.wID)
                if wCheck:
                    wSets,wSetCount = sets.collectSets(mP.mpGrid,mP.wID,mP.inlet[mP.wID],mP.outlet[mP.wID],mP.loopInfo[mP.wID],mP.subDomain)
                    wGrid = eqDist.getInletConnectedNodes(wSets,1)

                    # fileName = "dataOut/test/wGrid"+str(p)
                    # dataOutput.saveGrid(fileName,mP.subDomain,wGrid)

                # Steb 3c and 3d - Already checked at Step 3 so Collect Sets with ID = 1
                indSets,indSetCount = sets.collectSets(ind,1,mP.inlet[mP.nwID],mP.outlet[mP.nwID],mP.loopInfo[mP.nwID],mP.subDomain)
                ind = eqDist.getInletConnectedNodes(indSets,1)

                # Step 3e - no Step 3e ha. 

                # Step 3f -- Unsure about these checks!
                if wCheck and nwCheck:
                    ind = np.where( (ind == 1) & (nwGrid == 1) & (wGrid == 0),1,0).astype(np.uint8)
                elif nwCheck:
                    ind = np.where( (ind == 1) & (nwGrid == 1),1,0).astype(np.uint8)
                elif wCheck:
                    ind = np.where( (ind == 1) & (wGrid == 1),1,0).astype(np.uint8)

                # fileName = "dataOut/test/indFinal"+str(p)
                # dataOutput.saveGrid(fileName,mP.subDomain,ind)

                # Step 3g
                morph = morphology.morph(ind,mP.subDomain,eqDist.probeR)

                # fileName = "dataOut/test/morph"+str(p)
                # dataOutput.saveGrid(fileName,mP.subDomain,morph)

                mP.mpGrid = np.where( (morph == 1) & (wGrid == 1),mP.nwID,mP.mpGrid)

                # Step 4
                sw = eqDist.calcSaturation(mP.mpGrid,mP.nwID)
                if mP.subDomain.ID == 0:
                    print("Capillary pressure: %e Wetting Phase Saturation: %e" %(p,sw))

            if save:
                fileName = "dataOut/twoPhase/twoPhase_pc_"+str(p)
                dataOutput.saveGrid(fileName,mP.subDomain,mP.mpGrid)        

    return eqDist
