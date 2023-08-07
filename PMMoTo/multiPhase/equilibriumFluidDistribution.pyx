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
        gridOut = np.zeros_like(self.multiPhase.mpGrid)

        for s in sets:
            if not s.inlet:
                for node in s.nodes:
                    nodes.append(node)

        for n in nodes:
            gridOut[n[0],n[1],n[2]] = flag

        return gridOut
    
    def removeSmallSets(self,sets,gridIn,nwID,minSetSize):
        """
        Remove sets smaller than target size
        """
        nodes = []
        gridOut = np.copy(gridIn)

        for s in sets:
            # print(s.numNodes)
            if s.numGlobalNodes < minSetSize:
                for node in s.nodes:
                    nodes.append(node)
                    
        for n in nodes:
            gridOut[n[0],n[1],n[2]] = nwID

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
        
        # print(allnwNodes[0],self.porousMedia.totalPoreNodes[0])
        return sw


    def checkPoints(self,grid,ID,includeInlet = False):
        """
        Check to make sure nodes of type ID exist in domain
        """
        if includeInlet:
            own = self.multiPhase.ownNodesIndex[ID]
        else:
            own = self.subDomain.ownNodesIndex

        noPoints = False
        if ID == 0:
            count = np.size(grid) - np.count_nonzero(grid > 0)
        else:
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

    ## Make sure pc targets are ordered smallest to largest
    pc.sort(reverse=False)

    # fileName = "dataOut/test/distCSV"
    # dataOutput.saveGridcsv(fileName,mP.subDomain,mP.subDomain.x,mP.subDomain.y,mP.subDomain.z,poreSpaceDist,removeHalo = True)

    # fileName = "dataOut/test/dist"
    # dataOutput.saveGrid(fileName,mP.subDomain,poreSpaceDist)

    # setSaveDict = {'inlet': 'inlet',
    #                'outlet':'outlet',
    #                 'boundary': 'boundary',
    #                 'localID': 'localID'}

    result = []
    
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
            # fileName = "dataOut/test/Step2"
            # dataOutput.saveGrid(fileName,mP.subDomain,ind)
    
            # Step 3 - Check if Points were Marked
            continueFlag = eqDist.checkPoints(ind,1,True)

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
                    
                    # fileName = "dataOut/drain/NWset"+str(p)
                    # dataOutput.saveSetData(fileName,mP.subDomain,eqDist,**setSaveDict)
                    
                    # fileName = "dataOut/test/nwGrid"+str(p)
                    # dataOutput.saveGrid(fileName,mP.subDomain,nwGrid)

                # Step 3b and 3d- Check if W Phases Exists then Collect W Sets
                wCheck = eqDist.checkPoints(mP.mpGrid,mP.wID)
                if wCheck:
 
                    wSets,wSetCount = sets.collectSets(mP.mpGrid,mP.wID,mP.inlet[mP.wID],mP.outlet[mP.wID],mP.loopInfo[mP.wID],mP.subDomain)
                    wGrid = eqDist.getInletConnectedNodes(wSets,1)
                    
                    
                    # setSaveDict = {'inlet': 'inlet',
                    #                'outlet':'outlet',
                    #                'boundary': 'boundary',
                    #                'localID': 'localID'}

                    # eqDist.Sets = wSets
                    # eqDist.setCount = wSetCount

                    # dataOutput.saveSetData("dataOut/Wset",mP.subDomain,eqDist,**setSaveDict)

                    # fileName = "dataOut/test/wGrid"+str(p)
                    # dataOutput.saveGrid(fileName,mP.subDomain,wGrid)

                # Steb 3c and 3d - Already checked at Step 3 so Collect Sets with ID = 1
                indSets,indSetCount = sets.collectSets(ind,1,mP.inlet[mP.nwID],mP.outlet[mP.nwID],mP.loopInfo[mP.nwID],mP.subDomain)
                ind2 = eqDist.getInletConnectedNodes(indSets,1)
                # fileName = "dataOut/test/Step3c"
                # dataOutput.saveGrid(fileName,mP.subDomain,ind2)
            
                # Step 3e - no Step 3e ha. 

                # Step 3f -- Unsure about these checks!
                if nwCheck:
                    ind = np.where( (ind2 != 1) & (nwGrid != 1),0,ind).astype(np.uint8)
                    morph = morphology.morph(ind,mP.subDomain,eqDist.probeR)
                else:
                    morph = morphology.morph(ind2,mP.subDomain,eqDist.probeR)

                # Step 3g
                # fileName = "dataOut/test/Step3g"
                # dataOutput.saveGrid(fileName,mP.subDomain,morph)
            
                ## Turn wetting films on or off here
                mP.mpGrid = np.where( (morph == 1) & (wGrid == 1),mP.nwID,mP.mpGrid)  ### films off
                #mP.mpGrid = np.where( (morph == 1),mP.nwID,mP.mpGrid)                ### films on

                # Step 4
                sw = eqDist.calcSaturation(mP.mpGrid,mP.nwID)
                if mP.subDomain.ID == 0:
                    print("Capillary pressure: %e Wetting Phase Saturation: %e" %(p,sw))
                    result.append(sw)

            if save:
                fileName = "dataOut/twoPhase/twoPhase_drain_pc_"+str(p)
                dataOutput.saveGrid(fileName,mP.subDomain,mP.mpGrid)        

    return eqDist, result

def calcDrainageSW(sW,mP,interval):

    ### Get Distance from Solid to Pore Space (Ignore Fluid Phases)
    poreSpaceDist = distance.calcEDT(mP.subDomain,mP.porousMedia.grid)
    eqDist = equilibriumDistribution(mP)
    save = True
    
    ## Make sure sw targets are ordered largest to smallest
    sW.sort(reverse=True)

    ## Find intial radius target (largest EDT value)
    rad_temp = np.amax(poreSpaceDist[:,:,:])
    rad = np.array([rad_temp])
    comm.Allreduce(MPI.IN_PLACE, rad, op=MPI.MAX)

    minrad = 0.0000000000000001  ##fix this, want half voxel in physical units
    sW_new = 1.0

    for s in sW:

        while sW_new > s and rad[0] > minrad:

            if rad[0] > 0:
                p = 2/rad[0]
            else:
                p = 0

            ### Get Sphere Radius from Pressure
            eqDist.getDiameter(p)
            
            # Step 2 - Dilate Solid Phase and Flag Allowable Fluid Voxes as 1 
            ind = np.where( (poreSpaceDist >= eqDist.probeR) & (mP.porousMedia.grid == 1),1,0).astype(np.uint8)

            # Step 3 - Check if Points were Marked
            continueFlag = eqDist.checkPoints(ind,1,True)
            if continueFlag:

                # Step 3a and 3d - Check if NW Phases Exists then Collect NW Sets
                nwCheck = eqDist.checkPoints(mP.mpGrid,mP.nwID)
                if nwCheck:
                    nwSets,nwSetCount = sets.collectSets(mP.mpGrid,mP.nwID,mP.inlet[mP.nwID],mP.outlet[mP.nwID],mP.loopInfo[mP.nwID],mP.subDomain)
                    nwGrid = eqDist.getInletConnectedNodes(nwSets,1)

                # Step 3b and 3d- Check if W Phases Exists then Collect W Sets
                wCheck = eqDist.checkPoints(mP.mpGrid,mP.wID)
                if wCheck:
                    wSets,wSetCount = sets.collectSets(mP.mpGrid,mP.wID,mP.inlet[mP.wID],mP.outlet[mP.wID],mP.loopInfo[mP.wID],mP.subDomain)
                    wGrid = eqDist.getInletConnectedNodes(wSets,1)

                # Steb 3c and 3d - Already checked at Step 3 so Collect Sets with ID = 1
                indSets,indSetCount = sets.collectSets(ind,1,mP.inlet[mP.nwID],mP.outlet[mP.nwID],mP.loopInfo[mP.nwID],mP.subDomain)
                ind2 = eqDist.getInletConnectedNodes(indSets,1)

                # Step 3f -- Unsure about these checks!
                if nwCheck:
                    ind = np.where( (ind2 != 1) & (nwGrid != 1),0,ind).astype(np.uint8)
                    # Step 3g
                    morph = morphology.morph(ind,mP.subDomain,eqDist.probeR)
                else:
                    morph = morphology.morph(ind2,mP.subDomain,eqDist.probeR)

                ## Turn wetting films on or off here
                mP.mpGrid = np.where( (morph == 1) & (wGrid == 1),mP.nwID,mP.mpGrid)  ##films off
                #mP.mpGrid = np.where( (morph == 1),mP.nwID,mP.mpGrid)                ##films on


            # Step 4
            sW_new = eqDist.calcSaturation(mP.mpGrid,mP.nwID)

            if mP.subDomain.ID == 0:
                if sW_new <= s:
                    print("SAVE Capillary pressure: %e Radius: %e Wetting Phase Saturation: %e Target Saturation: %e" %(p,rad[0],sW_new,s))
                else:
                    print("SKIP Capillary pressure: %e Radius: %e Wetting Phase Saturation: %e Target Saturation: %e" %(p,rad[0],sW_new,s))

            rad[0] *= interval
            
        if save:
            fileName = "dataOut/twoPhase/twoPhase_drain_sw_"+str(sW_new)
            dataOutput.saveGrid(fileName,mP.subDomain,mP.mpGrid)      

    return eqDist


def calcOpenSW(sW,mP,interval,minSetSize):

    ### Get Distance from Solid to Pore Space (Ignore Fluid Phases)

    poreSpaceDist = distance.calcEDT(mP.subDomain,mP.porousMedia.grid)

    eqDist = equilibriumDistribution(mP)
 
    save = True  #Save result?
    
    ## Make sure sw targets are ordered largest to smallest
    sW.sort(reverse=True)

    ## Find intial radius target (largest EDT value)
    rad_temp = np.amax(poreSpaceDist[:,:,:])
    rad = np.array([rad_temp])
    comm.Allreduce(MPI.IN_PLACE, rad, op=MPI.MAX)

    minrad = np.min([mP.Domain.dX,mP.Domain.dY,mP.Domain.dZ])/2. ## TMW Fixed ##fix this, want half voxel in physical units
    sW_new = 1.0


    for s in sW:

        while sW_new > s and rad[0] > minrad:

            if rad[0] > 0:
                p = 2/rad[0]
            else:
                p = 0

            ### Get Sphere Radius from Pressure
            eqDist.getDiameter(p)
            
            # Step 2 - Dilate Solid Phase and Flag Allowable Fluid Voxes as 1 
            ind = np.where( (poreSpaceDist >= eqDist.probeR) & (mP.porousMedia.grid == 1),1,0).astype(np.uint8)

            # Step 3 - Check if Points were Marked
            continueFlag = eqDist.checkPoints(ind,1,True)
            if continueFlag:

                # Step 3g
                morph = morphology.morph(ind,mP.subDomain,eqDist.probeR)

                mP.mpGrid = np.where( (morph == 1),mP.nwID,mP.mpGrid).astype(np.uint8)
                
                if minSetSize > 0:
                    wCheck = eqDist.checkPoints(mP.mpGrid,mP.wID)
                    if wCheck:
                        sW_new = eqDist.calcSaturation(mP.mpGrid,mP.nwID)
                        wSets,wSetCount = sets.collectSets(mP.mpGrid,mP.wID,mP.inlet[mP.wID],mP.outlet[mP.wID],mP.loopInfo[mP.wID],mP.subDomain)
                        mP.mpGrid = eqDist.removeSmallSets(wSets,mP.mpGrid,mP.nwID,minSetSize)
            
            # Step 4
            sW_new = eqDist.calcSaturation(mP.mpGrid,mP.nwID)

            if mP.subDomain.ID == 0:
                if sW_new <= s:
                    print("SAVE Capillary pressure: %e Radius: %e Wetting Phase Saturation: %e Target Saturation: %e" %(p,rad[0],sW_new,s))

                else:
                    print("SKIP Capillary pressure: %e Radius: %e Wetting Phase Saturation: %e Target Saturation: %e" %(p,rad[0],sW_new,s))

            rad[0] *= interval

        if save:
            fileName = "dataOut/Open/twoPhase_open_sw_"+str(s)
            dataOutput.saveGrid(fileName,mP.subDomain,mP.porousMedia.grid)      
                
            fileName = "dataOut/OpenCSV/twoPhase_open_sw_"+str(s)
            dataOutput.saveGridcsv(fileName,mP.subDomain,mP.subDomain.x,mP.subDomain.y,mP.subDomain.z,mP.mpGrid,removeHalo = True)

                    
    return eqDist


def calcImbibition(pc,mP):

    ### Get Distance from Solid to Pore Space (Ignore Fluid Phases)
    poreSpaceDist = distance.calcEDT(mP.subDomain,mP.porousMedia.grid)
    eqDist = equilibriumDistribution(mP)
    save = True
    
    # fileName = "dataOut/test/porespacedist"
    # dataOutput.saveGrid(fileName,mP.subDomain,poreSpaceDist)
    
    ## Make sure pc targets are ordered largest to smallest
    pc.sort(reverse=True)
    
    sW = eqDist.calcSaturation(mP.mpGrid,2)

    # fileName = "dataOut/test/Input"
    # dataOutput.saveGrid(fileName,mP.subDomain,mP.mpGrid)
    result = []
    ### Loop through all Pressures
    for p in pc:
        # print(p)
        if p == 0:
            sW = 1
        else:
            ### Get Sphere Radius from Pressure
            eqDist.getDiameter(p)
            
            gridCopy = np.copy(mP.mpGrid)
            ## A Locate any disconnected nonwetting phase and make it a 4
            nwCheck = eqDist.checkPoints(gridCopy,mP.nwID)
            if nwCheck:
                ## nw pts < pore space pts?
                own = mP.subDomain.ownNodesIndex
                ownGrid =  gridCopy[own[0]:own[1],
                                own[2]:own[3],
                                own[4]:own[5]]
                nwNodes = np.count_nonzero(ownGrid==1)
                allnwNodes = np.zeros(1,dtype=np.uint64)
                comm.Allreduce( [np.int64(nwNodes), MPI.INT], [allnwNodes, MPI.INT], op = MPI.SUM )

                if allnwNodes[0] != mP.porousMedia.totalPoreNodes[0]:
                    ## run connected sets, otherwise skip
   
                    nwSets,nwSetCount = sets.collectSets(mP.mpGrid,mP.nwID,mP.inlet[mP.nwID],mP.outlet[mP.nwID],mP.loopInfo[mP.nwID],mP.subDomain)
                    nwGrid = eqDist.getDisconnectedNodes(nwSets,1)
                    
                    # setSaveDict = {'inlet': 'inlet',
                    # 'outlet':'outlet',
                    # 'boundary': 'boundary',
                    # 'localID': 'localID'}

                    # eqDist.Sets = nwSets
                    # eqDist.setCount = nwSetCount
                    
                    # fileName = "dataOut/test/nwGrid_imbibe_pc_"+str(p)
                    # dataOutput.saveGrid(fileName,mP.subDomain,nwGrid)
                    
                    # fileName = "dataOut/imbibe/NWset"+str(p)
                    # dataOutput.saveSetData(fileName,mP.subDomain,eqDist,**setSaveDict)
                    
                    ## A1 Save copy of current result, disconnected nonwetting = 4
                    gridCopy = np.where( (nwGrid == 1),4,mP.mpGrid).astype(np.uint8)
                    
                    # fileName = "dataOut/test/gridCopy_imbibe_pc_"+str(p)
                    # dataOutput.saveGrid(fileName,mP.subDomain,gridCopy)

            mP.mpGrid = np.where( (mP.mpGrid == 2),1,mP.mpGrid).astype(np.uint8)
            # fileName = "dataOut/test/mpGrid"
            # dataOutput.saveGrid(fileName,mP.subDomain,mP.mpGrid)
            
            ## B Locate everywhere with poreSpaceDist >= radius AND mpGrid == nonwetting phase (0: True, 1: False)
            ind = np.where( (poreSpaceDist >= eqDist.probeR) & (gridCopy == 1),1,0).astype(np.uint8)

            
            ## C Check if any points were found in B, only proceed if true
            indCheck = eqDist.checkPoints(ind,1)
            if indCheck:
                ## D Get EDT of B
                morph = morphology.morph(ind,mP.subDomain,eqDist.probeR)

                ## E Locate everywhere in porespace where D <= radius, make it a 3 in running (not saved) mpGrid
                mP.mpGrid = np.where( (morph == 1),3,mP.mpGrid).astype(np.uint8)

                ## F Get CC (w) connected to w reservoir on mpGrid
                wCheck = eqDist.checkPoints(mP.mpGrid,mP.nwID)
                if wCheck:
                    wSets,wSetCount = sets.collectSets(mP.mpGrid,mP.nwID,mP.inlet[mP.wID],mP.outlet[mP.wID],mP.loopInfo[mP.wID],mP.subDomain)
                    wGrid = eqDist.getInletConnectedNodes(wSets,1)

                    ## G If porespace point NOT in F set AND result prev is NOT 2 (w), mpGrid = n ELSE mpGrid = w

                    mP.mpGrid = np.where( (wGrid != 1) & (gridCopy != 2),1,2).astype(np.uint8)

                    ##put disconnected n (4) back in
                    mP.mpGrid = np.where( (gridCopy == 4),1,mP.mpGrid).astype(np.uint8)
                    ##put the solid back in
                    mP.mpGrid = np.where( (gridCopy == 0),0,mP.mpGrid).astype(np.uint8)

            else:
                ##disconnected n should remain, otherwise everything goes to w
                mP.mpGrid = np.where( (mP.mpGrid == 1) & (gridCopy != 4),2,mP.mpGrid).astype(np.uint8)
            

            sw = eqDist.calcSaturation(mP.mpGrid,mP.nwID)
            if mP.subDomain.ID == 0:
                print("Capillary pressure: %e Wetting Phase Saturation: %e" %(p,sw))
                result.append(sw)

            if save:
                fileName = "dataOut/twoPhase/twoPhase_imbibe_pc_"+str(p)
                dataOutput.saveGrid(fileName,mP.subDomain,mP.mpGrid)   
                

    return eqDist,result