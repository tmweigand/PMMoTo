import math
import numpy as np
cimport numpy as cnp
cimport cython
from libc.stdlib cimport malloc, free

from mpi4py import MPI
comm = MPI.COMM_WORLD
from pmmoto.filters import distance
from pmmoto.filters import morphology
from pmmoto.core import sets
from pmmoto.io import dataOutput
from pmmoto.core import Orientation


class equilibriumDistribution(object):
    def __init__(self,multiphase):
        self.multiphase  = multiphase
        self.subdomain   = multiphase.subdomain
        self.porousmedia = multiphase.porousmedia
        self.gamma = 1
        self.d_probe = 0
        self.r_probe = 0
        self.pc = 0

    def get_diameter(self,pc):
        if pc == 0:
            self.d_probe = 0
            self.r_probe = 0
        else:
            self.r_probe = 2.*self.gamma/pc
            self.d_probe = 2.*self.r_probe

    def get_pc(self,radius):
        self.pc = 2.*self.gamma/radius

    def get_inlet_connected_nodes(self,Sets,flag):
        """
        Grab from Sets that are on the Inlet Reservoir and create binary grid
        """
        nodes = []
        _grid_out = np.zeros_like(self.multiphase.mp_grid)

        for s in Sets.sets:
            if s.inlet:
                for node in s.nodes:
                    nodes.append(node)

        for n in nodes:
            _grid_out[n[0],n[1],n[2]] = flag

        return _grid_out

    def get_disconnected_nodes(self,sets,flag):
        """
        Grab from Sets that are on the Inlet Reservoir and create binary grid
        """
        nodes = []
        _grid_out = np.zeros_like(self.multiphase.mp_grid)

        for s in sets:
            if not s.inlet:
                for node in s.nodes:
                    nodes.append(node)

        for n in nodes:
            _grid_out[n[0],n[1],n[2]] = flag

        return _grid_out
    
    def remove_small_sets(self,sets,grid_in,nw_ID,min_set_size):
        """
        Remove sets smaller than target size
        """
        nodes = []
        _grid_out = np.copy(grid_in)

        for s in sets:
            # print(s.numNodes)
            if s.numGlobalNodes < min_set_size:
                for node in s.nodes:
                    nodes.append(node)
                    
        for n in nodes:
            _grid_out[n[0],n[1],n[2]] = nw_ID

        return _grid_out
    
    def drain_info(self,max_dist,min_dist):
        """
        """
        self.get_pc(max_dist)
        print("Minimum pc",self.pc)
        self.get_pc(min_dist)
        print("Maximum pc",self.pc)

    def calc_saturation(self,grid,nw_ID):
        """
        """
        _own = self.subdomain.index_own_nodes
        _own_grid =  grid[_own[0]:_own[1],
                          _own[2]:_own[3],
                          _own[4]:_own[5]]
        _nw_nodes = np.count_nonzero(_own_grid==nw_ID)
        _all_nw_nodes = np.zeros(1,dtype=np.uint64)
        comm.Allreduce( [np.int64(_nw_nodes), MPI.INT], [_all_nw_nodes, MPI.INT], op = MPI.SUM )
        s_w = 1. - _all_nw_nodes[0]/self.porousmedia.total_pore_nodes[0]
        return s_w


    def check_points(self,grid,ID,include_inlet = False):
        """
        Check to make sure nodes of type ID exist in domain
        """
        if include_inlet:
            _own = self.multiphase.index_own_nodes[ID]
        else:
            _own = self.subdomain.index_own_nodes

        no_points = False
        if ID == 0:
            count = np.size(grid) - np.count_nonzero(grid > 0)
        else:
            _own_grid =  grid[_own[0]:_own[1],
                            _own[2]:_own[3],
                            _own[4]:_own[5]]
            _count = np.count_nonzero(_own_grid==ID)

        _all_count = np.zeros(1,dtype=np.uint64)
        comm.Allreduce( [np.int64(_count), MPI.INT], [_all_count, MPI.INT], op = MPI.SUM )

        if _all_count > 0:
            no_points =  True

        return no_points


def calc_drainage(pc,mp):

    ### Get Distance from Solid to Pore Space (Ignore Fluid Phases)
    dist_pm = distance.calcEDT(mp.subdomain,mp.porousmedia.grid)

    eq_dist = equilibriumDistribution(mp)
    sw = eq_dist.calc_saturation(mp.mp_grid,2)
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
            sw = 1
        else:
            ### Get Sphere Radius from Pressure
            eq_dist.get_diameter(p)

            # Step 1 - Reservoirs are not contained in mpGrid or grid but rather added when needed so this step is unnecessary
            
            # Step 2 - Dilate Solid Phase and Flag Allowable Fluid Voxes as 1 
            ind = np.where( (dist_pm >= eq_dist.r_probe) & (mp.porousmedia.grid == 1),1,0).astype(np.uint8)
            # fileName = "dataOut/test/Step2"
            # dataOutput.saveGrid(fileName,mP.subDomain,ind)
    
            # Step 3 - Check if Points were Marked
            continue_flag = eq_dist.check_points(ind,1,True)

            if continue_flag:

                # Step 3a and 3d - Check if NW Phases Exists then Collect NW Sets
                nwCheck = eq_dist.check_points(mp.mp_grid,mp.nw_ID)
                if nwCheck:
 
                    _nw_sets,nw_set_count = sets.collect_sets(mp.mp_grid,mp.nw_ID,mp.inlet[mp.nw_ID],mp.outlet[mp.nw_ID],mp.loop_info[mp.nw_ID],mp.subsomain)
                    _nw_grid = eq_dist.get_inlet_connected_nodes(_nw_sets,1)

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
                w_check = eq_dist.check_points(mp.mp_grid,mp.w_ID)
                if w_check:
 
                    _w_sets = sets.collect_sets(mp.mp_grid,mp.w_ID,mp.inlet[mp.w_ID],mp.outlet[mp.w_ID],mp.loop_info[mp.w_ID],mp.subdomain)
                    _w_grid = eq_dist.get_inlet_connected_nodes(_w_sets,1)
                    
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
                _ind_sets = sets.collect_sets(ind,1,mp.inlet[mp.nw_ID],mp.outlet[mp.nw_ID],mp.loop_info[mp.nw_ID],mp.subdomain)
                _ind_2 = eq_dist.get_inlet_connected_nodes(_ind_sets,1)
                # fileName = "dataOut/test/Step3c"
                # dataOutput.saveGrid(fileName,mP.subDomain,ind2)
            
                # Step 3e - no Step 3e ha. 

                # Step 3f -- Unsure about these checks!
                if nwCheck:
                    ind = np.where( (_ind_2 != 1) & (_nw_grid != 1),0,ind).astype(np.uint8)
                    morph = morphology.morph_add(mp.subDomain,ind,0,eq_dist.r_probe)
                else:
                    morph = morphology.morph_add(mp.subdomain,_ind_2,0,eq_dist.r_probe)

                # Step 3g
                # fileName = "dataOut/test/Step3g"
                # dataOutput.saveGrid(fileName,mP.subDomain,morph)
            
                ## Turn wetting films on or off here
                mp.mp_Grid = np.where( (morph == 1) & (_w_grid == 1),mp.nw_ID,mp.mp_grid)  ### films off
                #mP.mpGrid = np.where( (morph == 1),mP.nwID,mP.mpGrid)                ### films on

                # Step 4
                sw = eq_dist.calc_saturation(mp.mp_grid,mp.nw_ID)
                if mp.subdomain.ID == 0:
                    print("Capillary pressure: %e Wetting Phase Saturation: %e" %(p,sw))
                    result.append(sw)

            if save:
                fileName = "dataOut/twoPhase/twoPhase_drain_pc_"+str(p)
                dataOutput.saveGrid(fileName,mp.subdomain,mp.mp_grid)        

    return eq_dist, result

def calcDrainageSW(sW,mP,interval):

    ### Get Distance from Solid to Pore Space (Ignore Fluid Phases)
    poreSpaceDist = distance.calcEDT(mP.subdomain,mP.porousMedia.grid)
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
            eqDist.get_diameter(p)
            
            # Step 2 - Dilate Solid Phase and Flag Allowable Fluid Voxes as 1 
            ind = np.where( (poreSpaceDist >= eqDist.probeR) & (mP.porousMedia.grid == 1),1,0).astype(np.uint8)

            # Step 3 - Check if Points were Marked
            continueFlag = eqDist.checkPoints(ind,1,True)
            if continueFlag:

                # Step 3a and 3d - Check if NW Phases Exists then Collect NW Sets
                nwCheck = eqDist.checkPoints(mP.mpGrid,mP.nwID)
                if nwCheck:
                    nwSets,nwSetCount = sets.collect_sets(mP.mpGrid,mP.nwID,mP.inlet[mP.nwID],mP.outlet[mP.nwID],mP.loopInfo[mP.nwID],mP.subDomain)
                    nwGrid = eqDist.getInletConnectedNodes(nwSets,1)

                # Step 3b and 3d- Check if W Phases Exists then Collect W Sets
                wCheck = eqDist.checkPoints(mP.mpGrid,mP.wID)
                if wCheck:
                    wSets,wSetCount = sets.collect_sets(mP.mpGrid,mP.wID,mP.inlet[mP.wID],mP.outlet[mP.wID],mP.loopInfo[mP.wID],mP.subDomain)
                    wGrid = eqDist.getInletConnectedNodes(wSets,1)

                # Steb 3c and 3d - Already checked at Step 3 so Collect Sets with ID = 1
                indSets,indSetCount = sets.collect_sets(ind,1,mP.inlet[mP.nwID],mP.outlet[mP.nwID],mP.loopInfo[mP.nwID],mP.subDomain)
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
 
    save = False  #Save result?
    
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
            eqDist.get_diameter(p)
            
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
                        wSets,wSetCount = sets.collect_sets(mP.mpGrid,mP.wID,mP.inlet[mP.wID],mP.outlet[mP.wID],mP.loopInfo[mP.wID],mP.subDomain)
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
            eqDist.get_diameter(p)
            
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
   
                    nwSets,nwSetCount = sets.collect_sets(mP.mpGrid,mP.nwID,mP.inlet[mP.nwID],mP.outlet[mP.nwID],mP.loopInfo[mP.nwID],mP.subDomain)
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
                    wSets,wSetCount = sets.collect_sets(mP.mpGrid,mP.nwID,mP.inlet[mP.wID],mP.outlet[mP.wID],mP.loopInfo[mP.wID],mP.subDomain)
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