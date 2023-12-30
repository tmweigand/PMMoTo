import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
from .. import dataOutput
from .. import dataRead
from ..core import Orientation

#### TO DO: Make phase ID generic
# 0 is always solid
# 1 is Wetting Phase
# 2 is NonWetting Phase


class multiPhase(object):
    def __init__(self,porousmedia,num_phases):
        self.porousmedia  = porousmedia
        self.subdomain  = porousmedia.subdomain
        self.num_phases = num_phases
        self.fluid_ID = list(range(1,num_phases+1))
        self.index_own_nodes  = {}
        self.mp_grid = None
        self.w_ID = 2
        self.nw_ID = 1
        self.inlet = {}
        self.outlet = {}
        self.loop_info = {}

    def initialize_mp_grid(self,constant_phase = -1,input_file = None):
        """
        Set The initial distribution of fluids. 
        If fully saturated by a given phase, set constantPhase = phaseID
        If inputFile, read data
        Else set poreSpace to 1 and warn
        """
        if constant_phase > -1:
            self.mp_grid = np.where(self.porousmedia.grid == 1,constant_phase,0).astype(np.uint8)
        elif input_file is not None:
            self.mp_grid = dataRead.readVTKGrid(self.subdomain.ID,self.subdomain.domain.numSubDomains,input_file)
        else:
            self.mp_grid = np.copy(self.porousmedia.grid)
            if self.subdomain.ID:
                print("No input Parameter given. Setting phase distribution to 1")

    def save_mp_grid(self,file_name):
        """
        """
        dataOutput.saveMultiPhaseData(file_name,self.subdomain.ID,self.subdomain.domain,self.subdomain,self)

    def get_boundary_info(self,inlets,outlets,res_size):
        """
        Determine Inlet/Outlet for Each Fluid Phase
        TO DO: Optimize loopInfo so phases dont loop over other phase reservoirs
        """
        pad = np.zeros([self.num_phases,6],dtype = np.int8)
        for fN,fluid in enumerate(self.fluid_ID):
            ### INLET ###
            self.inlet[fluid] = np.zeros([Orientation.num_faces],dtype = np.int8)
            if (self.subdomain.boundary_ID[0] == 0 and inlets[fluid][0][0]):
                self.inlet[fluid][0] = res_size
            if (self.subdomain.boundary_ID[1] == 0 and inlets[fluid][0][1]):
                self.inlet[fluid][1] = res_size
            if (self.subdomain.boundary_ID[2] == 0 and inlets[fluid][1][0]):
                self.inlet[fluid][2] = res_size
            if (self.subdomain.boundary_ID[3] == 0 and inlets[fluid][1][1]):
                self.inlet[fluid][3] = res_size
            if (self.subdomain.boundary_ID[4] == 0 and inlets[fluid][2][0]):
                self.inlet[fluid][4] = res_size
            if (self.subdomain.boundary_ID[5] == 0 and inlets[fluid][2][1]):
                self.inlet[fluid][5] = res_size

            ### OUTLET ###
            self.outlet[fluid] = np.zeros([Orientation.num_faces],dtype = np.int8)
            if (self.subdomain.boundary_ID[0] == 0 and outlets[fluid][0][0]):
                self.outlet[fluid][0] = res_size
            if (self.subdomain.boundary_ID[1] == 0 and outlets[fluid][0][1]):
                self.outlet[fluid][1] = res_size
            if (self.subdomain.boundary_ID[2] == 0 and outlets[fluid][1][0]):
                self.outlet[fluid][2] = res_size
            if (self.subdomain.boundary_ID[3] == 0 and outlets[fluid][1][1]):
                self.outlet[fluid][3] = res_size
            if (self.subdomain.boundary_ID[4] == 0 and outlets[fluid][2][0]):
                self.outlet[fluid][4] = res_size
            if (self.subdomain.boundary_ID[5] == 0 and outlets[fluid][2][1]):
                self.outlet[fluid][5] = res_size
    
            ### Only Pad Inlet 
            for f in range(0,Orientation.num_faces):
                pad[fN,f] = self.inlet[fluid][f]      

            
            ### If Inlet/Outlet Res, Pad and Update XYZ
            if np.sum(pad[fN]) > 0:
                self.mp_grid = np.pad(self.mp_grid, ( (pad[fN,0], pad[fN,1]),
                                                    (pad[fN,2], pad[fN,3]),
                                                    (pad[fN,4], pad[fN,5]) ),
                                                    'constant', constant_values = fluid)
                
                self.porousmedia.grid = np.pad( self.porousmedia.grid , ( (pad[fN,0], pad[fN,1]),
                                                                          (pad[fN,2], pad[fN,3]),
                                                                          (pad[fN,4], pad[fN,5]) ), 
                                                                          'constant', constant_values = 1)
                
            ### Update Subdomain Information     
            self.subdomain.get_coordinates_mulitphase(pad[fN],inlets[fluid],res_size)


        for fN,fluid in enumerate(self.fluid_ID):
            self.loop_info[fluid] = Orientation.get_loop_info(self.mp_grid,self.subdomain,self.inlet[fluid],self.outlet[fluid],res_size)
            
            ### Get own nodes including inlet for fluids
            self.index_own_nodes[fluid] = np.copy(self.subdomain.index_own_nodes)
            if pad[fN,0] > 0:
                self.index_own_nodes[fluid][0] = self.subdomain.index_own_nodes[0] - res_size
            if pad[fN,1] > 0:
                self.index_own_nodes[fluid][1] = self.subdomain.index_own_nodes[1] + res_size
            if pad[fN,2] > 0:
                self.index_own_nodes[fluid][2] = self.subdomain.index_own_nodes[2] - res_size
            if pad[fN,3] > 0:
                self.index_own_nodes[fluid][3] = self.subdomain.index_own_nodes[3] + res_size
            if pad[fN,4] > 0:
                self.index_own_nodes[fluid][4] = self.subdomain.index_own_nodes[4] - res_size
            if pad[fN,5] > 0:
                self.index_own_nodes[fluid][5] = self.subdomain.index_own_nodes[5] + res_size
