import numpy as np
from mpi4py import MPI
from pmmoto.core import Orientation
from pmmoto.analysis import stats

comm = MPI.COMM_WORLD

class PorousMedia:
    """
    Porous media class 
    """
    def __init__(self,subdomain,grid):
        self.subdomain = subdomain
        self.grid = grid
        self.inlet = np.zeros([Orientation.num_faces],dtype = np.uint8)
        self.outlet = np.zeros([Orientation.num_faces],dtype = np.uint8)
        self.loop_info = np.zeros([Orientation.num_faces+1,3,2],dtype = np.int64)
        self.porosity = None

    def set_inlet_outlet(self,res_size):
        """
        Determine inlet/outlet info and pad grid but only inlet!
        """
        inlet_size = np.zeros_like(self.inlet)

        for n in range(0,self.subdomain.domain.dims):
            if (self.subdomain.boundary_type[n*2] == 0): # Make sure boundary type is zero
                
                if self.subdomain.domain.inlet[n][0]:
                    self.inlet[n*2] = True
                    inlet_size[n*2]  = res_size
                
                if self.subdomain.domain.outlet[n][0]:
                    self.outlet[n*2] = True

            if (self.subdomain.boundary_type[n*2+1] == 0): # Make sure boundary type is zero
                
                if self.subdomain.domain.inlet[n][1]:
                    self.inlet[n*2+1] = True
                    inlet_size[n*2+1]  = res_size

                if self.subdomain.domain.outlet[n][1]:
                    self.outlet[n*2+1] = True

        pad = np.zeros([self.subdomain.domain.dims*2],dtype = np.int8)
        for f in range(0,Orientation.num_faces):
            pad[f] = inlet_size[f]
        
        ### If Inlet/Outlet Res, Pad and Update XYZ
        if np.sum(pad) > 0:
            self.grid = np.pad(self.grid, ( (pad[0], pad[1]), (pad[2], pad[3]), (pad[4], pad[5]) ), 'constant', constant_values=1)
            self.subdomain.get_coordinates(pad)

    def set_wall_bcs(self):
        """
        If wall boundary conditions are specified, force solid on external boundaries
        """
        if self.subdomain.boundary_type[0] == 1:
            self.grid[0,:,:] = 0
        if self.subdomain.boundary_type[1] == 1:
            self.grid[-1,:,:] = 0
        if self.subdomain.boundary_type[2] == 1:
            self.grid[:,0,:] = 0
        if self.subdomain.boundary_type[3] == 1:
            self.grid[:,-1,:] = 0
        if self.subdomain.boundary_type[4] == 1:
            self.grid[:,:,0] = 0
        if self.subdomain.boundary_type[5] == 1:
            self.grid[:,:,-1] = 0

    def get_porosity(self):
        """
        Calalcaute the porosity of porous media grid 
        """
        self.porosity = 1. - stats.get_volume_fraction(self.subdomain,self.grid,0)


def gen_pm(subdomain,grid,res_size = 0):
    """
    Initialize the porousmedia class and set inlet/outlet/wall bcs
    Gather loop_info for efficient looping
    """
    pm = PorousMedia(subdomain = subdomain, grid = grid)
    pm.set_inlet_outlet(res_size)
    pm.set_wall_bcs()
    pm.loop_info = Orientation.get_loop_info(pm.grid,subdomain,pm.inlet,pm.outlet,res_size)

    return pm
