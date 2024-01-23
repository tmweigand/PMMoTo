"""Multiphase.py"""
import numpy as np
from . import Orientation
from . import utils

__all__ = [
    "get_probe_radius",
    "get_pc",
    "initialize_mp"
]

def get_probe_radius(pc,gamma):
    """
    Return the probe radius givena capillary pressure and surface tension
    """
    if pc == 0:
        r_probe = 0
    else:
        r_probe = 2.*gamma/pc
    return r_probe

def get_pc(radius,gamma):
    """
    Return the capillary pressure given a surface tension and radius
    """
    return 2.*gamma/radius


class Multiphase:
    """Multiphase Class"""
    def __init__(self,porousmedia,num_phases,res_size = 0):
        self.porousmedia  = porousmedia
        self.subdomain  = porousmedia.subdomain
        self.num_phases = num_phases
        self.res_size = res_size
        self.fluids = list(range(1,num_phases+1))
        self.index_own_nodes  = {}
        self.grid = None
        self.inlet = {}
        self.outlet = {}
        self.loop_info = {}

    def set_inlet_outlet(self,inlets,outlets):
        """
        Determine Inlet/Outlet for each fluid phase
        """
        for fluid in self.fluids:
            self.inlet[fluid] = np.zeros([Orientation.num_faces],dtype = np.int8)
            self.outlet[fluid] = np.zeros([Orientation.num_faces],dtype = np.int8)
            for n in range(0,self.subdomain.domain.dims):
                if (self.subdomain.boundary_ID[n*2] == 0 and inlets[fluid][n][0]):
                    self.inlet[fluid][n*2] = True
                if (self.subdomain.boundary_ID[n*2+1] == 0 and inlets[fluid][n][1]):
                    self.inlet[fluid][n*2+1] = True
                if (self.subdomain.boundary_ID[n*2] == 0 and outlets[fluid][n][0]):
                    self.outlet[fluid][n*2] = True
                if (self.subdomain.boundary_ID[n*2+1] == 0 and outlets[fluid][n][1]):
                    self.outlet[fluid][n*2+1] = True

    def create_reservoir(self):
        """
        Pad the grid is inlet and res_size > 0 and update coordinates
        """
        pad = np.zeros([self.num_phases,6],dtype = np.int8)
        for n_fluid,fluid in enumerate(self.fluids):
            for n_face in range(0,Orientation.num_faces):
                if self.inlet[fluid][n_face]:
                    pad[n_fluid,n_face] = self.res_size

            # If Inlet/Outlet Res, Pad and Update XYZ
            if np.sum(pad[n_fluid]) > 0:
                self.grid = utils.constant_pad(self.grid,pad[n_fluid],fluid)
                self.porousmedia.grid = utils.constant_pad(self.porousmedia.grid ,pad[n_fluid],1)
                
            # Update subdomain information
            self.subdomain.get_coordinates(pad[n_fluid])

    def get_loop_info(self):
        """
        Get the multphase class loop_info
        TO DO: Optimize loop_info so phases dont loop over other phase reservoirs
        """
        for fluid in self.fluids:
            self.loop_info[fluid] = np.copy(self.porousmedia.loop_info)
            self.loop_info[fluid] = Orientation.get_loop_info(self.grid,self.subdomain,self.inlet[fluid],self.outlet[fluid],self.res_size)
            

    def get_index_own_nodes(self):
        """
        Get index_own_nodes for the phaes which may include the inlet reservoirs
        """
        sd = self.subdomain
        for fluid in self.fluids:
            self.index_own_nodes[fluid] = np.copy(sd.index_own_nodes)
            for n in range(0,sd.domain.dims):
                if self.inlet[fluid][n*2]:
                    self.index_own_nodes[fluid][n*2] = sd.index_own_nodes[n*2] - self.res_size
                if self.inlet[fluid][n*2+1]:
                    self.index_own_nodes[fluid][n*2+1] = sd.index_own_nodes[n*2+1] + self.res_size

    def update_grid(self):
        """
        Once grid is created, update necessary parameters including create reserovir, 
        loop_info, and index_own_nodes
        """
        self.create_reservoir()
        self.get_loop_info()
        self.get_index_own_nodes()


def initialize_mp(porousmedia,num_phases,inlets,outlets,res_size = 0):
    """
    Initialize the multiphase class and set inlet/outlet reservoirs, get loop_info and own_nodes
    """
    mp = Multiphase(porousmedia = porousmedia, num_phases = num_phases, res_size = res_size)
    mp.set_inlet_outlet(inlets,outlets)

    return mp
