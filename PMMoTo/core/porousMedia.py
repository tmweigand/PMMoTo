import numpy as np
from mpi4py import MPI
from . import domainGeneration
from . import communication
from . import Orientation
comm = MPI.COMM_WORLD

class PorousMedia:
    """
    Porous media class 
    """
    def __init__(self,subdomain,domain):
        self.subdomain = subdomain
        self.domain = domain
        self.grid = None
        self.inlet = np.zeros([Orientation.num_faces],dtype = np.uint8)
        self.outlet = np.zeros([Orientation.num_faces],dtype = np.uint8)
        self.loop_info = np.zeros([Orientation.num_faces+1,3,2],dtype = np.int64)
        self.index_own_nodes = np.zeros([6],dtype = np.int64)
        self.pore_nodes = 0
        self.total_pore_nodes = np.zeros(1,dtype=np.uint64)

    def check_grid(self):
        """

        """
        if (np.sum(self.grid) == np.prod(self.subdomain.nodes)):
            print("This code requires at least 1 solid voxel in each subdomain. Please reorder processors!")
            communication.raiseError()

    def gen_pm_spheres(self,sphere_data):
        """
        """
        self.grid = domainGeneration.domainGen(self.subdomain.coords[0],self.subdomain.coords[1],self.subdomain.coords[2],sphere_data)
        self.check_grid()
        self.grid = communication.update_buffer(self.subdomain,self.grid)

    def gen_pm_verlet_spheres(self,sphere_data,verlet=[20,20,20]):
        """
        """
        self.grid = domainGeneration.domainGenVerlet(verlet,self.subdomain.coords[0],self.subdomain.coords[1],self.subdomain.coords[2],sphere_data)
        self.check_grid()
        self.grid = communication.update_buffer(self.subdomain,self.grid)

    def gen_pm_inkbottle(self):
        """
        """
        self.grid = domainGeneration.domainGenINK(self.subdomain.coords[0],self.subdomain.coords[1],self.subdomain.coords[2])
        self.check_grid()
        self.grid = communication.update_buffer(self.subdomain,self.grid)


    def set_inlet_outlet(self,res_size):
        """
        Determine inlet/outlet info and pad grid
        """
        inlet_size = np.zeros_like(self.inlet)
        outlet_size = np.zeros_like(self.outlet)

        for n in range(0,self.domain.dims):
            if (self.subdomain.boundary_ID[n*2] == 0):
                if self.domain.inlet[n][0]:
                    self.inlet[n*2] = True
                    inlet_size[n*2]  = res_size
                if self.domain.inlet[n][1]:
                    self.inlet[n*2+1] = True
                    inlet_size[n*2+1]  = res_size
                if self.domain.outlet[n][0]:
                    self.outlet[n*2] = True
                    outlet_size[n*2]  = res_size
                if self.domain.outlet[n][1]:
                    self.outlet[n*2+1] = True
                    outlet_size[n*2+1]  = res_size

        pad = np.zeros([self.domain.dims*2],dtype = np.int8)
        for f in range(0,Orientation.num_faces):
            pad[f] = inlet_size[f] + outlet_size[f]
        
        ### If Inlet/Outlet Res, Pad and Update XYZ
        if np.sum(pad) > 0:
            self.grid = np.pad(self.grid, ( (pad[0], pad[1]), (pad[2], pad[3]), (pad[4], pad[5]) ), 'constant', constant_values=1)
            self.subdomain.get_coordinates(pad)


    def set_wall_bcs(self):
        """
        If wall boundary conditions are specified, force solid on external boundaries
        """
        if self.subdomain.boundary_ID[0] == 1:
            self.grid[0,:,:] = 0
        if self.subdomain.boundary_ID[1] == 1:
            self.grid[-1,:,:] = 0
        if self.subdomain.boundary_ID[2] == 1:
            self.grid[:,0,:] = 0
        if self.subdomain.boundary_ID[3] == 1:
            self.grid[:,-1,:] = 0
        if self.subdomain.boundary_ID[4] == 1:
            self.grid[:,:,0] = 0
        if self.subdomain.boundary_ID[5] == 1:
            self.grid[:,:,-1] = 0

    def calc_porosity(self):
        own = self.subdomain.index_own_nodes
        own_grid =  self.grid[own[0]:own[1],
                             own[2]:own[3],
                             own[4]:own[5]]
        self.pore_nodes = np.sum(own_grid)
        comm.Allreduce( [self.pore_nodes, MPI.INT], [self.total_pore_nodes, MPI.INT], op = MPI.SUM )


def gen_pm(subdomain,data_format,sphere_data = None,res_size = 0):
    """
    """
    pm = PorousMedia(domain = subdomain.domain, subdomain = subdomain)

    if data_format == "Sphere":
        pm.gen_pm_spheres(sphere_data)
    if data_format == "SphereVerlet":
        pm.gen_pm_verlet_spheres(sphere_data)
    if data_format == "InkBotle":
        pm.gen_pm_inkbottle()
    pm.set_inlet_outlet(res_size)
    pm.set_wall_bcs()
    pm.loop_info = Orientation.get_loop_info(pm.grid,subdomain,pm.inlet,pm.outlet,res_size)
    pm.calc_porosity()

    return pm
