import numpy as np
from mpi4py import MPI
import time
import pmmoto
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def my_function():

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    subdomains = [2,2,2] # Specifies how Domain is broken among procs
    nodes = [500,500,500] # Total Number of Nodes in Domain, controls resolution

    ## Ordering for Inlet/Outlet ( (-x,+x) , (-y,+y) , (-z,+z) )
    boundaries = [[0,0],[0,0],[0,0]] # 0: Nothing Assumed  1: Walls 2: Periodic
    inlet  = [[0,0],[0,0],[0,0]]
    outlet = [[0,0],[0,0],[0,0]]

    r_lookup_file = './testDomains/lammps/PA.rLookup'
    r_lookup = pmmoto.io.read_r_lookup_file(r_lookup_file,water = True)

    lammps_file = './testDomains/lammps/membranedata.71005000.gz'
    sphere_data,domain_data = pmmoto.io.read_lammps_atoms(lammps_file,r_lookup)
    domain_data[2] = [-75.329262, 75.329262]

    sd = pmmoto.initialize(rank,size,subdomains,nodes,boundaries,inlet,outlet)
    pm = pmmoto.domain_generation.gen_pm_verlet_spheres(sd,sphere_data,domain_data,verlet=[10,10,10])

    print(sd.domain.voxel, 1.5/sd.domain.voxel)

    if sd.ID == 0:
        pm.grid[0:5,0:5,0:5] = 3

    pmmoto.io.save_grid_data("dataOut/test_lammps_read_grid",sd,pm.grid)

if __name__ == "__main__":
    my_function()
    MPI.Finalize()
