import numpy as np
from mpi4py import MPI
import time
import pmmoto


def my_function():

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    subdomains = [1,1,1] # Specifies how Domain is broken among rrocs
    nodes = [100,100,100] # Total Number of Nodes in Domain

    ## Ordering for Inlet/Outlet ( (-x,+x) , (-y,+y) , (-z,+z) )
    boundaries = [[0,0],[0,0],[0,0]] # 0: Nothing Assumed  1: Walls 2: Periodic
    dataReadBoundaries = [[0,0],[0,0],[0,0]] # 0: Nothing Assumed  1: Walls 2: Periodic
    inlet  = [[0,0],[0,0],[1,0]]
    outlet = [[0,0],[0,0],[0,1]]


    r_lookup_file = './testDomains/lammps/single_atom_rlookup.txt'
    r_lookup = pmmoto.io.read_r_lookup_file(r_lookup_file)

    lammps_file = './testDomains/lammps/lammps_single_atom.out'
    sphere_data,domain_data = pmmoto.io.read_lammps_atoms(lammps_file,r_lookup)

    sd = pmmoto.initialize(rank,size,subdomains,nodes,boundaries,inlet,outlet)
    pm = pmmoto.domain_generation.gen_pm_spheres_domain(sd,sphere_data,domain_data)

    edt = pmmoto.filters.calc_edt(sd,pm.grid)
    pmmoto.io.save_grid_data("dataOut/test_lammps_read_grid",sd,pm.grid,dist=edt)


if __name__ == "__main__":
    my_function()
    MPI.Finalize()
