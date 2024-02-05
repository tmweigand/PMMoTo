import numpy as np
from mpi4py import MPI
import time
import pmmoto


def my_function():

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    subdomains = [2,2,2] # Specifies how domain is broken among processes
    nodes = [100,100,100] # Total Number of Nodes in Domain

    ## Ordering for Inlet/Outlet ( (-x,+x) , (-y,+y) , (-z,+z) )
    boundaries = [[2,2],[2,2],[2,2]] # 0: Nothing Assumed  1: Walls 2: Periodic
    inlet  = [[0,0],[0,0],[0,0]]
    outlet = [[0,0],[0,0],[0,0]]

    file = './testDomains/test_periodic_spheres.out'

    sd = pmmoto.initialize(rank,size,subdomains,nodes,boundaries,inlet,outlet)
    sphere_data,domain_data = pmmoto.io.read_sphere_pack_xyzr_domain(file)
    pm = pmmoto.domain_generation.gen_pm_spheres_domain(sd,sphere_data,domain_data)

    # Save Grid Data where kwargs are used for saving other grid data (i.e. edt, Medial Axis)
    pmmoto.io.save_grid_data("dataOut/test_no_periodic_spheres",sd,pm.grid)

    pm = pmmoto.domain_generation.gen_pm_spheres_domain(
                                    sd,
                                    sphere_data,
                                    domain_data,
                                    add_periodic = True
                                    )

    # Save Grid Data where kwargs are used for saving other grid data (i.e. edt, Medial Axis)
    pmmoto.io.save_grid_data("dataOut/test_added_periodic_spheres",sd,pm.grid)

if __name__ == "__main__":
    my_function()
    MPI.Finalize()
