import os
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
import pmmoto

def test_repeated_call():
    """
    Test repeate calls to pm.grid
    """
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    subdomains = [1,1,1] # Specifies how Domain is broken among procs
    nodes = [50,50,50] # Total Number of Nodes in Domain

    ## Ordering for Inlet/Outlet ( (-x,+x) , (-y,+y) , (-z,+z) )
    boundaries = [[2,2],[1,1],[0,0]] # 0: Nothing Assumed  1: Walls 2: Periodic
    inlet  = [[0,0],[0,0],[0,0]]
    outlet = [[0,0],[0,0],[0,0]]

    file = './testDomains/50pack.out'

    sphere_data,domain_data = pmmoto.io.read_sphere_pack_xyzr_domain(file)
    # sphere_data = sphere_data*1.1

    sd = pmmoto.initialize(rank,size,subdomains,nodes,boundaries,inlet,outlet)
    pm = pmmoto.domain_generation.gen_pm_spheres_domain(sd,sphere_data,domain_data)
    pm_sum = np.sum(pm.grid)

    print(f'OG: Sum: {pm_sum} Domain Size {sd.domain.size_domain}')

    sphere_data = sphere_data*1.1
    pm_bigger = pmmoto.domain_generation.gen_pm_spheres_domain(sd,sphere_data,domain_data)
    pm_bigger_sum = np.sum(pm_bigger.grid)

    print(f'RE-DO: Sum: {pm_bigger_sum} Domain Size {sd.domain.size_domain}')

    sd_bigger = pmmoto.initialize(rank,size,subdomains,nodes,boundaries,inlet,outlet)
    pm_new = pmmoto.domain_generation.gen_pm_spheres_domain(sd_bigger,sphere_data,domain_data)
    pm_new_sum = np.sum(pm_new.grid)

    print(f'Re-Initialize: Sum: {pm_new_sum} Domain Size {sd_bigger.domain.size_domain}')


if __name__ == "__main__":
    test_repeated_call()
    MPI.Finalize()