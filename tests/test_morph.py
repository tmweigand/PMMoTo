import numpy as np
from mpi4py import MPI
import math
import time
import pmmoto
import sys
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision=50)

def my_function():

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    subdomains = [1,1,1] # Specifies how Domain is broken among procs
    nodes = [300,300,300] # Total Number of Nodes in Domain

    ## Ordering for Inlet/Outlet ( (-x,+x) , (-y,+y) , (-z,+z) )
    boundaries = [[2,2],[1,1],[0,0]] # 0: Nothing Assumed  1: Walls 2: Periodic
    inlet  = [[0,0],[0,0],[0,0]]
    outlet = [[0,0],[0,0],[0,0]]

    file = './testDomains/50pack.out'

    save_data = True

    sd = pmmoto.initialize(rank,size,subdomains,nodes,boundaries,inlet,outlet)
    sphere_data,domain_data = pmmoto.io.read_sphere_pack_xyzr_domain(file)
    pm = pmmoto.domain_generation.gen_pm_spheres_domain(sd,sphere_data,domain_data)

    edt = pmmoto.filters.calc_edt(sd,pm.grid)

    radius = sd.domain.voxel[0]*4 + sd.domain.voxel[0]*1.e-6

    # start_time = time.time()
    # morph_edt = pmmoto.morph_add(sd,pm.grid,radius = radius,fft = False)
    # end_time = time.time()
    # print(f"Morph Add Distance Runtime {end_time-start_time} seconds")

    # start_time = time.time()
    # morph_fft = pmmoto.morph_add(sd,pm.grid,radius = radius,fft = True)
    # end_time = time.time()
    # print(f"Morph Add FFT Runtime {end_time-start_time} seconds")

    # print(f"Add results equal: {np.array_equal(morph_edt, morph_fft)}")

    start_time = time.time()
    morph_edt = pmmoto.filters.closing(sd,pm.grid,radius = radius,fft = False)
    end_time = time.time()
    print(f"Morph Sub Distance Runtime {end_time-start_time} seconds")

    start_time = time.time()
    morph_fft = pmmoto.filters.closing(sd,pm.grid,radius = radius,fft = True)
    end_time = time.time()
    print(f"Morph Sub FFT Runtime {end_time-start_time} seconds")

    print(f"Sub results equal: {np.array_equal(morph_edt, morph_fft)}")

    if save_data:
        ### Save Grid Data where kwargs are used for saving other grid data (i.e. EDT, Medial Axis)
        pmmoto.io.save_grid_data("dataOut/test_morph",sd,pm.grid,morph_edt=morph_edt,morph_fft = morph_fft, edt = edt)


if __name__ == "__main__":
    my_function()
    MPI.Finalize()
