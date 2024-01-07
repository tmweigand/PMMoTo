import numpy as np
from mpi4py import MPI
import math
import time
import PMMoTo
import sys
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision=50)

def my_function():

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    subDomains = [1,1,1] # Specifies how Domain is broken among procs
    nodes = [300,300,300] # Total Number of Nodes in Domain

    ## Ordering for Inlet/Outlet ( (-x,+x) , (-y,+y) , (-z,+z) )
    boundaries = [[2,2],[1,1],[0,0]] # 0: Nothing Assumed  1: Walls 2: Periodic
    inlet  = [[0,0],[0,0],[0,0]]
    outlet = [[0,0],[0,0],[0,0]]

    file = './testDomains/50pack.out'

    save_data = True

    domain,sDL,pML = PMMoTo.genDomainSubDomain(rank,size,subDomains,nodes,boundaries,inlet,outlet,"Sphere",file,PMMoTo.readPorousMediaXYZR)


    edt = PMMoTo.calcEDT(sDL,pML.grid)

    radius = domain.voxel[0]*4 + domain.voxel[0]*1.e-6

    # start_time = time.time()
    # morph_edt = PMMoTo.morph_add(sDL,pML.grid,radius = radius,fft = False)
    # end_time = time.time()
    # print(f"Morph Add Distance Runtime {end_time-start_time} seconds")

    # start_time = time.time()
    # morph_fft = PMMoTo.morph_add(sDL,pML.grid,radius = radius,fft = True)
    # end_time = time.time()
    # print(f"Morph Add FFT Runtime {end_time-start_time} seconds")

    # print(f"Add results equal: {np.array_equal(morph_edt, morph_fft)}")

    start_time = time.time()
    morph_edt = PMMoTo.closing(sDL,pML.grid,radius = radius,fft = False)
    end_time = time.time()
    print(f"Morph Sub Distance Runtime {end_time-start_time} seconds")

    start_time = time.time()
    morph_fft = PMMoTo.closing(sDL,pML.grid,radius = radius,fft = True)
    end_time = time.time()
    print(f"Morph Sub FFT Runtime {end_time-start_time} seconds")

    print(f"Sub results equal: {np.array_equal(morph_edt, morph_fft)}")

    if save_data:
        ### Save Grid Data where kwargs are used for saving other grid data (i.e. EDT, Medial Axis)
        PMMoTo.saveGridData("dataOut/test_morph",rank,domain,sDL,pML.grid,morph_edt=morph_edt,morph_fft = morph_fft, edt = edt)


if __name__ == "__main__":
    my_function()
    MPI.Finalize()
